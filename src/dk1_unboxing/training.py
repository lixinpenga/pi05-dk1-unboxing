"""Translate optimizer-update recipes into the pinned LeRobot CLI."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def wsd_factor(step: int, total: int, warmup: int, decay: int, minimum: float = 0.0) -> float:
    if not 0 <= minimum <= 1 or min(warmup, decay) < 0 or warmup + decay > total:
        raise ValueError("WSD 阶段或学习率下限无效")
    if step < warmup:
        factor = step / warmup
    elif decay and step >= total - decay:
        factor = (1 - min(1, (step - (total - decay)) / decay)) ** 2
    else:
        factor = 1.0
    return minimum + (1 - minimum) * factor


@dataclass(frozen=True)
class Recipe:
    dataset: str
    updates: int
    ebs: int
    microbatch: int
    gpus: int
    lr: float
    decay: int
    warmup: int = 1500
    min_ratio: float = 0.05
    save_every: int = 10000
    eval_every: int = 10_000
    seed: int = 1000
    workers: int = 8
    lora_rank: int = 0
    lora_alpha: int = 32
    augmentation: bool = False
    relative: bool = False
    compile_model: bool = False

    @property
    def accumulation(self) -> int:
        denominator = self.microbatch * self.gpus
        if denominator < 1 or self.ebs < 1 or self.ebs % denominator:
            raise ValueError("EBS 必须能被 GPU 数 × 每卡 microbatch 整除")
        return self.ebs // denominator

    def __post_init__(self):
        if min(self.microbatch, self.gpus, self.updates, self.lr, self.lora_alpha) <= 0:
            raise ValueError("训练步数、学习率、设备、microbatch 和 alpha 必须大于零")
        if min(self.lora_rank, self.save_every, self.eval_every, self.workers) < 0:
            raise ValueError("LoRA rank、保存和验证间隔、worker 数量不得为负")
        _ = self.accumulation
        wsd_factor(0, self.updates, self.warmup, self.decay, self.min_ratio)


def build_train_args(
    recipe: Recipe,
    out: Path,
    init: str,
    root: Path | None = None,
    revision: str | None = None,
    device: str = "cuda",
    init_revision: str | None = None,
    episodes: list[int] | None = None,
    eval_fraction: float | None = None,
) -> list[str]:
    if not init.strip():
        raise ValueError("请通过 --init 提供初始 checkpoint")
    if recipe.relative and root is None:
        raise ValueError("相对动作实验需要带相对动作归一化统计的本地数据根目录")
    ga = recipe.accumulation
    settings = {
        "dataset.repo_id": recipe.dataset,
        "dataset.eval_split": (eval_fraction or 0.05) if recipe.eval_every else 0.0,
        "dataset.video_backend": "pyav",
        "dataset.image_transforms.enable": recipe.augmentation,
        "output_dir": str(out),
        "seed": recipe.seed,
        "steps": recipe.updates * ga,
        "batch_size": recipe.microbatch,
        "num_workers": recipe.workers,
        "accelerator.gradient_accumulation.steps": ga,
        "policy.type": "pi05",
        "policy.pretrained_path": init,
        "policy.chunk_size": 50,
        "policy.n_action_steps": 50,
        "policy.n_obs_steps": 1,
        "policy.num_inference_steps": 10,
        "policy.dtype": "bfloat16" if device == "cuda" else "float32",
        "policy.device": device,
        "policy.gradient_checkpointing": True,
        "policy.compile_model": recipe.compile_model,
        "policy.compile_mode": "default",
        "policy.tokenizer_max_length": 200,
        "policy.image_resolution": [224, 224],
        "policy.freeze_vision_encoder": False,
        "policy.train_expert_only": False,
        "policy.use_relative_actions": recipe.relative,
        "policy.push_to_hub": False,
        "use_policy_training_preset": False,
        "optimizer.type": "adamw",
        "optimizer.lr": recipe.lr,
        "optimizer.betas": [0.9, 0.95],
        "optimizer.eps": 1e-8,
        "optimizer.weight_decay": 0.01,
        "optimizer.grad_clip_norm": 1.0,
        "scheduler.type": "quadratic_wsd",
        "scheduler.num_warmup_steps": recipe.warmup * ga,
        "scheduler.decay": recipe.decay * ga,
        "scheduler.min_ratio": recipe.min_ratio,
        "save_freq": recipe.save_every * ga,
        "eval_steps": recipe.eval_every * ga,
        "max_eval_samples": recipe.microbatch * recipe.gpus * 16,
        "env_eval_freq": 0,
        "log_freq": 50 * ga,
    }
    if device == "cuda":
        settings["accelerator.mixed_precision"] = "bf16"
    if episodes is not None:
        settings["dataset.episodes"] = episodes
    if root:
        settings["dataset.root"] = str(root)
    if revision:
        settings["dataset.revision"] = revision
    if init_revision:
        settings["policy.pretrained_revision"] = init_revision
    if recipe.lora_rank:
        settings.update(
            {
                "peft.method_type": "LORA",
                "peft.target_modules": "all-linear",
                "peft.r": recipe.lora_rank,
                "peft.lora_alpha": recipe.lora_alpha,
            }
        )
    settings["dataset.streaming"] = False
    return [
        f"--{key}={json.dumps(value) if isinstance(value, (bool, list)) else value}"
        for key, value in settings.items()
    ]


def launch_command(gpus: int, args: list[str]) -> list[str]:
    if gpus == 1:
        return [sys.executable, "-m", "dk1_unboxing.train_entry", *args]
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        str(gpus),
        "--num_machines",
        "1",
        "-m",
        "dk1_unboxing.train_entry",
        *args,
    ]


def launch(command: list[str]) -> None:
    os.execv(command[0], command)
