"""Pinned model artifacts and legacy checkpoint compatibility."""

import json
from pathlib import Path

LEGACY_FIELDS = {
    "image_resize_to",
    "drop_unused_lm_head",
    "scheduler_type",
    "scheduler_decay_ratio",
    "scheduler_max_decay_steps",
    "velocity_debias_sidecar",
    "velocity_debias_max_ratio",
}
# Some checkpoints omit the language heads, which the robot flow-matching path does not use.
UNUSED_HEADS = {
    "model.paligemma_with_expert.paligemma.lm_head.weight",
    "model.paligemma_with_expert.gemma_expert.lm_head.weight",
}


def fetch_model(
    repo: str, root: Path | None, metadata_only: bool = False, *, revision: str | None = None
) -> Path:
    from huggingface_hub import snapshot_download

    paths = ["config.json", "train_config.json", "policy_*.json", "policy_*processor*.safetensors"]
    if not metadata_only:
        paths.append("model.safetensors")
    return Path(
        snapshot_download(
            repo,
            revision=revision,
            local_dir=root,
            allow_patterns=paths,
        )
    )


def load_pi05_config(path: Path, device: str):
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    values = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if values.pop("type", None) != "pi05":
        raise ValueError("需要 π0.5 checkpoint")
    if values.get("velocity_debias_sidecar"):
        raise ValueError("此策略带额外训练权重，此版本暂不支持加载")
    for field in LEGACY_FIELDS:
        values.pop(field, None)
    values.update(device=device, compile_model=False, trust_remote_code=False, pretrained_path=str(path))
    return draccus.decode(PI05Config, values)


def verified_from_pretrained(cls, pretrained_name_or_path, *, config=None, revision=None, **kwargs):
    """Fail on missing/unexpected action-path weights before any physical command."""
    from huggingface_hub import hf_hub_download
    from lerobot.configs import PreTrainedConfig
    from safetensors.torch import load_file

    path = Path(pretrained_name_or_path)
    weight = (
        path / "model.safetensors"
        if path.is_dir()
        else Path(
            hf_hub_download(
                str(pretrained_name_or_path),
                "model.safetensors",
                revision=revision,
            )
        )
    )
    if config is None:
        config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, revision=revision)
    model = cls(config, **kwargs)
    state = model._convert_openpi_state_dict(load_file(weight))
    errors = model.load_state_dict(state, strict=False)
    missing = set(errors.missing_keys) - UNUSED_HEADS
    unexpected = set(errors.unexpected_keys) - UNUSED_HEADS
    if missing or unexpected:
        raise RuntimeError(f"π0.5 权重键不匹配。缺少 {sorted(missing)}，多余 {sorted(unexpected)}")
    return model.to(config.device).eval()


def load_processor_stats(path: Path) -> dict:
    from safetensors.torch import load_file

    from .scene import ACTION_DIM

    def read_pipeline(name: str, registry_name: str) -> dict:
        pipeline = json.loads((path / f"policy_{name}.json").read_text(encoding="utf-8"))
        state_files = [
            step["state_file"]
            for step in pipeline["steps"]
            if step.get("registry_name") == registry_name
        ]
        if len(state_files) != 1 or Path(state_files[0]).name != state_files[0]:
            raise ValueError(f"checkpoint 的 {name} 缺少可识别的归一化状态文件")
        out = {}
        for flat_key, tensor in load_file(path / state_files[0]).items():
            field, stat = flat_key.rsplit(".", 1)
            out.setdefault(field, {})[stat] = tensor
        return out

    stats = read_pipeline("preprocessor", "normalizer_processor")
    post = read_pipeline("postprocessor", "unnormalizer_processor")
    for field in ("observation.state", "action"):
        for quantile in ("q01", "q99"):
            value = stats.get(field, {}).get(quantile)
            if value is None or tuple(value.shape) != (ACTION_DIM,) or not value.isfinite().all():
                raise ValueError(f"checkpoint 缺少有效 {field}.{quantile} 统计")
        if (stats[field]["q01"] > stats[field]["q99"]).any():
            raise ValueError(f"checkpoint 的 {field} 分位数颠倒")
    for key in ("q01", "q99"):
        if not stats["action"][key].equal(post.get("action", {}).get(key, stats["action"][key][:0])):
            raise ValueError("checkpoint 的动作正向和逆向归一化统计不同")
    return stats
