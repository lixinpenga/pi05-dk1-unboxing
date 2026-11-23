"""Optional integration checks when the pinned training dependencies are installed."""

from pathlib import Path

import pytest

from dk1_unboxing.training import Recipe, build_train_args


def test_cli_arguments_parse_with_lerobot(tmp_path):
    pytest.importorskip("lerobot")
    import draccus
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    from dk1_unboxing.train_entry import QuadraticWSD

    recipe = Recipe("hq", 20, 4, 2, 1, 5e-5, 4, warmup=2, eval_every=0, workers=0)
    args = build_train_args(
        recipe, tmp_path / "run", init="weights/initial", root=Path("data/local"), device="cpu"
    )
    cfg = draccus.parse(TrainPipelineConfig, args=args)
    cfg.validate()
    assert isinstance(cfg.policy, PI05Config)
    assert cfg.policy.dtype == torch.float32
    assert isinstance(cfg.scheduler, QuadraticWSD)
    assert cfg.scheduler.num_warmup_steps == 4 and cfg.scheduler.decay == 8
    assert cfg.steps == 40 and cfg.accelerator.gradient_accumulation.steps == 2
