"""Episode holdout and offline action-chunk evaluation."""

import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from random import Random

import numpy as np

from .dataset import inspect
from .recording import Tick
from .rtc import HORIZON, PRESETS
from .scene import CAMERAS, FPS, vector


def episode_split(total: int, root: Path | None) -> tuple[list[int], list[int]]:
    if total < 2:
        raise ValueError("离线验证至少需要两个回合")
    groups = [[index] for index in range(total)]
    manifest = root / "meta/unboxing_manifest.json" if root else None
    if manifest and manifest.is_file():
        group_ids = defaultdict(list)
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            group_ids[item["raw_file"]].append(item["dataset_episode"])
        groups = list(group_ids.values())
        if len(groups) < 2 or sorted(i for group in groups for i in group) != list(range(total)):
            raise ValueError("验证需要至少两个有效的原始回合，且导出清单应覆盖所有数据片段")
    Random(42).shuffle(groups)
    train, valid = [], []
    desired = ceil(total * 0.05)
    while len(valid) < desired and len(groups) > 1:
        valid.extend(groups.pop())
    for group in groups:
        train.extend(group)
    return train, valid


def estimate_action_error(
    checkpoint: Path,
    repo: str,
    root: Path | None,
    revision: str | None,
    max_anchors: int,
    device: str,
) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .policy import Pi05Predictor

    if max_anchors < 1:
        raise ValueError("max-anchors 需为正")
    info = inspect(repo, root, revision)
    _, valid = episode_split(info["total_episodes"], root)
    dataset = LeRobotDataset(
        repo,
        root=root,
        episodes=valid,
        revision=revision,
        video_backend="pyav",
        return_uint8=True,
        delta_timestamps={"action": [i / FPS for i in range(HORIZON)]},
    )
    if len(dataset) == 0:
        raise ValueError("验证集没有帧")
    torch.manual_seed(42)
    total_norm, steps = 0.0, 0
    predictors = {}
    anchors = np.linspace(0, len(dataset) - 1, min(max_anchors, len(dataset)), dtype=int)
    for index in anchors:
        item = dataset[int(index)]
        task = item["task"]
        if task not in predictors:
            predictors[task] = Pi05Predictor(checkpoint, device, task, PRESETS["unguided"])
        images = {name: item[f"observation.images.{name}"].permute(1, 2, 0).numpy() for name in CAMERAS}
        zero = dict.fromkeys(CAMERAS, 0)
        obs = Tick(0, vector(item["observation.state"].numpy()), images, zero, zero)
        future = predictors[task](obs, np.empty((0, len(obs.state)), dtype=np.float32), 0, 0)
        label = item["action"].numpy()
        valid_steps = ~item["action_is_pad"].numpy()
        total_norm += float(np.linalg.norm(future[valid_steps] - label[valid_steps], axis=1).sum())
        steps += int(valid_steps.sum())
    return {
        "metric": "mean_action_l2_per_valid_step",
        "action_l2": total_norm / steps,
        "anchors": len(anchors),
        "valid_steps": steps,
        "held_out_episodes": len(valid),
        "seed": 42,
    }
