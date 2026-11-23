"""Exercise the data and control contracts without a robot or model weights."""

import json
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from .evaluation import Progress, summarize
from .recording import Episode, Recorder, Tick, write_json
from .rtc import HORIZON, PRESETS, AsyncRTC
from .scene import ACTION_DIM, CAMERAS, FPS, Scene


def offline_smoke(root: Path, lerobot: bool) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    raw_path = root / "raw"
    images = {name: np.full((480, 640, 3), i * 70, dtype=np.uint8) for i, name in enumerate(CAMERAS)}
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    timings = dict.fromkeys(CAMERAS, 0.0)
    seq = dict.fromkeys(CAMERAS, 0)
    score = Progress()
    for timestamp, key in [(0.2, "2"), (0.4, "3"), (1.5, "h"), (2, "5")]:
        score.update(timestamp, key)
    for slot, (label, kind, rating) in enumerate(
        [("hq", "hq", 5), ("drop", "demo", 2), ("mixed", "dagger", 0), ("success", "dagger", 0)]
    ):
        episode = Episode(label, kind, "lightbox", "mock", "mock", slot, rating)
        with Recorder(raw_path, label, max_pending=70) as recorder:
            for tick in range(60):
                target = zero.copy()
                target[0] = min(47, max(0, tick - 5)) * 0.01
                source = "expert" if kind != "dagger" else "policy"
                if label == "mixed" and 10 <= tick < 20:
                    source = "intervention"
                recorder.add(Tick(tick / FPS, target, images, timings, seq), target, source)
            recorder.finish(episode, score.result() if kind == "dagger" else None)

    def mock_predict(obs, prefix, delay, horizon):
        sleep(0.12)
        actions = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
        if horizon:
            actions[:horizon] = prefix[:horizon]
        return actions

    with AsyncRTC(mock_predict, PRESETS["fast"]) as rtc:
        start = monotonic()
        for index in range(32):
            sleep(max(0, start + index / FPS - monotonic()))
            rtc.step(Tick(monotonic() - start, zero, images, timings, seq))
        if not any(event["discarded_actions"] > 0 for event in rtc.events):
            raise RuntimeError("RTC 没有完成有重叠的新块切换")
        events = rtc.events
    plan = Scene().plan(1, 42, False)
    result = {"round": 0, "slot": 0, "verdict": score.result()}
    write_json(root / "plan.json", plan)
    write_json(root / "result.json", result)
    report = {
        "synthetic": True,
        "raw_episodes": 4,
        "rtc": events,
        "evaluation": summarize(plan, [result]),
    }
    if lerobot:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from .dataset import export

        path, repo = root / "lerobot", "local/dk1-smoke"
        report["export"] = export(raw_path, path, repo, "dagger", 0.004, 5)
        dataset = LeRobotDataset(
            repo,
            root=path,
            video_backend="pyav",
            delta_timestamps={"action": [i / FPS for i in range(HORIZON)]},
        )
        example = dataset[0]
        assert tuple(example["action"].shape) == (HORIZON, ACTION_DIM)
        assert all(
            tuple(example[f"observation.images.{name}"].shape) == (3, 224, 224) for name in CAMERAS
        )
        manifest = json.loads((path / "meta/unboxing_manifest.json").read_text())
        first_intervention = manifest[0]["end_exclusive"] - manifest[0]["start"]
        assert dataset[first_intervention]["action_is_pad"][-1].item()
        report["training_item_shape"] = list(example["action"].shape)
    write_json(root / "report.json", report)
    return report
