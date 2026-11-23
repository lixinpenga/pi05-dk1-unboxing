from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dk1_unboxing.evaluation import Progress, summarize, wilson
from dk1_unboxing.hardware import Limits
from dk1_unboxing.recording import Tick, select_segments, write_json
from dk1_unboxing.rtc import HORIZON, PRESETS, AsyncRTC, Pending
from dk1_unboxing.scene import ACTION_DIM, Scene, center_crop
from dk1_unboxing.training import Recipe, build_train_args, wsd_factor
from dk1_unboxing.validation import episode_split


def test_fixed_source_and_tray_order():
    scene = Scene()
    assert scene.parts == 40
    assert scene.slot(0) == {
        "index": 0,
        "source": {"row": 0, "column": 7},
        "tray": {"row": 0, "column": 0},
    }
    assert scene.slot(15)["tray"] == {"row": 1, "column": 0}
    assert scene.slot(39)["source"] == {"row": 4, "column": 0}
    assert Scene().plan(6, 42, True) == scene.plan(6, 42, True)
    with pytest.raises(ValueError):
        scene.slot(40)


def test_crop_matches_center_pixels():
    frame = np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert np.array_equal(center_crop(frame), frame[128:352, 208:432])


def test_rtc_uses_new_chunk_at_the_measured_offset():
    old = np.zeros((30, ACTION_DIM), dtype=np.float32)
    new = np.repeat(np.arange(HORIZON, dtype=np.float32)[:, None], ACTION_DIM, axis=1)
    with AsyncRTC(lambda *_: new, PRESETS["baseline"]) as rtc:
        completed = Future()
        completed.set_result((new, 0.13))
        rtc.queue, rtc.pending, rtc.tick = old[7:], Pending(completed, 10, 0, 30, 7, 0), 17
        rtc._receive(0.14)
        assert np.array_equal(rtc.queue[0], new[7])
        assert len(rtc.queue) == 43
        assert rtc.events[-1]["guided_prefix_end"] == 16


def test_inflight_rtc_result_cannot_survive_intervention():
    new = np.ones((HORIZON, ACTION_DIM), dtype=np.float32)
    with AsyncRTC(lambda *_: new, PRESETS["fast"]) as rtc:
        working = Future()
        working.set_running_or_notify_cancel()
        rtc.pending = Pending(working, 10, 0, 50, 7, 0)
        rtc.invalidate()
        working.set_result((new, 0.13))
        rtc.tick = 17
        rtc._receive(0.14)
        assert len(rtc.queue) == 0
        assert rtc.pending is None


def test_rtc_stops_when_old_chunk_is_exhausted():
    obs = Tick(0, np.zeros(ACTION_DIM), {}, {}, {})
    with AsyncRTC(lambda *_: None, PRESETS["baseline"]) as rtc:
        rtc.last_send_time = 0
        rtc.pending = Pending(Future(), 0, 0, 30, 7, 0)
        with pytest.raises(RuntimeError, match="队列耗尽"):
            rtc.step(obs)


def make_dagger_episode(sources, stage=5):
    rows = []
    for i, source in enumerate(sources):
        vector = [i * 0.01] * ACTION_DIM
        rows.append({"source": source, "state": vector, "action": vector})
    return {"episode": {"kind": "dagger", "rating": 0}, "ticks": rows, "verdict": {"stage": stage}}


def test_interventions_are_kept_separate_from_failed_policy_data():
    data = make_dagger_episode(
        ["policy", "intervention", "intervention", "hold", "intervention", "policy"]
    )
    assert select_segments(data, "dagger") == [slice(1, 3), slice(4, 5)]
    assert select_segments(data, "baseline") == []
    assert select_segments(make_dagger_episode(["policy"] * 10), "dagger") == [slice(0, 10)]
    assert select_segments(make_dagger_episode(["policy"] * 10, 4), "dagger") == []


def test_progress_requires_correct_order_and_one_second_hold():
    grade = Progress()
    for dt, key in [(0.1, "5"), (0.5, "2"), (0.6, "3"), (1.3, "h"), (1.4, "c"), (1.8, "h")]:
        grade.update(dt, key)
    assert grade.stage == 2
    for dt, key in [(2, "3"), (3.1, "h"), (3.2, "4"), (4, "5")]:
        grade.update(dt, key)
    assert grade.stage == 5 and grade.reason == "success"

    wrong = Progress()
    wrong.update(1, "2")
    wrong.update(2, "x")
    assert wrong.stage == 1 and wrong.reason == "wrong_order"


def test_report_denominator_includes_failures_and_tracks_exclusions():
    plan = Scene().plan(1, 42, False)
    results = [
        {"round": 0, "slot": i, "verdict": {"stage": stage, "reason": why}}
        for i, (stage, why) in enumerate([(5, "success"), (2, "no_progress"), (1, "hardware_fault")])
    ]
    report = summarize(plan, results)
    assert report["evaluated"] == 2
    assert report["excluded"] == [[0, 2]]
    assert report["success_rate"] == 0.5
    assert report["wilson95"] == wilson(1, 2)
    with pytest.raises(ValueError):
        summarize(plan, results + results[:1])


def test_grouped_holdout_does_not_split_a_raw_episode(tmp_path):
    group = ["a"] * 3 + ["b"] * 3 + ["c"] * 4
    write_json(
        tmp_path / "meta/unboxing_manifest.json",
        [{"raw_file": original, "dataset_episode": index} for index, original in enumerate(group)],
    )
    train, valid = episode_split(10, tmp_path)
    assert sorted(train + valid) == list(range(10))
    assert {group[i] for i in train}.isdisjoint({group[i] for i in valid})
    assert episode_split(198, None) == episode_split(198, None)
    assert len(episode_split(198, None)[1]) == 10


def test_limits_check_both_target_jump_and_position():
    limits = Limits(
        {
            "lower": [0.0] * ACTION_DIM,
            "upper": [1.0] * ACTION_DIM,
            "max_jump": [0.2] * ACTION_DIM,
            "max_tracking_error": [0.4] * ACTION_DIM,
        }
    )
    start = np.zeros(ACTION_DIM)
    assert np.allclose(limits.check(start + 0.1, start, start), 0.1)
    with pytest.raises(RuntimeError, match="left_joint_1"):
        limits.check(start + 0.3, start, start)
    with pytest.raises(RuntimeError, match="left_joint_1"):
        limits.check(start + 0.5, start, None)


def test_training_units_and_quadratic_floor():
    recipe = Recipe("local/dk1", 50_000, 256, 8, 4, 5e-5, 10_000)
    args = build_train_args(recipe, Path("outputs/test"), init="weights/initial")
    assert "--dataset.repo_id=local/dk1" in args
    assert "--policy.pretrained_path=weights/initial" in args
    with pytest.raises(ValueError, match="初始 checkpoint"):
        build_train_args(recipe, Path("outputs/test"), init="")
    assert recipe.accumulation == 8
    assert "--steps=400000" in args
    assert "--scheduler.num_warmup_steps=12000" in args
    assert "--scheduler.decay=80000" in args
    assert "--save_freq=80000" in args
    assert wsd_factor(0, 50_000, 1500, 10_000, 0.05) == 0.05
    assert wsd_factor(40_000, 50_000, 1500, 10_000, 0.05) == 1.0
    assert wsd_factor(45_000, 50_000, 1500, 10_000, 0.05) == pytest.approx(0.2875)
    assert wsd_factor(50_000, 50_000, 1500, 10_000, 0.05) == 0.05
    with pytest.raises(ValueError, match="EBS"):
        replace(recipe, microbatch=5)
