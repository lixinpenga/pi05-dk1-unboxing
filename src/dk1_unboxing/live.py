"""Interactive demonstration, supervised rollout and DAgger collection."""

import logging
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from .evaluation import Progress
from .recording import Episode, Recorder, write_json
from .rtc import AsyncRTC, Predictor, RTCSettings
from .scene import FPS, joint_dict, vector

logger = logging.getLogger(__name__)


@contextmanager
def keyboard():
    if not sys.stdin.isatty():
        raise RuntimeError("真机采集和部署需要 Linux 交互终端")
    with open("/dev/tty", encoding="utf-8") as terminal:
        fd = terminal.fileno()
        original = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:

            def poll():
                return (
                    os.read(fd, 1).decode("ascii", errors="ignore").lower()
                    if select.select([fd], [], [], 0)[0]
                    else ""
                )

            yield poll
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)


class Clock:
    def __init__(self):
        self.start = monotonic()
        self.next_tick = self.start
        self.overruns = 0

    def wait(self) -> None:
        delay = self.next_tick - monotonic()
        if delay > 0:
            sleep(delay)
        elif delay < -2 / FPS:
            raise TimeoutError("控制循环滞后超过两个周期")
        elif delay < -1 / FPS:
            self.overruns += 1
            self.next_tick = monotonic()
        self.next_tick += 1 / FPS


@contextmanager
def connected_teleop(teleop):
    try:
        teleop.connect()
        yield teleop
    finally:
        devices = (teleop.left_arm, teleop.right_arm) if hasattr(teleop, "left_arm") else (teleop,)
        for device in devices:
            if device.is_connected:
                device.disconnect()


def seed_teleop(teleop, state, command):
    if hasattr(teleop, "seed_qpos_from_obs"):
        target = state.copy()
        if command is not None:
            target[[6, 13]] = command[[6, 13]]
        teleop.seed_qpos_from_obs(joint_dict(target))


def check_start():
    if input("确认工作区清空、已就位并持有急停。输入 START 开始本回合：").strip() != "START":
        raise ValueError("已取消本回合")


def record_demo(robot, teleop, raw: Path, episode: Episode) -> Path:
    print("采集期间 e 保存，q 放弃，Ctrl+C 中止。先对齐主从臂。")
    check_start()
    robot.previous = None
    seed_teleop(teleop, robot.observe(monotonic()).state, None)
    with Recorder(raw, episode.episode_id) as recorder:
        with keyboard() as key:
            clock = Clock()
            while monotonic() - clock.start < 120:
                clock.wait()
                pressed = key()
                if pressed == "q":
                    raise RuntimeError("已放弃回合")
                if pressed == "e":
                    break
                obs = robot.observe(clock.start)
                executed = robot.send(vector(teleop.get_action()), obs.state)
                recorder.add(obs, executed, "expert")
            else:
                raise TimeoutError("示教回合超过 120 秒")
        while True:
            rating = input("质量评级 1–5，回车放弃：").strip()
            if not rating:
                raise RuntimeError("已放弃回合")
            if rating in {"1", "2", "3", "4", "5"}:
                break
        return recorder.finish(replace(episode, rating=int(rating)))


def rollout(
    robot,
    predictor: Predictor,
    settings: RTCSettings,
    episode: Episode,
    result_path: Path,
    raw: Path | None = None,
    teleop=None,
    checkpoint: str | None = None,
) -> dict:
    from contextlib import ExitStack

    if result_path.exists():
        raise FileExistsError(result_path)
    progress = Progress()
    source = "policy"
    fault = None
    p95 = None
    motion_started = False
    with ExitStack() as stack:
        rtc = stack.enter_context(AsyncRTC(predictor, settings))
        recorder = stack.enter_context(Recorder(raw, episode.episode_id)) if raw else None
        rtc.warmup(robot.observe(monotonic()))
        print("2 正确取件，3 开始右手持件，c 持件中断，h 持满一秒，4 入槽有偏，5 干净入槽")
        print("x 拿错/插错/顺序错误，s 卡死，q 中止。i 人工接管，p 交还策略。")
        check_start()
        robot.previous = None
        clock = Clock()
        try:
            with keyboard() as key:
                while progress.reason is None:
                    clock.wait()
                    obs = robot.observe(clock.start)
                    pressed = key()
                    progress.update(obs.elapsed_s, pressed)
                    if progress.reason:
                        break
                    if teleop and pressed == "i" and source == "policy":
                        rtc.invalidate()
                        seed_teleop(teleop, obs.state, robot.previous)
                        source = "intervention"
                    if teleop and pressed == "p" and source == "intervention":
                        rtc.invalidate()
                        source = "policy"
                    target = vector(teleop.get_action()) if source == "intervention" else rtc.step(obs)
                    if target is not None:
                        executed = robot.send(target, obs.state)
                        motion_started = True
                        if recorder:
                            recorder.add(obs, executed, source)
                    elif recorder and motion_started:
                        recorder.add(obs, robot.previous, "hold")
        except KeyboardInterrupt:
            progress.reason = "operator_abort"
        except Exception as exc:
            logger.exception("运行中止")
            progress.reason = "hardware_fault"
            fault = str(exc)

        verdict = progress.result()
        if (
            recorder
            and (recorder.rows or recorder.pending)
            and verdict["reason"] not in {"operator_abort", "hardware_fault"}
        ):
            recorder.finish(episode, verdict)
        if rtc.events:
            p95 = float(np.percentile([event["inference_ms"] for event in rtc.events], 95))
        result = {
            "id": episode.episode_id,
            "round": episode.round,
            "slot": episode.slot,
            "scene": episode.scene,
            "source_offset_cm": episode.source_offset_cm,
            "tray_offset_cm": episode.tray_offset_cm,
            "verdict": verdict,
            "rtc": rtc.events,
            "rtc_settings": asdict(settings),
            "checkpoint": checkpoint,
            "inference_p95_ms": p95,
            "fault": fault,
            "control_overruns": clock.overruns,
        }
        write_json(result_path, result)
    return result
