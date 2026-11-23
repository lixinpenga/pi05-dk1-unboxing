"""Asynchronous action-chunk playback on the 50 Hz control timeline."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

import numpy as np

from .recording import Tick
from .scene import ACTION_DIM

HORIZON = 50


@dataclass(frozen=True)
class RTCSettings:
    reinfer: int = 30
    blend: int = 16
    initial_delay: int = 7
    guided: bool = True

    def __post_init__(self):
        if not 1 <= self.reinfer <= HORIZON or not 0 <= self.blend <= HORIZON:
            raise ValueError("RTC 的 reinfer 必须为 1–50，blend 必须为 0–50")
        if not 0 <= self.initial_delay < self.reinfer:
            raise ValueError("推理延迟必须小于请求时的排队动作数")


PRESETS = {
    "baseline": RTCSettings(),
    "fast": RTCSettings(reinfer=50, blend=20),
    "unguided": RTCSettings(guided=False, blend=0),
}


class Predictor(Protocol):
    def __call__(self, obs: Tick, prefix: np.ndarray, delay: int, horizon: int) -> np.ndarray: ...


@dataclass
class Pending:
    future: Future
    request_tick: int
    request_time: float
    prefix_len: int
    predicted_delay: int
    generation: int


class AsyncRTC:
    def __init__(self, predictor: Predictor, settings: RTCSettings):
        self.predictor, self.settings = predictor, settings
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-inference")
        self.pending: Pending | None = None
        self.queue = np.empty((0, ACTION_DIM), dtype=np.float32)
        self.tick = 0
        self.generation = 0
        self.delay_estimate = float(settings.initial_delay)
        self.events: list[dict] = []
        self.last_send_time: float | None = None
        self.replan_bootstrap_time: float | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def invalidate(self):
        self.generation += 1
        self.queue = self.queue[:0]
        self.last_send_time = self.replan_bootstrap_time = None
        if self.pending is not None:
            self.pending.future.cancel()

    def warmup(self, obs: Tick, timeout: float = 30):
        empty = self.queue[:0]
        actions, _ = self.executor.submit(self._predict, obs, empty, 0).result(timeout=timeout)
        if self.settings.guided and self.settings.blend:
            self.executor.submit(
                self._predict,
                obs,
                actions,
                self.settings.initial_delay,
            ).result(timeout=timeout)

    def _predict(self, obs: Tick, prefix: np.ndarray, delay: int):
        horizon = min(self.settings.blend, len(prefix)) if self.settings.guided else 0
        actions = np.asarray(
            self.predictor(obs, prefix, delay if horizon else 0, horizon), dtype=np.float32
        )
        if actions.shape != (HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ValueError(f"模型须返回有限的 {HORIZON}×{ACTION_DIM} 动作块")
        return actions, monotonic()

    def step(self, obs: Tick) -> np.ndarray | None:
        self.tick += 1
        now = monotonic()
        self._receive(now)
        if self.replan_bootstrap_time is None:
            self.replan_bootstrap_time = now
        if self.pending is None and len(self.queue) <= self.settings.reinfer:
            prefix = self.queue.copy()
            estimate = round(self.delay_estimate)
            if 0 < len(prefix) <= estimate:
                raise RuntimeError("RTC 待执行动作少于延迟估计，请提高 reinfer 或降低推理延迟")
            delay = min(estimate, len(prefix))
            future = self.executor.submit(self._predict, obs, prefix, delay)
            self.pending = Pending(future, self.tick, now, len(prefix), delay, self.generation)
        if not len(self.queue):
            if self.last_send_time is not None:
                raise RuntimeError("RTC 动作队列耗尽，已停止下发新目标")
            if now - self.replan_bootstrap_time > 3:
                raise TimeoutError("RTC 首块动作三秒内未到达")
            return None
        action = self.queue[0]
        self.queue = self.queue[1:]
        self.last_send_time = now
        return action

    def _receive(self, now: float):
        pending = self.pending
        if pending is None or not pending.future.done():
            return
        self.pending = None
        if pending.generation != self.generation:
            # Consume exceptions from the obsolete request without using its results.
            if not pending.future.cancelled():
                pending.future.exception()
            return
        actions, finished = pending.future.result()
        lag = self.tick - pending.request_tick
        used = lag if pending.prefix_len else 0
        self.events.append(
            {
                "request_tick": pending.request_tick,
                "queued_at_request": pending.prefix_len,
                "predicted_delay_ticks": pending.predicted_delay,
                "measured_delay_ticks": lag,
                "inference_ms": (finished - pending.request_time) * 1000,
                "observed_arrival_ms": (now - pending.request_time) * 1000,
                "discarded_actions": used,
                "guided_prefix_end": min(pending.prefix_len, self.settings.blend)
                if self.settings.guided
                else 0,
            }
        )
        # A latency spike may still arrive exactly on the first tick with an empty old queue.
        if used > pending.prefix_len or used >= HORIZON:
            raise RuntimeError("RTC 新块到达过晚，无法延续动作时间轴")
        self.delay_estimate = min(HORIZON - 1, max(1, 0.7 * self.delay_estimate + 0.3 * lag))
        self.queue = actions[used:]
