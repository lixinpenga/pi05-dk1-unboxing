"""Stage scoring and round-aware reporting."""

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass
class Progress:
    timeout_s: float = 45
    stalled_s: float = 10
    stage: int = 1
    last_progress_s: float = 0
    receiver_since: float | None = None
    reason: str | None = None
    events: list[dict] = field(default_factory=list)

    def update(self, now: float, key: str = "") -> None:
        if self.reason:
            return
        if now >= self.timeout_s:
            self.reason = "time_limit"
            return
        if key in {"x", "q", "s"}:
            self.reason = {"x": "wrong_order", "q": "operator_abort", "s": "visibly_stuck"}[key]
            if key == "x":
                self.stage = 1
            self.events.append({"t_s": now, "event": self.reason})
            return
        if key == "2" and self.stage == 1:
            self._advance(2, now)
        if key == "3" and self.stage == 2:
            self.receiver_since = now
            self.events.append({"t_s": now, "event": "receiver_hold_start"})
        if key == "c" and self.stage == 2 and self.receiver_since is not None:
            self.receiver_since = None
            self.events.append({"t_s": now, "event": "receiver_hold_lost"})
        if (
            key == "h"
            and self.stage == 2
            and self.receiver_since is not None
            and now - self.receiver_since >= 1
        ):
            self._advance(3, now)
        if key == "4" and self.stage == 3:
            self._advance(4, now)
        if key == "5" and self.stage >= 3:
            if self.stage == 3:
                self._advance(4, now)
            self._advance(5, now)
            self.reason = "success"
        if self.reason is None and now - self.last_progress_s >= self.stalled_s:
            self.reason = "no_progress"

    def _advance(self, stage: int, now: float):
        self.stage = stage
        self.last_progress_s = now
        self.events.append({"t_s": now, "stage": stage})

    def result(self) -> dict:
        return {"stage": self.stage, "reason": self.reason, "events": self.events}


def wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(plan: dict, results: list[dict]) -> dict[str, Any]:
    planned = {(row["round"], slot["index"]) for row in plan["trials"] for slot in row["slots"]}
    stages = {i: 0 for i in range(1, 6)}
    seen = set()
    excluded = []
    for item in results:
        key = (item["round"], item["slot"])
        if key in seen or key not in planned:
            raise ValueError(f"重复或不属于此评测计划的回合：{key}")
        seen.add(key)
        verdict = item["verdict"]
        stage = verdict["stage"]
        if stage not in stages or (stage == 5) != (verdict["reason"] == "success"):
            raise ValueError(f"回合阶段或成功标记无效：{key}")
        if verdict["reason"] in {"operator_abort", "hardware_fault"}:
            excluded.append(list(key))
        else:
            stages[stage] += 1
    n = sum(stages.values())
    return {
        "planned": len(planned),
        "evaluated": n,
        "missing": sorted(planned - seen),
        "excluded": excluded,
        "stage_counts": stages,
        "success_rate": stages[5] / n if n else None,
        "wilson95": wilson(stages[5], n) if n else None,
    }
