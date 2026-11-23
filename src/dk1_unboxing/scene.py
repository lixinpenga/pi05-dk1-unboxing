from dataclasses import asdict, dataclass
from random import Random
from typing import Any

import numpy as np

FPS = 50
CAMERAS = ("left_wrist", "right_wrist", "top")
JOINTS = tuple(
    f"{arm}_{joint}.pos"
    for arm in ("left", "right")
    for joint in (*[f"joint_{i}" for i in range(1, 7)], "gripper")
)
ACTION_DIM = len(JOINTS)
TASK = "Take an actuator from the box and place it in the tray"
SCENES = ("desk_1", "desk_2", "desk_3", "desk_4", "lightbox")


@dataclass(frozen=True)
class Scene:
    source_rows: int = 5
    source_columns: int = 8
    tray_rows: int = 9
    tray_columns: int = 15
    jitter_cm: float = 11

    def __post_init__(self) -> None:
        if min(self.source_rows, self.source_columns, self.tray_rows, self.tray_columns) < 1:
            raise ValueError("格子行列数必须为正")
        if self.parts > self.tray_rows * self.tray_columns or self.jitter_cm < 0:
            raise ValueError("托盘容量或随机化范围无效")

    @property
    def parts(self) -> int:
        return self.source_rows * self.source_columns

    def slot(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.parts:
            raise ValueError(f"工件编号超出范围 0–{self.parts - 1}")
        row, col = divmod(index, self.source_columns)
        tray_row, tray_col = divmod(index, self.tray_columns)
        return {
            "index": index,
            "source": {"row": row, "column": self.source_columns - 1 - col},
            "tray": {"row": tray_row, "column": tray_col},
        }

    def plan(self, rounds: int, seed: int, diverse: bool) -> dict[str, Any]:
        if rounds < 1:
            raise ValueError("至少需要一轮")
        random = Random(seed)
        radius = self.jitter_cm / 2

        def offsets() -> list[float]:
            return [round(random.uniform(-radius, radius), 2) for _ in range(2)] if diverse else [0, 0]

        return {
            "scene": asdict(self),
            "seed": seed,
            "scene_name": "lightbox",
            "trials": [
                {
                    "round": round_index,
                    "source_offset_cm": offsets(),
                    "tray_offset_cm": offsets(),
                    "slots": [self.slot(i) for i in range(self.parts)],
                }
                for round_index in range(rounds)
            ],
        }


def vector(values: Any) -> np.ndarray:
    if isinstance(values, dict):
        values = [values[name] for name in JOINTS]
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        raise ValueError(f"关节向量必须包含 {ACTION_DIM} 个有限数值")
    return array


def joint_dict(values: np.ndarray) -> dict[str, float]:
    return dict(zip(JOINTS, map(float, values), strict=True))


def center_crop(frame: np.ndarray, size: int = 224) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("相机帧必须为 RGB uint8 HWC")
    height, width = frame.shape[:2]
    if min(height, width) < size:
        raise ValueError(f"相机帧小于 {size}×{size}")
    y, x = (height - size) // 2, (width - size) // 2
    return np.ascontiguousarray(frame[y : y + size, x : x + size])
