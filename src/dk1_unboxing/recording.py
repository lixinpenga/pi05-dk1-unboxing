"""Raw per-episode archive and the deterministic preprocessing rules."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

from .scene import CAMERAS, FPS, TASK, center_crop, vector

SOURCES = {"expert", "policy", "intervention", "hold"}


@dataclass(frozen=True)
class Tick:
    elapsed_s: float
    state: np.ndarray
    images: dict[str, np.ndarray]
    camera_age_s: dict[str, float]
    camera_seq: dict[str, int]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    kind: str
    scene: str
    operator: str
    teleop: str
    slot: int
    # For a mixed episode, the stage-5 verdict is gathered by the evaluator.
    rating: int
    task: str = TASK
    round: int = 0
    source_offset_cm: tuple[float, float] = (0, 0)
    tray_offset_cm: tuple[float, float] = (0, 0)
    fps: int = FPS


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Recorder:
    """Keep JPEG encoding off the control thread with a bounded backlog."""

    def __init__(self, root: Path, episode_id: str, max_pending: int = 20):
        root.mkdir(parents=True, exist_ok=True)
        if not episode_id or Path(episode_id).name != episode_id or episode_id.startswith("."):
            raise ValueError("episode_id 必须为单个路径名称")
        self.path = root / f"{episode_id}.zip"
        if self.path.exists():
            raise FileExistsError(self.path)
        self.temporary = root / f".{episode_id}.{uuid4().hex}.partial"
        self.archive = ZipFile(self.temporary, "w", compression=ZIP_STORED)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="record")
        self.pending = []
        self.max_pending = max_pending
        self.previous: dict[str, tuple[int, str]] = {}
        self.rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.archive.close()
        self.temporary.unlink(missing_ok=True)

    def add(self, tick: Tick, action: np.ndarray, source: str):
        if source not in SOURCES:
            raise ValueError(source)
        while self.pending and self.pending[0].done():
            self.pending.pop(0).result()
        if len(self.pending) >= self.max_pending:
            raise RuntimeError("记录线程积压。该回合已中止，请检查 JPEG 编码和磁盘速度")
        self.pending.append(self.pool.submit(self._write_tick, tick, vector(action).copy(), source))

    def _write_tick(self, tick: Tick, action: np.ndarray, source: str):
        index = len(self.rows)
        paths = {}
        for name in CAMERAS:
            seq = tick.camera_seq[name]
            if name in self.previous and self.previous[name][0] == seq:
                paths[name] = self.previous[name][1]
                continue
            path = f"{name}/{index:06d}.jpg"
            buffer = BytesIO()
            Image.fromarray(tick.images[name]).save(buffer, format="JPEG", quality=95, subsampling=0)
            self.archive.writestr(path, buffer.getvalue())
            self.previous[name] = (seq, path)
            paths[name] = path
        self.rows.append(
            {
                "state": vector(tick.state).tolist(),
                "action": action.tolist(),
                "elapsed_s": tick.elapsed_s,
                "camera_age_s": tick.camera_age_s,
                "camera_seq": tick.camera_seq,
                "images": paths,
                "source": source,
            }
        )

    def finish(self, episode: Episode, verdict: dict | None = None) -> Path:
        for future in self.pending:
            future.result()
        if not self.rows:
            raise ValueError("回合为空")
        self.archive.writestr(
            "episode.json",
            json.dumps(
                {"schema": 1, "episode": asdict(episode), "verdict": verdict, "ticks": self.rows}
            ),
        )
        self.archive.close()
        # Hard link publishes atomically and fails if another run already created the target.
        self.path.hardlink_to(self.temporary)
        return self.path


def read_episode(archive: ZipFile) -> dict:
    data = json.loads(archive.read("episode.json"))
    rows = data["ticks"]
    if data["schema"] != 1 or not rows or data["episode"]["fps"] != FPS:
        raise ValueError("需要非空的 schema=1、50 Hz 回合")
    timing = np.asarray([row["elapsed_s"] for row in rows], dtype=float)
    if not np.isfinite(timing).all() or timing[0] < 0 or (np.diff(timing) <= 0).any():
        raise ValueError("控制时间戳必须有限且严格递增")
    if (np.diff(timing) > 2 / FPS + 0.005).any():
        raise ValueError("回合中有超过两周期的断档，请先检查原始日志")
    for row in rows:
        if row["source"] not in SOURCES:
            raise ValueError(f"未知数据来源 {row['source']}")
        vector(row["state"])
        vector(row["action"])
    return data


def trim_idle(rows: list[dict], threshold: float, margin: int) -> tuple[int, int]:
    a = np.stack([vector(row["action"]) for row in rows])
    q = np.stack([vector(row["state"]) for row in rows])
    motion = np.maximum(np.abs(np.diff(a, axis=0)), np.abs(np.diff(q, axis=0))).max(axis=1)
    active = np.flatnonzero(motion >= threshold)
    if not len(active):
        return 0, 0
    return max(0, int(active[0]) - margin), min(len(rows), int(active[-1]) + 2 + margin)


def select_segments(data: dict, route: str, threshold: float = 0.004, margin: int = 5) -> list[slice]:
    if route not in {"baseline", "hq", "dagger"} or threshold <= 0 or margin < 0:
        raise ValueError("无效的数据路线或裁剪参数")
    episode, rows = data["episode"], data["ticks"]
    kind = episode["kind"]
    if kind in {"demo", "hq"}:
        if episode["rating"] < (3 if route == "baseline" else 5):
            return []
        if route != "baseline" and kind != "hq":
            return []
        start, end = trim_idle(rows, threshold, margin)
        return [slice(start, end)] if end > start else []
    if kind != "dagger" or route != "dagger":
        return []
    intervention = [row["source"] == "intervention" for row in rows]
    if not any(intervention):
        full_policy = all(row["source"] == "policy" for row in rows)
        if full_policy and (data["verdict"] or {}).get("stage") == 5:
            start, end = trim_idle(rows, threshold, margin)
            return [slice(start, end)] if end > start else []
        return []
    changes = np.flatnonzero(np.diff([False, *intervention, False])).tolist()
    return [slice(start, end) for start, end in zip(changes[::2], changes[1::2], strict=True)]


def frame_for_dataset(archive: ZipFile, row: dict, task: str) -> dict:
    result = {"observation.state": vector(row["state"]), "action": vector(row["action"]), "task": task}
    for camera, path in row["images"].items():
        with Image.open(BytesIO(archive.read(path))) as image:
            result[f"observation.images.{camera}"] = center_crop(np.asarray(image.convert("RGB")))
    return result
