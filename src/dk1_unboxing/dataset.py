"""LeRobot v3 export and metadata checks shared by training and inspection."""

import json
import shutil
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from .recording import frame_for_dataset, read_episode, select_segments, write_json
from .scene import ACTION_DIM, CAMERAS, FPS, JOINTS


def check_info(info: dict) -> None:
    if info["fps"] != FPS or info["codebase_version"] != "v3.0":
        raise ValueError(f"期望 {FPS} Hz 的 LeRobot v3.0 数据")
    features = info["features"]
    for field in ("observation.state", "action"):
        feature = features[field]
        if tuple(feature["shape"]) != (ACTION_DIM,) or tuple(feature["names"]) != JOINTS:
            raise ValueError(f"{field} 的顺序或维度与 DK1 不一致")
    for camera in CAMERAS:
        feature = features[f"observation.images.{camera}"]
        if tuple(feature["shape"]) != (224, 224, 3) or feature["dtype"] not in {"image", "video"}:
            raise ValueError(f"{camera} 必须为 224×224 RGB 图像或视频")


def inspect(repo: str | None, root: Path | None, revision: str | None = None) -> dict:
    if root is None:
        if not repo:
            raise ValueError("请传入 Hugging Face 数据集 ID 或本地数据目录")
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo,
                "meta/info.json",
                repo_type="dataset",
                revision=revision,
            )
        )
    else:
        path = root / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    check_info(info)
    return info


def export(raw: Path, out: Path, repo: str, route: str, threshold: float, margin: int) -> dict:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out.exists():
        raise FileExistsError(out)
    selected = []
    for path in sorted(raw.glob("*.zip")):
        with ZipFile(path) as archive:
            data = read_episode(archive)
            segments = select_segments(data, route, threshold, margin)
        if segments:
            selected.append((path, segments))
    if not selected:
        raise ValueError(f"{raw} 无符合 {route} 条件的回合")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{uuid4().hex}.partial")
    features = {
        field: {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(JOINTS)}
        for field in ("observation.state", "action")
    }
    features.update(
        {
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            }
            for name in CAMERAS
        }
    )
    manifest = []
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo,
            FPS,
            features,
            root=temporary,
            robot_type="bi_dk1_follower",
            image_writer_threads=3,
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        )
        for path, segments in selected:
            with ZipFile(path) as archive:
                data = read_episode(archive)
                episode = data["episode"]
                for part in segments:
                    for row in data["ticks"][part]:
                        dataset.add_frame(frame_for_dataset(archive, row, episode["task"]))
                    dataset.save_episode()
                    manifest.append(
                        {
                            "dataset_episode": len(manifest),
                            "raw_file": path.name,
                            "start": part.start,
                            "end_exclusive": part.stop,
                            **episode,
                        }
                    )
        dataset.finalize()
        dataset = None
        check_info(json.loads((temporary / "meta/info.json").read_text(encoding="utf-8")))
        write_json(temporary / "meta/unboxing_manifest.json", manifest)
        temporary.rename(out)
    finally:
        if dataset is not None:
            try:
                dataset.finalize()
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        elif temporary.exists():
            shutil.rmtree(temporary)
    return {
        "episodes": len(manifest),
        "frames": sum(item["end_exclusive"] - item["start"] for item in manifest),
    }
