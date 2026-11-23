"""TRLC-DK1 plugin adapter and three independent USB camera readers."""

import logging
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np

from .recording import Tick
from .scene import CAMERAS, FPS, JOINTS, joint_dict, vector

logger = logging.getLogger(__name__)


class Limits:
    def __init__(self, config: dict):
        self.lower = vector(config["lower"])
        self.upper = vector(config["upper"])
        self.jump = vector(config["max_jump"])
        self.tracking = vector(config["max_tracking_error"])
        grippers = [6, 13]
        if (
            np.any(self.lower >= self.upper)
            or np.any(self.jump <= 0)
            or np.any(self.tracking <= 0)
            or np.any(self.lower[grippers] < 0)
            or np.any(self.upper[grippers] > 1)
        ):
            raise ValueError("请填写有效的现场关节、夹爪、相邻目标和跟踪误差限制")

    def check(self, target: np.ndarray, state: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        target = vector(target)
        invalid = (
            (target < self.lower) | (target > self.upper) | (np.abs(target - state) > self.tracking)
        )
        if previous is not None:
            invalid |= np.abs(target - previous) > self.jump
        if invalid.any():
            names = ", ".join(np.asarray(JOINTS)[invalid])
            raise RuntimeError(f"目标超出现场限值，已停止下发新指令 ({names})")
        return target


class CameraStream:
    def __init__(self, dev: int | str, fps: int):
        import cv2

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        try:
            if not self.cap.isOpened():
                raise OSError(f"无法打开摄像头 {dev}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual = self.cap.get(cv2.CAP_PROP_FPS)
            codec = int(self.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
            if abs(actual - fps) > 1 or codec != b"MJPG":
                raise OSError(f"{dev} 未协商到 MJPG {fps} Hz，实际为 {codec!r} {actual:.1f} Hz")
            self.cv2 = cv2
            self.lock, self.ready, self.stop = Lock(), Event(), Event()
            self.last = None
            self.error = None
            self.worker = Thread(target=self._read, daemon=True, name=f"camera:{dev}")
            self.worker.start()
        except BaseException:
            self.cap.release()
            raise

    def _read(self):
        seq = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                timestamp = monotonic()
                if not ok or frame.shape != (480, 640, 3):
                    raise OSError("摄像头未返回 640×480 帧")
                rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.last = (timestamp, seq, rgb)
                self.ready.set()
                seq += 1
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.error = exc
            self.ready.set()
        finally:
            self.cap.release()

    def read(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("相机读取失败") from self.error
            if self.last is None:
                raise RuntimeError("相机尚未返回帧")
            return self.last

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            logger.error("相机读取线程未在两秒内退出")


def close_arm(arm):
    if arm.is_connected:
        try:
            arm.disconnect()
        except Exception:
            logger.exception("DK1 断开连接失败")


class LiveRobot:
    def __init__(self, config: dict):
        from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig

        self.limits = Limits(config["limits"])
        self.fps = config.get("camera_fps", FPS)
        self.max_skew = float(config.get("camera_max_skew_ms", 45)) / 1000
        self.max_age = float(config.get("camera_max_age_ms", 100)) / 1000
        if self.fps <= 0 or not 0 < self.max_skew <= self.max_age:
            raise ValueError("相机频率或时序限制无效")
        self.devices = config["cameras"]
        if set(self.devices) != set(CAMERAS) or len(set(self.devices.values())) != len(CAMERAS):
            raise ValueError("需要三台独立相机，键名为 left_wrist、right_wrist、top")
        robot = dict(config["robot"])
        identity = robot.pop("id", "dk1-unboxing")
        ports = {side: robot.pop(f"{side}_arm_port") for side in ("left", "right")}
        if "cameras" in robot or len(set(ports.values())) != 2:
            raise ValueError("需要不同的从臂串口，摄像头配置应放在顶层")
        mode = robot.get("control_mode", "impedance")
        if mode not in {"impedance", "pos_vel"}:
            raise ValueError("SDK 控制模式需为 impedance 或 pos_vel")
        if mode == "impedance":
            try:
                from trlc_dk1_control import DK1Robot  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "阻抗模式需要以 editable 安装 DK1 SDK 源码目录，详见真机手册"
                ) from exc
        self.arms = {
            side: DK1Follower(DK1FollowerConfig(port=port, id=f"{identity}-{side}", **robot))
            for side, port in ports.items()
        }
        actual = {f"{side}_{name}" for side, arm in self.arms.items() for name in arm.action_features}
        if actual != set(JOINTS):
            raise ValueError("SDK 动作字段和数据集的 14 维字段不一致")
        self.stack = ExitStack()
        self.previous = None

    def __enter__(self):
        with ExitStack() as cleanup:
            self.cameras = {}
            for name, dev in self.devices.items():
                camera = CameraStream(dev, self.fps)
                cleanup.callback(camera.close)
                self.cameras[name] = camera
            for camera in self.cameras.values():
                if not camera.ready.wait(timeout=3):
                    raise TimeoutError("摄像头启动超时")
                camera.read()
            for arm in self.arms.values():
                cleanup.callback(close_arm, arm)
                arm.connect()
            self.stack = cleanup.pop_all()
        return self

    def __exit__(self, *_):
        self.stack.close()

    def observe(self, start: float) -> Tick:
        state = vector(
            {
                f"{side}_{name}": value
                for side, arm in self.arms.items()
                for name, value in arm.get_observation().items()
            }
        )
        samples = {name: camera.read() for name, camera in self.cameras.items()}
        now = monotonic()
        stamps = [sample[0] for sample in samples.values()]
        if now - min(stamps) > self.max_age or max(stamps) - min(stamps) > self.max_skew:
            raise RuntimeError("相机帧过期或三路时间差超限")
        return Tick(
            now - start,
            state,
            {name: sample[2] for name, sample in samples.items()},
            {name: now - sample[0] for name, sample in samples.items()},
            {name: sample[1] for name, sample in samples.items()},
        )

    def send(self, target: np.ndarray, state: np.ndarray) -> np.ndarray:
        checked = self.limits.check(target, state, self.previous)
        request = joint_dict(checked)
        sent = {}
        for side, arm in self.arms.items():
            prefix = f"{side}_"
            targets = {
                key.removeprefix(prefix): value
                for key, value in request.items()
                if key.startswith(prefix)
            }
            sent.update({f"{prefix}{key}": value for key, value in arm.send_action(targets).items()})
        self.previous = vector(sent)
        return self.previous.copy()


def make_teleop(name: str, config: dict):
    settings = config["teleop"][name]
    if name == "gello":
        from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig

        return BiDK1Leader(BiDK1LeaderConfig(**settings))
    if name != "vr":
        raise ValueError(name)
    try:
        from vr_teleop_kit.lerobot import BiQuestTeleoperator, BiQuestTeleoperatorConfig
    except ModuleNotFoundError as exc:
        if exc.name != "vr_teleop_kit":
            raise
        raise RuntimeError("VR 模式需要预先安装兼容的 WebXR 遥操作适配器") from exc

    rotation = np.asarray(settings["r_calib"], dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.01)
        or not np.isclose(np.linalg.det(rotation), 1, atol=0.01)
        or not Path(settings["urdf_path"]).is_file()
    ):
        raise ValueError("请填写 VR 到机器人基座的有效旋转矩阵及 URDF 路径")
    for side in ("left", "right"):
        rest = np.asarray(settings[f"rest_qpos_{side}"], dtype=float)
        if rest.shape != (6,) or not np.isfinite(rest).all():
            raise ValueError("请填写左右臂各六维的参考姿态")
    return BiQuestTeleoperator(BiQuestTeleoperatorConfig(**settings))
