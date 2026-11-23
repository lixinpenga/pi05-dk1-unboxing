"""LeRobot π0.5 checkpoint adapter for the local RTC loop."""

from pathlib import Path

import numpy as np

from .checkpoints import load_pi05_config, load_processor_stats, verified_from_pretrained
from .recording import Tick
from .rtc import HORIZON, RTCSettings
from .scene import ACTION_DIM, CAMERAS, JOINTS, center_crop


class Pi05Predictor:
    def __init__(self, checkpoint: Path, device: str, task: str, settings: RTCSettings):
        import torch
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.utils import prepare_observation_for_inference

        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"需要本地的 pretrained_model 目录：{checkpoint}")
        self.torch = torch
        self.prepare = prepare_observation_for_inference
        self.device = torch.device(device)
        self.task = task
        cfg = load_pi05_config(checkpoint, device)
        self._check_config(cfg)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        cfg.device = device
        cfg.gradient_checkpointing = False
        stats = load_processor_stats(checkpoint)
        cfg.rtc_config = RTCConfig(enabled=settings.guided, execution_horizon=settings.blend)
        if cfg.use_peft:
            from peft import PeftConfig, PeftModel

            adapter = PeftConfig.from_pretrained(str(checkpoint))
            base = verified_from_pretrained(
                PI05Policy,
                adapter.base_model_name_or_path,
                config=cfg,
                revision=adapter.revision,
            )
            self.model = PeftModel.from_pretrained(base, str(checkpoint), config=adapter)
        else:
            self.model = verified_from_pretrained(PI05Policy, checkpoint, config=cfg)
        self.model.eval()
        self.pre, self.post = make_pi05_pre_post_processors(cfg, dataset_stats=stats)

    @staticmethod
    def _check_config(cfg):
        if cfg.type != "pi05" or cfg.chunk_size != HORIZON or cfg.n_obs_steps != 1:
            raise ValueError("需要单帧观测、50 步动作块的 π0.5 checkpoint")
        wanted = {f"observation.images.{name}" for name in CAMERAS}
        if set(cfg.image_features) != wanted:
            raise ValueError("checkpoint 必须直接使用 DK1 的三路相机键名")
        if any(
            feature is None or tuple(feature.shape) != (ACTION_DIM,)
            for feature in (cfg.robot_state_feature, cfg.action_feature)
        ):
            raise ValueError("需要 14 维 DK1 状态和动作")
        if tuple(cfg.action_feature_names or ()) != JOINTS:
            raise ValueError("无法核对 checkpoint 的 14 维关节顺序")
        if cfg.use_visual_memory or cfg.use_proprioceptive_memory:
            raise ValueError("此部署入口使用单帧 π0.5 策略")

    def __call__(self, obs: Tick, prefix: np.ndarray, delay: int, horizon: int) -> np.ndarray:
        data = {
            "observation.state": obs.state,
            **{f"observation.images.{name}": center_crop(obs.images[name]) for name in CAMERAS},
        }
        with self.torch.no_grad():
            base = self.prepare(data, self.device, self.task, "bi_dk1_follower")
            if horizon:
                # Re-encode physical actions under the CURRENT observation. With
                # relative actions the previous normalized chunk has a stale origin.
                base["action"] = self.torch.from_numpy(prefix.copy()).to(self.device).unsqueeze(0)
            batch = self.pre(base)
            options = (
                {
                    "prev_chunk_left_over": batch["action"],
                    "inference_delay": min(delay, horizon),
                    "execution_horizon": horizon,
                }
                if horizon
                else {}
            )
            raw = self.model.predict_action_chunk(batch, **options)
            return self.post(raw).squeeze(0).detach().float().cpu().numpy()
