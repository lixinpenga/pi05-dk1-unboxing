"""Accelerate entry point registering the blog's quadratic WSD schedule."""

from dataclasses import dataclass

from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim.lr_scheduler import LambdaLR

from .training import wsd_factor


@LRSchedulerConfig.register_subclass("quadratic_wsd")
@dataclass
class QuadraticWSD(LRSchedulerConfig):
    num_warmup_steps: int
    decay: int
    min_ratio: float = 0.0

    def build(self, optimizer, num_training_steps):
        return LambdaLR(
            optimizer,
            lambda step: wsd_factor(
                step, num_training_steps, self.num_warmup_steps, self.decay, self.min_ratio
            ),
        )


if __name__ == "__main__":
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train

    from .checkpoints import verified_from_pretrained

    PI05Policy.from_pretrained = classmethod(verified_from_pretrained)
    make_processors = lerobot_train.make_pre_post_processors

    def processors_for_repo(policy_cfg, pretrained_path=None, **kwargs):
        if policy_cfg.type == "pi05" and kwargs.get("dataset_stats") is not None:
            # Reuse weights across datasets while rebuilding the standard Pi05
            # preprocessing, including relative actions, with THIS dataset's stats.
            return make_processors(policy_cfg, dataset_stats=kwargs["dataset_stats"])
        return make_processors(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    lerobot_train.make_pre_post_processors = processors_for_repo
    lerobot_train.main()
