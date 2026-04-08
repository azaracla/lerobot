import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import LRSchedulerConfig


@LRSchedulerConfig.register_subclass("wsd")
@dataclass
class WSDSchedulerConfig(LRSchedulerConfig):
    """Warmup-Stable-Decay scheduler, matching the Facebook VJEPA2/DROID training schedule.

    Linear warmup → constant peak LR → cosine anneal to final_lr.
    """

    num_warmup_steps: int = 4500
    num_stable_steps: int = 85500
    num_anneal_steps: int = 4500
    start_lr: float = 7.5e-5
    peak_lr: float = 4.25e-4
    final_lr: float = 0.0

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # LambdaLR multiplies the base optimizer LR by the returned value.
        # Normalise by peak_lr so the multiplier is in [0, 1].
        peak = self.peak_lr

        def lr_lambda(current_step):
            if current_step < self.num_warmup_steps:
                t = current_step / max(1, self.num_warmup_steps)
                return (self.start_lr + (peak - self.start_lr) * t) / peak
            elif current_step < self.num_warmup_steps + self.num_stable_steps:
                return 1.0
            else:
                progress = (current_step - self.num_warmup_steps - self.num_stable_steps) / max(
                    1, self.num_anneal_steps
                )
                progress = min(progress, 1.0)
                lr = peak + (self.final_lr - peak) * 0.5 * (1 + math.cos(math.pi * progress))
                return lr / peak

        return LambdaLR(optimizer, lr_lambda, -1)


from .configuration_vjepa_ac import VjepaAcConfig
from .modeling_vjepa_ac import VjepaAcPolicy
from .processor_vjepa_ac import make_vjepa_ac_pre_post_processors
from .transforms import DroidRandomResizedCrop

# Register custom transform with LeRobot
try:
    from lerobot.datasets import transforms as lerobot_transforms

    # Patch make_transform_from_config to support DroidRandomResizedCrop
    original_make_transform = lerobot_transforms.make_transform_from_config

    def patched_make_transform_from_config(cfg):
        if cfg.type == "DroidRandomResizedCrop":
            return DroidRandomResizedCrop(**cfg.kwargs)
        return original_make_transform(cfg)

    lerobot_transforms.make_transform_from_config = patched_make_transform_from_config
except Exception:
    # Fail silently if LeRobot is not available
    pass

__all__ = [
    "VjepaAcConfig",
    "VjepaAcPolicy",
    "make_vjepa_ac_pre_post_processors",
    "DroidRandomResizedCrop",
]
