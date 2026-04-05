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
