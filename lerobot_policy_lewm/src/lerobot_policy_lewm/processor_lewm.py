"""Pre/post processors for the LeWM world model policy.

Minimal pipeline:
- Preprocessor: Resize images to 224x224, ImageNet normalize, z-score normalize
- Postprocessor: Unnormalize actions
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from lerobot.processor.pipeline import (
    DataProcessorPipeline,
    ObservationProcessorStep,
    ActionProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.configs.types import NormalizationMode

# ImageNet stats (used for VISUAL=IDENTITY normalization)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePreprocessorStep(ObservationProcessorStep):
    """Resize images to target size and apply ImageNet normalization."""

    def __init__(self, target_size: int = 224):
        super().__init__()
        self.target_size = target_size

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Resize images and apply ImageNet normalization.

        Handles both (B, C, H, W) and (B, T, C, H, W) shapes.
        """
        for key, value in observation.items():
            # Match "observation.image", "observation.image_<name>", or bare "image*"
            if ("image" in key.lower()) and isinstance(value, torch.Tensor) and value.ndim >= 4:
                if value.ndim == 5:
                    B, T, C, H, W = value.shape
                    value = value.reshape(B * T, C, H, W)
                    value = F.interpolate(value, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
                    mean = value.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
                    std = value.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
                    value = (value - mean) / std
                    value = value.reshape(B, T, C, self.target_size, self.target_size)
                else:
                    value = F.interpolate(value, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
                    mean = value.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
                    std = value.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
                    value = (value - mean) / std
                observation[key] = value
        return observation

    def transform_features(self, features: dict) -> dict:
        """Image resize + ImageNet norm doesn't change feature shapes."""
        return features


@ProcessorStepRegistry.register("lewm_image_preprocessor")
class RegisteredImagePreprocessorStep(ImagePreprocessorStep):
    """Registered variant for serialization."""
    pass


def make_lewm_pre_post_processors(
    policy_cfg,
    pretrained_path: Optional[str] = None,
    dataset_stats: Optional[dict] = None,
    **kwargs,
) -> tuple:
    """Create pre and post processor pipelines for LeWM.

    Args:
        policy_cfg: LeWMConfig instance.
        pretrained_path: Path to pretrained model (for loading).
        dataset_stats: Dataset statistics for normalization.

    Returns:
        (preprocessor, postprocessor) tuple of DataProcessorPipeline.
    """
    img_size = getattr(policy_cfg, "img_size", 224)

    # Build preprocessor steps.
    # Include standard LeRobot steps (device, rename) for eval override compatibility.
    pre_steps = []
    post_steps = []

    try:
        from lerobot.processor.normalize_processor import (
            NormalizerProcessorStep,
            UnnormalizerProcessorStep,
        )
        from lerobot.processor.device_processor import DeviceProcessorStep
        from lerobot.processor.rename_processor import RenameObservationsProcessorStep

        # Image resize + ImageNet norm
        pre_steps.append(ImagePreprocessorStep(target_size=img_size))

        # Standard steps required for lerobot-eval override compatibility
        pre_steps.append(RenameObservationsProcessorStep(rename_map={}))
        pre_steps.append(DeviceProcessorStep(device=policy_cfg.device))

        if dataset_stats is not None:
            normalizer = NormalizerProcessorStep(
                stats=dataset_stats,
                features={**policy_cfg.input_features, **policy_cfg.output_features},
                norm_map=policy_cfg.normalization_mapping,
            )
            pre_steps.append(normalizer)

            unnormalizer = UnnormalizerProcessorStep(
                stats=dataset_stats,
                features=policy_cfg.output_features,
                norm_map=policy_cfg.normalization_mapping,
            )
            post_steps.append(unnormalizer)
        else:
            # Even without stats, unnormalizer is needed for eval compatibility
            pass

        preprocessor = DataProcessorPipeline(
            name="policy_preprocessor",
            steps=pre_steps,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )

        postprocessor = DataProcessorPipeline(
            name="policy_postprocessor",
            steps=post_steps,
            to_transition=lambda x: {"action": x},
            to_output=lambda t: t["action"],
        )
    except ImportError:
        # Fallback: minimal pipeline without LeRobot normalizers
        preprocessor = DataProcessorPipeline(
            name="policy_preprocessor",
            steps=pre_steps,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = DataProcessorPipeline(
            name="policy_postprocessor",
            steps=[],
            to_transition=lambda x: {"action": x},
            to_output=lambda t: t["action"],
        )

    return preprocessor, postprocessor
