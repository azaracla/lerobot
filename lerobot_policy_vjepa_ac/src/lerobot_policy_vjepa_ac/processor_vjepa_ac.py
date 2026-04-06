from typing import Any
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.pipeline import ProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from .configuration_vjepa_ac import VjepaAcConfig


@ProcessorStepRegistry.register(name="vjepa_ac_logging_processor")
class VjepaAcLoggingProcessorStep(ProcessorStep):
    """Debug processor qui log les images sélectionnées par observation_delta_indices."""

    def __init__(
        self,
        enabled: bool = False,
        log_frequency: int = 100,
        save_images: bool = False,
        save_dir: str = "outputs/debug_images",
        delta_indices: list[int] = None,
        vfps: int = 30,
    ):
        self.enabled = enabled
        self.log_frequency = log_frequency
        self.save_images = save_images
        self.save_dir = Path(save_dir)
        self.delta_indices = delta_indices or [0]
        self.vfps = vfps
        self._call_count = 0

        if self.save_images:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return transition

        self._call_count += 1
        if self._call_count % self.log_frequency != 0:
            return transition

        observation = transition.get("observation", {})
        complementary = transition.get("complementary_data", {})
        
        # Recherche du timestamp réel ou reconstruction via l'index
        timestamps = None
        for target in [observation, transition, complementary]:
            for key in ["timestamp", "timestamps", "observation.timestamp"]:
                if key in target:
                    timestamps = target[key]
                    break
            if timestamps is not None:
                break

        ts_batch0 = None
        if timestamps is not None and isinstance(timestamps, torch.Tensor):
            if timestamps.ndim >= 1:
                ts_batch0 = timestamps[0].detach().cpu().numpy()
            else:
                ts_batch0 = [timestamps.item()]
        elif "index" in complementary:
            # Reconstruction : index de base + deltas / FPS
            base_idx = complementary["index"][0].item()
            ts_batch0 = [(base_idx + d) / self.vfps for d in self.delta_indices]

        for key, value in observation.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                stats = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": value.min().item(),
                    "max": value.max().item(),
                    "mean": value.mean().item(),
                }
                print(f"[VJEPA_AC_LOGGING] {key}: {stats}")

                if self.save_images and value.ndim >= 4:
                    self._save_image(value[0], key, self._call_count, ts_batch0)

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Returns features unchanged - this is just for logging."""
        return features

    def _save_image(self, tensor: torch.Tensor, key: str, step: int, timestamps=None):
        """Sauvegarde une planche contact de la séquence temporelle avec timestamps."""
        import PIL.Image
        import PIL.ImageDraw
        import torch

        # tensor shape: [T, C, H, W] ou [C, H, W]
        if tensor.ndim == 3:
            frames = [tensor]
        else:
            frames = [tensor[t] for t in range(tensor.size(0))]

        processed_frames = []
        for i, img in enumerate(frames):
            # [C, H, W] -> [H, W, C]
            if img.ndim == 3:
                if img.shape[0] in [1, 3]: # C est en premier
                    img = img.permute(1, 2, 0)
            
            if img.shape[-1] == 1: # Grayscale
                img = img.squeeze(-1)

            img_np = img.detach().cpu().numpy()
            
            # Normalisation (0-1 -> 0-255)
            if img_np.max() <= 1.01:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            pil_img = PIL.Image.fromarray(img_np)
            
            # Incrustation du timestamp
            if timestamps is not None and i < len(timestamps):
                draw = PIL.ImageDraw.Draw(pil_img)
                ts_text = f"{float(timestamps[i]):.2f}s"
                # Dessiner un petit rectangle noir derrière le texte pour la lisibilité
                draw.rectangle([5, 5, 60, 20], fill="black")
                draw.text((10, 7), ts_text, fill="white")
            
            processed_frames.append(pil_img)

        if not processed_frames:
            return

        w, h = processed_frames[0].size
        contact_sheet = PIL.Image.new("RGB", (w * len(processed_frames), h))
        
        for i, frame in enumerate(processed_frames):
            contact_sheet.paste(frame, (i * w, 0))

        safe_key = key.replace(".", "_")
        contact_sheet.save(
            self.save_dir / f"step{step:06d}_{safe_key}_sequence.png"
        )


def make_vjepa_ac_pre_post_processors(
    config: VjepaAcConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    if config.log_observation_images:
        input_steps.insert(
            0,
            VjepaAcLoggingProcessorStep(
                enabled=True,
                log_frequency=config.log_observation_frequency,
                save_images=config.save_images_to_disk,
                save_dir=config.save_images_dir,
                delta_indices=config.observation_delta_indices,
                vfps=config.vfps,
            ),
        )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
