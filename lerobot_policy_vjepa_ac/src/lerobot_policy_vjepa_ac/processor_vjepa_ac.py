from typing import Any
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.types import NormalizationMode, PipelineFeatureType, PolicyFeature
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
from lerobot.types import TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from .configuration_vjepa_ac import VjepaAcConfig

# Keys used to pass data through the transition (batch) between processor steps and the model.
RAW_DELTAS_KEY = "observation.raw_deltas"


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


@ProcessorStepRegistry.register(name="vjepa_ac_device_and_delta")
@ProcessorStepRegistry.register(name="device_processor")  # alias for policy_server override compat
class VjepaAcDeviceAndDeltaStep(ProcessorStep):
    """
    Fused step: device placement + state-to-delta-action computation.

    Replaces the separate DeviceProcessorStep + StateToDeltaActionProcessorStep pair,
    keeping the NormalizerProcessorStep at step index 3 (as expected by LeRobot's
    policy_server which hardcodes that index for loading normalizer stats).

    Pipeline:
      step 0: RenameObservationsProcessorStep
      step 1: AddBatchDimensionProcessorStep
      step 2: VjepaAcDeviceAndDeltaStep  ← this class (Device + Delta)
      step 3: NormalizerProcessorStep    ← always step_3 for policy_server compat

    Device logic mirrors DeviceProcessorStep exactly. Delta logic:
      - Computes raw_deltas = state[:, 1:] - state[:, :-1]  [B, T-1, D]
      - Writes raw_deltas into observation dict under RAW_DELTAS_KEY
      - Replaces ACTION tensor with raw_deltas (training mode)
      - Caches _current_state for the paired postprocessor
    """

    def __init__(self, device: str = "cpu", use_delta_actions: bool = True):
        from lerobot.utils.device_utils import get_safe_torch_device
        self.device = device
        self.use_delta_actions = use_delta_actions
        self.tensor_device = get_safe_torch_device(device)
        self.device = self.tensor_device.type
        self.non_blocking = "cuda" in str(self.device)
        self._current_state: torch.Tensor | None = None

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            target_device = tensor.device
        else:
            target_device = self.tensor_device
        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)
        return tensor

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        new_transition = transition.copy()

        # --- Device placement (mirrors DeviceProcessorStep) ---
        simple_tensor_keys = [
            TransitionKey.ACTION,
            TransitionKey.REWARD,
            TransitionKey.DONE,
            TransitionKey.TRUNCATED,
        ]
        dict_tensor_keys = [
            TransitionKey.OBSERVATION,
            TransitionKey.COMPLEMENTARY_DATA,
        ]
        for key in simple_tensor_keys:
            value = new_transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._to_device(value)
        for key in dict_tensor_keys:
            data_dict = new_transition.get(key)
            if data_dict is not None:
                new_transition[key] = {
                    k: self._to_device(v) if isinstance(v, torch.Tensor) else v
                    for k, v in data_dict.items()
                }

        if not self.use_delta_actions:
            return new_transition

        # --- Delta computation ---
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        state = obs.get("observation.state")  # [B, T, D]

        if state is None or not isinstance(state, torch.Tensor) or state.ndim < 2:
            return new_transition

        if state.ndim == 2:
            self._current_state = state  # [B, D] — single frame, cache only
            return new_transition

        # state: [B, T, D]
        self._current_state = state[:, -1]  # [B, D]
        raw_deltas = state[:, 1:] - state[:, :-1]  # [B, T-1, D]

        obs = dict(obs)
        obs[RAW_DELTAS_KEY] = raw_deltas
        new_transition[TransitionKey.OBSERVATION] = obs

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, torch.Tensor):
            new_transition[TransitionKey.ACTION] = raw_deltas

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"device": self.device, "use_delta_actions": self.use_delta_actions}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        self._current_state = None


@ProcessorStepRegistry.register(name="vjepa_ac_delta_to_absolute_action")
class DeltaToAbsoluteActionProcessorStep(ProcessorStep):
    """
    Postprocessor: converts delta action sequence (model output) back to absolute
    joint targets by adding the cached current state from the paired preprocessor.

    delta_actions: [B, H, D]  (output of CEM, already unnormalized)
    absolute:      [B, H, D]  cumulative sum starting from current_state
      absolute[:, 0] = current_state + delta[:, 0]
      absolute[:, k] = current_state + sum(delta[:, 0:k+1])
    """

    def __init__(self, preprocessor: "VjepaAcDeviceAndDeltaStep | None" = None):
        self._preprocessor = preprocessor

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        action = transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, torch.Tensor):
            return transition

        if self._preprocessor is None:
            raise RuntimeError(
                "DeltaToAbsoluteActionProcessorStep requires a reference to the paired "
                "StateToDeltaActionProcessorStep. Use make_vjepa_ac_pre_post_processors() "
                "to create both steps together."
            )
        current_state = getattr(self._preprocessor, "_current_state", None)
        if current_state is None:
            return transition

        # action: [B, H, D], current_state: [B, D]
        if action.ndim == 2:
            # [B, D] single-step delta
            absolute = current_state + action
        else:
            # [B, H, D] multi-step delta — cumsum gives trajectory of absolute targets
            absolute = current_state.unsqueeze(1) + torch.cumsum(action, dim=1)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass


def make_vjepa_ac_pre_post_processors(
    config: VjepaAcConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # When use_delta_actions=True, ACTION normalization must be IDENTITY:
    # delta stats ≠ absolute stats, and the action_encoder linear layer handles scaling.
    norm_map = dict(config.normalization_mapping)
    if config.use_delta_actions:
        norm_map["ACTION"] = NormalizationMode.IDENTITY

    # VjepaAcDeviceAndDeltaStep fuses Device + Delta into a single step so that
    # NormalizerProcessorStep is always at index 3, matching the hardcoded index
    # in LeRobot's policy_server._load_dataset_stats().
    device_and_delta_step = VjepaAcDeviceAndDeltaStep(
        device=config.device,
        use_delta_actions=config.use_delta_actions,
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        device_and_delta_step,  # step 2: Device + Delta (fused)
    ]

    input_steps.append(  # step 3: Normalizer — always at index 3
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=norm_map,
            stats=dataset_stats,
            device=config.device,
        )
    )

    if config.log_observation_images:
        # Inserting at 0 shifts all steps — NormalizerProcessorStep moves to step 4.
        # This is acceptable: logging is a debug mode not used in production inference.
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

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=norm_map, stats=dataset_stats
        ),
    ]

    if config.use_delta_actions:
        output_steps.append(DeltaToAbsoluteActionProcessorStep(preprocessor=device_and_delta_step))

    output_steps.append(DeviceProcessorStep(device="cpu"))

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
