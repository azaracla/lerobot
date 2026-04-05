from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("vjepa_ac")
@dataclass
class VjepaAcConfig(PreTrainedConfig):
    model_name: str = "vjepa2_1_vit_giant_384"
    encoder_repo_id: str = "facebookresearch/vjepa2"
    push_to_hub: bool = False
    img_size: int = 384
    patch_size: int = 16
    embed_dim: int = 1536

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    predictor_embed_dim: int = 1024
    action_dim: int = 6
    pred_depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_frames: int = 1
    tubelet_size: int = 1
    vfps: int = 30  # Original video FPS (LeRobot datasets are typically 30fps)
    fps: int = 4  # Target FPS for temporal sampling (matching DROID paper)

    mpc_horizon: int = 15
    cem_num_samples: int = 800
    cem_num_iters: int = 5
    cem_elite_ratio: float = 0.1
    cem_std: float = 0.5
    cem_momentum_mean: float = 0.25
    cem_momentum_std: float = 0.95
    cem_maxnorm: float = 0.05

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    scheduler_stable_steps: int = 0
    scheduler_anneal_steps: int = 0
    scheduler_start_lr: float = 7.5e-5
    scheduler_final_lr: float = 0.0
    loss_exp: float = 1.0
    auto_steps: int = 1
    normalize_reps: bool = False
    use_extrinsics: bool = False
    use_imagenet_for_visuals: bool = True

    augmentation_random_resized_crop_enabled: bool = False
    augmentation_random_resized_crop_scale: float = 1.777
    augmentation_random_resized_crop_ratio: tuple[float, float] = (0.75, 1.35)
    augmentation_random_resized_crop_target_size: int = 256
    augmentation_horizontal_flip: bool = False

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig | None:
        if self.scheduler_name == "wsd":
            return DiffuserSchedulerConfig(
                name=self.scheduler_name,
                num_warmup_steps=self.scheduler_warmup_steps,
            )
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("VjepaAc requires at least one image input.")

    @property
    def observation_delta_indices(self) -> list:
        """
        Returns frame indices for observation frames, sampled at target fps.

        To match DROID paper (4fps, 8 frames), we calculate the frame spacing
        based on the video fps (vfps) and target fps.

        At vfps=30 with target fps=4: frame_step = 30/4 = 7.5 ≈ 8 frames

        Args:
            n_obs_steps: Number of observation frames (from base class)
            vfps: Original video FPS (default: 30, LeRobot datasets)
            fps: Target FPS for temporal sampling (default: 4, matching DROID)

        Returns:
            List of frame indices (negative = past frames, 0 = current)
        """
        frame_step = round(self.vfps / self.fps)  # e.g., 30/4 = 7.5 → 8

        # Generate indices spaced by frame_step, ending at 0 (current frame)
        # For n_obs_steps=8, fps=4, vfps=30:
        # indices = [0, -8, -16, -24, -32, -40, -48, -56]
        indices = [0] + [-i * frame_step for i in range(1, self.n_obs_steps)]
        return indices

    @property
    def action_delta_indices(self) -> list:
        # We predict a horizon of actions
        return list(range(self.mpc_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
