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

    # When True, the preprocessor computes a_k = s_{k+1} - s_k from observation.state
    # (DROID/VJEPA-AC convention) instead of using batch["action"] absolute positions.
    # ACTION normalization is forced to IDENTITY because delta stats ≠ absolute stats.
    # The postprocessor then converts deltas back to absolute joint targets.
    use_delta_actions: bool = True

    predictor_embed_dim: int = 1024
    action_dim: int = 6
    pred_depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_frames: int = 1
    tubelet_size: int = 2
    vfps: int = 30  # Original video FPS (LeRobot datasets are typically 30fps)
    fps: int = 4  # Target FPS for temporal sampling (matching DROID paper)

    mpc_horizon: int = 1
    cem_num_samples: int = 200  # paper uses 800 on 4090 24GB; reduce if OOM on 16GB
    cem_num_iters: int = 10
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
    normalize_reps: bool = True
    use_extrinsics: bool = False
    use_imagenet_for_visuals: bool = True
    goal_image_path: str | None = None

    augmentation_random_resized_crop_enabled: bool = False
    augmentation_random_resized_crop_scale: float = 1.777
    augmentation_random_resized_crop_ratio: tuple[float, float] = (0.75, 1.35)
    augmentation_random_resized_crop_target_size: int = 256
    augmentation_horizontal_flip: bool = False

    log_observation_images: bool = False
    log_observation_frequency: int = 100
    save_images_to_disk: bool = False
    save_images_dir: str = "outputs/debug_images"

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
        Sorted in ascending order (past to present).
        """
        # Fallback to 1 if vfps/fps calculation is problematic
        step = max(1, round(self.vfps / self.fps))

        # Current frame is 0. We want n_obs_steps total frames.
        # Example for n=8, step=8: [-56, -48, -40, -32, -24, -16, -8, 0]
        indices = []
        for i in range(self.n_obs_steps):
            indices.append(-(self.n_obs_steps - 1 - i) * step)
        return indices

    @property
    def action_delta_indices(self) -> list:
        # We predict a horizon of actions
        return list(range(self.mpc_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
