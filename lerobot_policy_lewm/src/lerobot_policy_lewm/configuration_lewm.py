"""Configuration for LeWM (LeWorldModel) JEPA world model policy.

A JEPA (Joint Embedding Predictive Architecture) that trains end-to-end
from pixels with only 2 loss terms: next-embedding prediction (MSE) +
Gaussian regularization (SIGReg).

Reference: LeWM paper (arXiv 2603.19312v1), MIT License.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("lewm")
@dataclass
class LeWMConfig(PreTrainedConfig):
    """Configuration for LeWM JEPA world model.

    Key hyperparameters (defaults from the LeWM paper for PushT):
    - embed_dim=192 (ViT-Tiny)
    - img_size=224, patch_size=14
    - predictor: 6 layers, 16 heads, mlp_dim=2048
    - history_size=3, num_preds=1
    - sigreg_weight=0.09
    """

    # ---- Model architecture ----
    img_size: int = 224
    patch_size: int = 14
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3

    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64

    proj_hidden_dim: int = 2048
    action_emb_dim: int = 192

    # ---- Temporal structure ----
    # Number of observation steps. For JEPA: history_size = n_obs_steps - 1.
    n_obs_steps: int = 4
    # Number of future steps to predict (typically 1 for next-step prediction).
    num_preds: int = 1

    # ---- Loss ----
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    # ---- MPC / Planning ----
    horizon: int = 5
    cem_num_samples: int = 300
    cem_n_steps: int = 30
    cem_topk: int = 30
    cem_var_scale: float = 1.0

    # ---- Normalization ----
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # ImageNet normalization done inside model
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # ---- Optimizer / Scheduler ----
    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-3
    scheduler_warmup_steps: int = 500
    scheduler_name: str = "cosine"

    # ---- Misc ----
    push_to_hub: bool = False
    use_amp: bool = False
    interpolate_pos_encoding: bool = False  # for variable image sizes at eval
    device: str | None = None

    @property
    def history_size(self) -> int:
        """Number of context frames for the predictor."""
        return self.n_obs_steps - self.num_preds

    # ---- Abstract property implementations ----

    @property
    def observation_delta_indices(self) -> list[int]:
        """Frame indices for observation sampling.

        Returns indices [0, 1, ..., n_obs_steps-1] representing sequential
        frames at the configured FPS.
        """
        return list(range(self.n_obs_steps))

    @property
    def action_delta_indices(self) -> list[int]:
        """Frame indices for action sampling.

        Returns indices [0, ..., n_obs_steps-2] — one fewer than observations.
        This gives n_obs_steps observations and n_obs_steps-1 actions.
        action[t] conditions obs[t] → obs[t+1], so we need N-1 actions for N observations.
        """
        return list(range(self.n_obs_steps - 1))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        """World models don't use rewards. Return None to skip delta timestamps."""
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig | None:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """Validate that input/output features are compatible."""
        if not self.input_features:
            return
        # Must have at least one visual feature
        image_features = self.image_features
        if not image_features:
            raise ValueError(f"{self.type}: At least one VISUAL input feature is required.")
        # Must have an action output feature
        if self.output_features:
            for ft in self.output_features.values():
                if ft.type == FeatureType.ACTION:
                    break
            else:
                raise ValueError(f"{self.type}: ACTION output feature is required.")
