"""LeWM world model policy for LeRobot.

Wraps the JEPA world model as a LeRobot PreTrainedPolicy.
Training: self-supervised next-embedding prediction in latent space.
Inference: CEM-based Model Predictive Control (MPC).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType

from .configuration_lewm import LeWMConfig
from .jepa import JEPA
from .solver import CEMSolver

logger = logging.getLogger(__name__)

# ImageNet stats for encoding goal images
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class LeWMPolicy(PreTrainedPolicy):
    """LeWM JEPA world model policy.

    Training:
        forward(batch) computes:
        - pred_loss: MSE between predicted and target latent embeddings
        - sigreg_loss: Gaussian regularization on latent embeddings
        - loss = pred_loss + sigreg_loss

    Inference:
        select_action(batch) uses CEM to find the action sequence
        that minimizes the distance between the predicted final latent
        and the goal latent.
    """

    name = "lewm"
    config_class = LeWMConfig

    def __init__(
        self,
        config: LeWMConfig,
        dataset_stats: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(config)
        self.dataset_stats = dataset_stats

        # Store action normalization params for denormalizing CEM outputs
        self._action_min = None
        self._action_max = None
        if dataset_stats and "action" in dataset_stats:
            stats = dataset_stats["action"]
            if "min" in stats and "max" in stats:
                self._action_min = torch.as_tensor(stats["min"])
                self._action_max = torch.as_tensor(stats["max"])

        # Resolve action dimension from output features
        action_dim = 2  # default
        if config.output_features:
            for ft_name, ft in config.output_features.items():
                if ft.type == FeatureType.ACTION:
                    action_dim = ft.shape[0]
                    break

        # Resolve state dimension from input features
        state_dim = 0
        if config.input_features:
            for ft_name, ft in config.input_features.items():
                if ft.type == FeatureType.STATE:
                    state_dim = ft.shape[0]
                    break

        # Build the JEPA world model
        self.model = JEPA(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_heads=config.encoder_heads,
            num_frames=config.history_size,
            predictor_depth=config.predictor_depth,
            predictor_heads=config.predictor_heads,
            predictor_mlp_dim=config.predictor_mlp_dim,
            predictor_dim_head=config.predictor_dim_head,
            proj_hidden_dim=config.proj_hidden_dim,
            action_dim=action_dim,
            action_emb_dim=config.action_emb_dim,
            sigreg_knots=config.sigreg_knots,
            sigreg_num_proj=config.sigreg_num_proj,
            _interpolate_pos_encoding=config.interpolate_pos_encoding,
        )

        # Goal image cache (set during inference via set_goal)
        self._goal_image: Optional[torch.Tensor] = None
        self._goal_emb: Optional[torch.Tensor] = None

        # Observation cache for temporal context
        self._obs_cache: dict[str, list[torch.Tensor]] = {}

        self._action_dim = action_dim
        self._state_dim = state_dim
        self._num_preds = config.num_preds
        self._sigreg_weight = config.sigreg_weight
        self._history_size = config.history_size
        self._horizon = config.horizon

    # ---- LeRobot required methods ----

    def reset(self):
        """Clear observation/action caches. Called on env.reset()."""
        self._obs_cache = {}
        self._goal_emb = None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Training forward pass.

        Args:
            batch: dict with observation.image (B, T, C, H, W) and action (B, T-1, A).

        Returns:
            (loss, output_dict) where loss is a scalar tensor.
        """
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.model(batch, num_preds=self._num_preds)
        loss = output["pred_loss"] + self._sigreg_weight * output["sigreg_loss"]

        info = {
            "pred_loss": output["pred_loss"].item(),
            "sigreg_loss": output["sigreg_loss"].item(),
            "loss": loss.item(),
        }
        return loss, info

    def _update_embedding_cache(self, emb: torch.Tensor):
        """Maintain a rolling cache of encoded embeddings for temporal context.

        Args:
            emb: (B, 1, D) new embedding to append.
        """
        if "emb" not in self._obs_cache:
            self._obs_cache["emb"] = emb
        else:
            cached = self._obs_cache["emb"]
            # Keep last history_size embeddings
            self._obs_cache["emb"] = torch.cat([cached, emb], dim=1)[:, -self._history_size:]

    def _get_context_emb(self) -> torch.Tensor:
        """Get current context embeddings, padded if not enough history."""
        if "emb" not in self._obs_cache:
            return None
        cached = self._obs_cache["emb"]  # (B, T_cache, D)
        if cached.shape[1] < self._history_size:
            # Pad by repeating first embedding
            pad_len = self._history_size - cached.shape[1]
            pad = cached[:, :1].expand(-1, pad_len, -1)
            return torch.cat([pad, cached], dim=1)
        return cached[:, -self._history_size:]

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert normalized action [-1, 1] back to raw action space via MIN_MAX."""
        if self._action_min is not None and self._action_max is not None:
            denom = self._action_max - self._action_min
            return ((action + 1) / 2.0) * denom.to(action.device) + self._action_min.to(action.device)
        logger.warning("No dataset_stats action min/max — returning actions as-is")
        return action

    def select_action(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Select an action using CEM-based MPC.

        Encodes the current observation, maintains a rolling embedding cache,
        runs CEM to find the optimal action sequence, and returns the first action.

        Args:
            batch: dict with observation.image (B, 1, C, H, W) or (B, T, C, H, W).

        Returns:
            action: (B, action_dim) tensor.
        """
        # Handle images without temporal dim: (B, C, H, W) → (B, 1, C, H, W)
        batch_for_encode = {**batch}
        if batch_for_encode["observation.image"].ndim == 4:
            batch_for_encode["observation.image"] = batch_for_encode["observation.image"].unsqueeze(1)
        if "observation.state" in batch_for_encode:
            if batch_for_encode["observation.state"].ndim == 2:
                batch_for_encode["observation.state"] = batch_for_encode["observation.state"].unsqueeze(1)

        B = batch_for_encode["observation.image"].shape[0]

        # Ensure action key exists with zero placeholder for encoding
        if batch_for_encode.get("action") is None:
            batch_for_encode["action"] = torch.zeros(B, 1, self._action_dim, device=batch_for_encode["observation.image"].device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            # Encode current observation
            info = self.model.encode(batch_for_encode)
            # Update rolling cache
            self._update_embedding_cache(info["emb"])
            # Build context with enough history
            info["emb"] = self._get_context_emb()  # (B, history_size, D)

            # Set goal embedding
            if self._goal_image is not None:
                goal_batch = {
                    "observation.image": self._goal_image,
                    "action": torch.zeros(1, 1, self._action_dim, device=self._goal_image.device),
                }
                goal_info = self.model.encode(goal_batch)
                info["goal_emb"] = goal_info["emb"]
            else:
                info["goal_emb"] = info["emb"][:, -1:]  # (B, 1, D)

            # Run CEM to find optimal actions
            solver = CEMSolver(
                model=self.model,
                num_samples=self.config.cem_num_samples,
                n_steps=self.config.cem_n_steps,
                topk=self.config.cem_topk,
                var_scale=self.config.cem_var_scale,
                horizon=self._horizon,
                action_dim=self._action_dim,
                action_low=torch.tensor(self.config.action_low) if self.config.action_low else None,
                action_high=torch.tensor(self.config.action_high) if self.config.action_high else None,
                init_mean=self.config.cem_init_mean,
                device=str(info["emb"].device),
            )

            result = solver.solve(info)
            actions = result["actions"]  # (B, H, A)

        if was_training:
            self.model.train()
        return self._denormalize_action(actions[:, 0])  # (B, A)

    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Predict a chunk of actions (for action-chunking compatibility).

        Returns the full CEM-optimized action horizon as a chunk.
        """
        # Handle images without temporal dim: (B, C, H, W) → (B, 1, C, H, W)
        batch_for_encode = {**batch}
        if batch_for_encode["observation.image"].ndim == 4:
            batch_for_encode["observation.image"] = batch_for_encode["observation.image"].unsqueeze(1)
        if "observation.state" in batch_for_encode:
            if batch_for_encode["observation.state"].ndim == 2:
                batch_for_encode["observation.state"] = batch_for_encode["observation.state"].unsqueeze(1)

        B = batch_for_encode["observation.image"].shape[0]
        if batch_for_encode.get("action") is None:
            batch_for_encode["action"] = torch.zeros(B, 1, self._action_dim, device=batch_for_encode["observation.image"].device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            info = self.model.encode(batch_for_encode)

            if self._goal_image is not None:
                goal_batch = {
                    "observation.image": self._goal_image,
                    "action": torch.zeros(1, 1, self._action_dim, device=self._goal_image.device),
                }
                goal_info = self.model.encode(goal_batch)
                info["goal_emb"] = goal_info["emb"]
            else:
                info["goal_emb"] = info["emb"][:, -1:]

            solver = CEMSolver(
                model=self.model,
                num_samples=self.config.cem_num_samples,
                n_steps=self.config.cem_n_steps,
                topk=self.config.cem_topk,
                var_scale=self.config.cem_var_scale,
                horizon=self._horizon,
                action_dim=self._action_dim,
                action_low=torch.tensor(self.config.action_low) if self.config.action_low else None,
                action_high=torch.tensor(self.config.action_high) if self.config.action_high else None,
                init_mean=self.config.cem_init_mean,
                device=str(info["emb"].device),
            )

            result = solver.solve(info)
        if was_training:
            self.model.train()
        return self._denormalize_action(result["actions"])  # (B, H, A)

    def get_optim_params(self) -> dict:
        """Return parameter groups for optimizer.

        Returns groups compatible with LeRobot's optimizer factory.
        """
        return self.model.parameters()

    def set_goal(self, goal_image: torch.Tensor):
        """Set the goal image for MPC planning.

        Args:
            goal_image: (1, C, H, W) or (C, H, W) normalized image tensor.
        """
        if goal_image.dim() == 3:
            goal_image = goal_image.unsqueeze(0)
        # Ensure temporal dim
        if goal_image.dim() == 4:
            goal_image = goal_image.unsqueeze(1)  # (1, 1, C, H, W)
        self._goal_image = goal_image
        self._goal_emb = None  # Will be recomputed on next select_action

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
