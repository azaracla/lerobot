"""JEPA (Joint Embedding Predictive Architecture) world model.

Ported from https://github.com/lucas-maes/le-wm (MIT License).

The JEPA encodes observations into latent embeddings, then uses an
autoregressive predictor conditioned on actions to predict future embeddings.
Training uses only two loss terms:
  1. Next-embedding prediction loss (MSE)
  2. SIGReg (Gaussian regularization of latent embeddings)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    ARPredictor,
    Embedder,
    MLP,
    SIGReg,
    ViTEncoder,
)


class JEPA(nn.Module):
    """Joint Embedding Predictive Architecture for world modeling.

    Components:
    - encoder: ViT that maps pixels → patch + CLS tokens
    - predictor: ARPredictor that forecasts future latent embeddings
    - action_encoder: Embeds actions for conditioning the predictor
    - projector: Projects encoder output to prediction space
    - pred_proj: Projects predictor output (mirror of projector)

    Training: next-embedding prediction (MSE) + SIGReg (Gaussian prior).
    Inference: MPC via get_cost() → CEM/iCEM solver.
    """

    def __init__(
        self,
        # Encoder
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 192,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        # Predictor (num_frames = context length, i.e. n_obs_steps - num_preds)
        num_frames: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 2048,
        predictor_dim_head: int = 64,
        # Projectors
        proj_hidden_dim: int = 2048,
        # Action encoder
        action_dim: int = 2,
        action_emb_dim: Optional[int] = None,
        # SIGReg
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        # Inference
        _interpolate_pos_encoding: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_num_frames = num_frames  # context length for predictor
        self.action_dim = action_dim
        self._interpolate_pos_encoding = _interpolate_pos_encoding

        action_emb_dim = action_emb_dim or embed_dim

        # Vision encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # Action encoder
        self.action_encoder = Embedder(
            input_dim=action_dim,
            emb_dim=action_emb_dim,
        )

        # Latent predictor (autoregressive, action-conditioned)
        self.predictor = ARPredictor(
            num_frames=num_frames,
            hidden_dim=embed_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_dim=predictor_mlp_dim,
            dim_head=predictor_dim_head,
        )

        # Projection heads
        self.projector = MLP(
            input_dim=embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=embed_dim,
            norm=True,
        )
        self.pred_proj = MLP(
            input_dim=embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=embed_dim,
            norm=True,
        )

        # Regularizer
        self.sigreg = SIGReg(
            knots=sigreg_knots,
            num_proj=sigreg_num_proj,
            embed_dim=embed_dim,
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode observations and actions into latent embeddings.

        Args:
            batch: dict with keys:
                - "observation.image" or "pixels": (B, T, C, H, W)
                - "action": (B, T, action_dim)

        Returns:
            dict with keys:
                - "emb": (B, T, D) projected CLS embeddings
                - "act_emb": (B, T, D) action embeddings
        """
        # Resolve pixel key
        pixels = batch.get("observation.image", batch.get("pixels"))
        if pixels is None:
            raise KeyError("Batch must contain 'observation.image' or 'pixels'")
        actions = batch["action"]

        B, T, C, H, W = pixels.shape

        # Flatten B/T for ViT encoding
        pixels_flat = pixels.reshape(B * T, C, H, W)
        tokens = self.encoder(pixels_flat)  # (B*T, N_tokens, D)
        cls_token = tokens[:, 0]  # (B*T, D) — CLS token

        # Project CLS token
        emb = self.projector(cls_token)  # (B*T, D)
        emb = emb.view(B, T, -1)  # (B, T, D)

        # Encode actions
        act_emb = self.action_encoder(actions)  # (B, T, D)

        return {"emb": emb, "act_emb": act_emb}

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next-step latent embeddings.

        Args:
            emb: (B, T, D) context embeddings.
            act_emb: (B, T, D) action embeddings.

        Returns:
            pred_emb: (B, T, D) predicted next embeddings.
        """
        pred = self.predictor(emb, act_emb)
        return self.pred_proj(pred.reshape(-1, pred.shape[-1])).view_as(pred)

    def rollout(
        self,
        info: dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive rollout for MPC.

        Given an initial context and a sequence of candidate actions,
        predicts future latent embeddings step by step.

        Args:
            info: dict with 'emb' (B, T_ctx, D) from encode().
            action_sequence: (B, S, H, action_dim) — S samples, horizon H.
            history_size: number of context embeddings to keep.

        Returns:
            info dict updated with 'predicted_emb' (B, S, H, D).
        """
        B, S, H, _ = action_sequence.shape
        ctx_emb = info["emb"]  # (B, T_ctx, D)

        # Initialize running context, flattened to (B*S, HS, D) for parallel sample processing
        context = ctx_emb[:, -history_size:]  # (B, HS, D)
        HS, D = context.shape[1], context.shape[2]
        # Expand for S samples: (B, HS, D) → (B*S, HS, D)
        context = context.unsqueeze(1).expand(B, S, HS, D).reshape(B * S, HS, D)
        pred_embs = []

        for step in range(H):
            # Embed actions for this step: (B, S, action_dim) → (B*S, 1, D)
            act = action_sequence[:, :, step]  # (B, S, action_dim)
            act_emb = self.action_encoder(act)  # (B, S, D)
            act_emb_flat = act_emb.reshape(B * S, 1, D)

            # Predict next embedding
            pred = self.predict(context, act_emb_flat)  # (B*S, HS, D)
            next_emb = pred[:, -1:]  # (B*S, 1, D)
            next_emb_view = next_emb.view(B, S, 1, D)  # (B, S, 1, D)
            pred_embs.append(next_emb_view)

            # Update context: shift window, append prediction
            context = torch.cat([
                context[:, 1:, :],   # drop oldest, (B*S, HS-1, D)
                next_emb,            # add prediction, (B*S, 1, D)
            ], dim=1)  # (B*S, HS, D)

        info["predicted_emb"] = torch.cat(pred_embs, dim=2)  # (B, S, H, D)
        return info

    def get_cost(
        self,
        info: dict[str, torch.Tensor],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost for action candidates (for MPC).

        Cost = MSE between predicted final embedding and goal embedding.

        Args:
            info: dict with 'emb' (context embeddings) and 'goal_emb'.
            action_candidates: (B, S, H, action_dim).

        Returns:
            costs: (B, S) cost per candidate.
        """
        B, S, H, _ = action_candidates.shape

        # Encode goal if not already present
        if "goal_emb" not in info:
            goal_info = self.encode(info)
            info["goal_emb"] = goal_info["emb"]

        # Rollout predictions
        info = self.rollout(info, action_candidates)

        # Cost: MSE between final predicted embedding and goal
        predicted_final = info["predicted_emb"][:, :, -1]  # (B, S, D)
        goal_final = info["goal_emb"][:, :, -1]  # (B, 1, D) or (B, D)

        if goal_final.dim() == 2:
            goal_final = goal_final.unsqueeze(1)  # (B, 1, D)

        # MSE per sample
        costs = ((predicted_final - goal_final) ** 2).mean(dim=-1)  # (B, S)
        return costs

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        num_preds: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Computes:
        - pred_loss: MSE between predicted and target embeddings
        - sigreg_loss: Gaussian regularization on latent embeddings

        Args:
            batch: dict with pixels, action, and temporal structure.
            num_preds: number of future steps to predict.

        Returns:
            dict with 'pred_loss', 'sigreg_loss', 'loss'.
        """
        info = self.encode(batch)
        emb = info["emb"]  # (B, T, D)
        act_emb = info["act_emb"]  # (B, T, D)

        # Split into context and targets.
        # num_frames = total frames in sequence (n_obs_steps).
        # ctx_len = num_frames - num_preds (keep all but last num_preds as context).
        # tgt_emb = frames offset by num_preds.
        total_frames = emb.shape[1]
        ctx_len = total_frames - num_preds
        ctx_emb = emb[:, :ctx_len]  # (B, ctx_len, D)
        ctx_act = act_emb[:, :ctx_len]  # (B, ctx_len, D)
        tgt_emb = emb[:, num_preds:]  # (B, ctx_len, D) — offset by num_preds

        # Predict
        pred_emb = self.predict(ctx_emb, ctx_act)  # (B, ctx_len, D)

        # Prediction loss (MSE)
        pred_loss = F.mse_loss(pred_emb, tgt_emb)

        # SIGReg loss
        sigreg_loss = self.sigreg(emb.transpose(0, 1))  # (T, B, D)

        return {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "loss": pred_loss + sigreg_loss,
            "emb": emb,
            "pred_emb": pred_emb,
        }
