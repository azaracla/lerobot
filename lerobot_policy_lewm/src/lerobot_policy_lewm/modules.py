"""Building blocks for the LeWM JEPA world model.

Ported from https://github.com/lucas-maes/le-wm (MIT License).
Includes: ViT encoder, Transformer, ARPredictor, Embedder, SIGReg, etc.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# ViT Encoder (wraps timm for a clean interface)
# ---------------------------------------------------------------------------


class ViTEncoder(nn.Module):
    """Vision Transformer encoder, wraps timm ViT.

    Uses patch_size=14, image_size=224, embed_dim=192 (ViT-Tiny config).
    Outputs all tokens: CLS token + patch tokens.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=0,
            global_pool="",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into patch + CLS tokens.

        Args:
            x: (B, C, H, W) images, normalized with ImageNet stats.

        Returns:
            tokens: (B, num_patches + 1, embed_dim) — CLS token first.
        """
        return self.vit.forward_features(x)

    @property
    def num_tokens(self) -> int:
        return self.num_patches + 1


# ---------------------------------------------------------------------------
# FeedForward (MLP block)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation and dropout."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention with pre-LayerNorm and optional causal mask."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention (context=None) or cross-attention (context given)."""
        x_norm = self.norm(x)
        ctx = x_norm if context is None else context

        qkv = self.to_qkv(ctx).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(*t.shape[:-1], self.heads, self.dim_head).transpose(1, 2),
            qkv,
        )

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(context is None))
        out = out.transpose(1, 2).reshape(*x.shape[:-1], -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Standard Transformer Block (pre-LN)
# ---------------------------------------------------------------------------


class Block(nn.Module):
    """Standard pre-LN transformer block: attention + MLP with residuals."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Conditional Block (AdaLN-Zero modulation)
# ---------------------------------------------------------------------------


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning on action embeddings.

    The condition vector is projected into 6 modulation parameters:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp.

    Only the adaLN modulation projection is zero-initialized (as in DiT/LeWM).
    Attention and MLP paths keep standard init so gate gradients can flow.
    At t=0: gate=0 → identity + normally-initialized sub-blocks → gradients flow.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        inner_dim = dim_head * heads

        # Attention sub-block (standard init, NOT zero — gradients flow through gate)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # MLP sub-block (standard init, NOT zero)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        # Modulation projection: cond -> 6 * dim
        # Zero-init weight + bias as in original LeWM / DiT AdaLN-zero design.
        # Gates start at 0, but gradients flow through gate because attn/mlp
        # outputs are non-zero (standard init). Gates gradually open.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward with AdaLN conditioning.

        Args:
            x: (B, N, dim) token sequence.
            c: (B, N, cond_dim) per-token condition (action embeddings).

        Returns:
            (B, N, dim) transformed tokens.
        """
        # Modulation: (B, N, 6*dim) -> 6 x (B, N, dim)
        mod = self.adaLN_modulation(c)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention sub-block with modulation
        x_norm = self.norm1(x) * (1 + scale_attn) + shift_attn
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(*t.shape[:-1], self.heads, self.dim_head).transpose(1, 2),
            qkv,
        )
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(*x.shape[:-1], -1)
        x = x + gate_attn * self.to_out(attn_out)

        # MLP sub-block with modulation
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)

        return x


# ---------------------------------------------------------------------------
# Transformer (generic, dispatches on block type)
# ---------------------------------------------------------------------------


class Transformer(nn.Module):
    """Generic transformer with optional conditional blocks.

    Dispatches between standard Block and ConditionalBlock based on
    whether a condition projection is needed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
        use_conditional: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

        if cond_dim is not None and use_conditional:
            self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
            self.blocks = nn.ModuleList([
                ConditionalBlock(
                    dim=hidden_dim,
                    cond_dim=hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])
        else:
            self.cond_proj = None
            self.blocks = nn.ModuleList([
                Block(
                    dim=hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None and self.cond_proj is not None:
            c = self.cond_proj(c)
            for block in self.blocks:
                x = block(x, c)
        else:
            for block in self.blocks:
                x = block(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# MLP Projector
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Two-layer MLP with configurable normalization and GELU activation.

    Used as projector/pred_proj in JEPA: projects encoder CLS tokens
    to the prediction space and back.

    Uses configurable normalization (default: LayerNorm).
    The original le-wm uses BatchNorm1d for training-time regularization.

    Expects input of shape (B*T, D).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        norm: bool = True,
        norm_fn=None,
    ):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]
        if norm:
            if norm_fn is not None:
                layers.append(norm_fn(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Action Embedder
# ---------------------------------------------------------------------------


class Embedder(nn.Module):
    """Encodes raw actions into action embeddings for the predictor.

    Uses Conv1d (kernel=1) + 2-layer MLP with SiLU activation.
    Input: (B, T, action_dim), Output: (B, T, emb_dim).

    Matches original le-wm module.Embedder exactly.
    """

    def __init__(
        self,
        input_dim: int = 10,
        smoothed_dim: int = 10,
        emb_dim: int = 10,
        mlp_scale: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, int(mlp_scale * emb_dim)),
            nn.SiLU(),
            nn.Linear(int(mlp_scale * emb_dim), emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


# ---------------------------------------------------------------------------
# ARPredictor — Autoregressive latent predictor
# ---------------------------------------------------------------------------


class ARPredictor(nn.Module):
    """Autoregressive predictor operating in latent space.

    Takes context embeddings and action embeddings, produces
    predictions of future latent embeddings.

    Uses: learned positional embeddings + Transformer with AdaLN-Zero
    conditioning on action embeddings.
    """

    def __init__(
        self,
        num_frames: int = 3,
        hidden_dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # Learnable positional embeddings for up to num_frames positions
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            cond_dim=hidden_dim,
            use_conditional=True,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict future embeddings.

        Args:
            x: (B, T, D) context latent embeddings.
            c: (B, T, D) action embeddings (conditioning).

        Returns:
            (B, T, D) predicted next-step embeddings.
        """
        B, T, D = x.shape
        # Add positional embeddings
        x = x + self.pos_embed[:, :T, :]
        x = self.dropout(x)
        x = self.transformer(x, c=c)
        return x


# ---------------------------------------------------------------------------
# SIGReg — Sketch Isotropic Gaussian Regularizer
# ---------------------------------------------------------------------------


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)

    Exact copy from https://github.com/lucas-maes/le-wm (MIT License).
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
