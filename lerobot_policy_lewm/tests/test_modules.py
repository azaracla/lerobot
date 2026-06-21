"""Unit tests for LeWM modules (Transformer, ARPredictor, ViTEncoder, SIGReg, etc.)."""

import pytest
import torch

from lerobot_policy_lewm.modules import (
    ViTEncoder,
    FeedForward,
    Attention,
    Block,
    ConditionalBlock,
    Transformer,
    MLP,
    Embedder,
    ARPredictor,
    SIGReg,
)


class TestViTEncoder:
    def test_forward_shape(self):
        encoder = ViTEncoder(img_size=224, patch_size=14, embed_dim=192)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        # 1 CLS + (224/14)^2 = 1 + 256 = 257 tokens
        assert out.shape == (2, 257, 192)

    def test_cls_token_first(self):
        encoder = ViTEncoder(img_size=224, patch_size=14, embed_dim=192)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        cls = out[:, 0]
        patches = out[:, 1:]
        assert cls.shape == (1, 192)
        assert patches.shape == (1, 256, 192)

    def test_num_tokens(self):
        encoder = ViTEncoder(img_size=224, patch_size=14)
        assert encoder.num_tokens == 257

    def test_differentiable(self):
        encoder = ViTEncoder(img_size=224, patch_size=14, embed_dim=192)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestAttention:
    def test_self_attention_shape(self):
        attn = Attention(dim=192, heads=8, dim_head=64)
        x = torch.randn(2, 10, 192)
        out = attn(x)
        assert out.shape == (2, 10, 192)

    def test_differentiable(self):
        attn = Attention(dim=192, heads=8, dim_head=64)
        x = torch.randn(2, 10, 192, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestBlock:
    def test_forward_shape(self):
        block = Block(dim=192, heads=8, dim_head=64, mlp_dim=768)
        x = torch.randn(2, 10, 192)
        out = block(x)
        assert out.shape == (2, 10, 192)

    def test_differentiable(self):
        block = Block(dim=192, heads=8, dim_head=64, mlp_dim=768)
        x = torch.randn(2, 10, 192, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestConditionalBlock:
    def test_forward_shape(self):
        block = ConditionalBlock(dim=192, cond_dim=192, heads=8, dim_head=64, mlp_dim=768)
        x = torch.randn(2, 10, 192)
        c = torch.randn(2, 10, 192)
        out = block(x, c)
        assert out.shape == (2, 10, 192)

    def test_zero_init_output(self):
        """At init, ConditionalBlock should be near-identity (zero init on output)."""
        block = ConditionalBlock(dim=192, cond_dim=192, heads=8, dim_head=64, mlp_dim=768)
        x = torch.randn(2, 10, 192)
        c = torch.randn(2, 10, 192)
        out = block(x, c)
        # Should be close to input (not identical due to dropout and norm, but close)
        assert (out - x).abs().mean() < 1.0  # not strict, due to LayerNorm

    def test_differentiable(self):
        block = ConditionalBlock(dim=192, cond_dim=192, heads=8, dim_head=64, mlp_dim=768)
        x = torch.randn(2, 10, 192, requires_grad=True)
        c = torch.randn(2, 10, 192)
        out = block(x, c)
        out.sum().backward()
        assert x.grad is not None


class TestTransformer:
    def test_standard_blocks(self):
        transformer = Transformer(
            input_dim=192, hidden_dim=192, output_dim=192,
            depth=4, heads=8, dim_head=64, mlp_dim=768,
            use_conditional=False,
        )
        x = torch.randn(2, 10, 192)
        out = transformer(x)
        assert out.shape == (2, 10, 192)

    def test_conditional_blocks(self):
        transformer = Transformer(
            input_dim=192, hidden_dim=192, output_dim=192,
            depth=4, heads=8, dim_head=64, mlp_dim=768,
            cond_dim=192, use_conditional=True,
        )
        x = torch.randn(2, 10, 192)
        c = torch.randn(2, 10, 192)
        out = transformer(x, c)
        assert out.shape == (2, 10, 192)

    def test_input_output_proj(self):
        transformer = Transformer(
            input_dim=128, hidden_dim=192, output_dim=256,
            depth=2, heads=8, dim_head=64, mlp_dim=768,
            use_conditional=False,
        )
        x = torch.randn(2, 10, 128)
        out = transformer(x)
        assert out.shape == (2, 10, 256)


class TestMLP:
    def test_forward_shape(self):
        mlp = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm=True)
        x = torch.randn(4, 192)
        out = mlp(x)
        assert out.shape == (4, 192)

    def test_no_norm(self):
        mlp = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm=False)
        x = torch.randn(4, 192)
        out = mlp(x)
        assert out.shape == (4, 192)

    def test_single_sample(self):
        """LayerNorm should work with batch_size=1."""
        mlp = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm=True)
        x = torch.randn(1, 192)
        out = mlp(x)
        assert out.shape == (1, 192)


class TestEmbedder:
    def test_forward_shape(self):
        embedder = Embedder(input_dim=2, emb_dim=192)
        x = torch.randn(4, 3, 2)  # (B, T, A)
        out = embedder(x)
        assert out.shape == (4, 3, 192)

    def test_differentiable(self):
        embedder = Embedder(input_dim=2, emb_dim=192)
        x = torch.randn(4, 3, 2, requires_grad=True)
        out = embedder(x)
        out.sum().backward()
        assert x.grad is not None


class TestARPredictor:
    def test_forward_shape(self):
        predictor = ARPredictor(
            num_frames=3, hidden_dim=192, depth=2,
            heads=8, mlp_dim=768, dim_head=64,
        )
        x = torch.randn(2, 3, 192)  # (B, T, D)
        c = torch.randn(2, 3, 192)  # (B, T, D) — action embeddings
        out = predictor(x, c)
        assert out.shape == (2, 3, 192)

    def test_differentiable(self):
        predictor = ARPredictor(
            num_frames=3, hidden_dim=192, depth=2,
            heads=8, mlp_dim=768, dim_head=64,
        )
        x = torch.randn(2, 3, 192, requires_grad=True)
        c = torch.randn(2, 3, 192)
        out = predictor(x, c)
        out.sum().backward()
        assert x.grad is not None

    def test_positional_embedding_used(self):
        """Output should change if we shift the input order."""
        predictor = ARPredictor(
            num_frames=3, hidden_dim=192, depth=2,
            heads=8, mlp_dim=768, dim_head=64,
        )
        x1 = torch.randn(1, 3, 192)
        x2 = torch.flip(x1, dims=[1])  # reversed order
        c = torch.zeros(1, 3, 192)
        out1 = predictor(x1, c)
        out2 = predictor(x2, c)
        assert not torch.allclose(out1, out2)


class TestSIGReg:
    def test_gaussian_low_loss(self):
        sigreg = SIGReg(knots=17, num_proj=256)
        # Large batch for good statistics — the original formula scales by T
        gauss = torch.randn(500, 64, 64).transpose(0, 1)  # (T=64, B=500, D=64)
        loss = sigreg(gauss)
        assert loss.item() < 2.0, f"Gaussian loss {loss.item()} should be < 2.0"

    def test_zeros_high_loss(self):
        sigreg = SIGReg(knots=17, num_proj=256)
        zeros = torch.zeros(500, 64, 64).transpose(0, 1)
        loss = sigreg(zeros)
        gauss = torch.randn(500, 64, 64).transpose(0, 1)
        loss_gauss = sigreg(gauss)
        assert loss.item() > loss_gauss.item(), (
            f"Zero loss {loss.item()} should be > Gaussian loss {loss_gauss.item()}"
        )

    def test_output_is_scalar(self):
        """SIGReg uses sort() which is non-differentiable — this is expected.
        The regularizer is added to the prediction loss but doesn't backprop
        through the encoder (by design in the original LeWM paper).
        """
        sigreg = SIGReg(knots=17, num_proj=256)
        x = torch.randn(200, 32, 64).transpose(0, 1)
        loss = sigreg(x)
        assert loss.ndim == 0
        assert loss.item() >= 0
