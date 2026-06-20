"""Unit tests for LeWMPolicy (LeRobot integration)."""

import pytest
import torch
import tempfile
import os

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_lewm import LeWMConfig, LeWMPolicy


@pytest.fixture
def config():
    return LeWMConfig(
        n_obs_steps=4,
        num_preds=1,
        img_size=112,
        embed_dim=192,  # must be divisible by encoder_heads=3 (timm ViT constraint)
        encoder_depth=4,
        encoder_heads=3,  # 192/3 = 64
        predictor_depth=2,
        predictor_heads=8,
        predictor_mlp_dim=512,
        predictor_dim_head=64,
        proj_hidden_dim=512,
        action_emb_dim=192,
        sigreg_knots=5,
        sigreg_num_proj=64,
        cem_num_samples=20,
        cem_n_steps=5,
        cem_topk=5,
        device="cpu",
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 112, 112)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )


@pytest.fixture
def policy(config):
    return LeWMPolicy(config)


class TestLeWMPolicyInit:
    def test_creates_policy(self, config):
        policy = LeWMPolicy(config)
        assert policy.name == "lewm"
        assert policy.config_class == LeWMConfig

    def test_model_params_count(self, policy):
        n_params = sum(p.numel() for p in policy.model.parameters())
        # Small test config: ViT-tiny reduced + small predictor + projectors
        assert 2_000_000 < n_params < 20_000_000

    def test_config_type(self, config):
        assert config.type == "lewm"
        assert config.history_size == 3


class TestLeWMPolicyForward:
    def test_forward_returns_loss(self, policy):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        loss, info = policy.forward(batch)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert "pred_loss" in info
        assert "sigreg_loss" in info

    def test_forward_backward(self, policy):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        loss, info = policy.forward(batch)
        loss.backward()

        # Check encoder gradients
        encoder_grad_norm = sum(
            p.grad.norm().item()
            for p in policy.model.encoder.parameters()
            if p.grad is not None
        )
        assert encoder_grad_norm > 0, "Encoder should receive gradients"

        # Check predictor gradients
        predictor_grad_norm = sum(
            p.grad.norm().item()
            for p in policy.model.predictor.parameters()
            if p.grad is not None
        )
        assert predictor_grad_norm > 0, "Predictor should receive gradients"

    def test_forward_no_nan(self, policy):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        loss, info = policy.forward(batch)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_single_batch(self, policy):
        """Forward pass should work with batch_size=1."""
        batch = {
            "observation.image": torch.randn(1, 4, 3, 112, 112),
            "action": torch.randn(1, 4, 2),
        }
        loss, info = policy.forward(batch)
        assert loss.ndim == 0


class TestLeWMPolicySelectAction:
    def test_select_action_shape(self, policy):
        batch = {
            "observation.image": torch.randn(1, 1, 3, 112, 112),
            "action": torch.randn(1, 1, 2),
        }
        with torch.no_grad():
            action = policy.select_action(batch)
        assert action.shape == (1, 2)

    def test_select_action_no_nan(self, policy):
        batch = {
            "observation.image": torch.randn(1, 1, 3, 112, 112),
            "action": torch.randn(1, 1, 2),
        }
        with torch.no_grad():
            action = policy.select_action(batch)
        assert not torch.isnan(action).any()

    def test_select_action_with_embedding_cache(self, policy):
        """Multiple select_action calls should use the embedding cache."""
        batch = {
            "observation.image": torch.randn(1, 1, 3, 112, 112),
            "action": torch.randn(1, 1, 2),
        }
        with torch.no_grad():
            a1 = policy.select_action(batch)
            a2 = policy.select_action(batch)
            a3 = policy.select_action(batch)
        # All actions should have the same shape
        assert a1.shape == (1, 2)
        assert a2.shape == (1, 2)
        assert a3.shape == (1, 2)

    def test_reset_clears_cache(self, policy):
        batch = {
            "observation.image": torch.randn(1, 1, 3, 112, 112),
            "action": torch.randn(1, 1, 2),
        }
        with torch.no_grad():
            policy.select_action(batch)
        assert "emb" in policy._obs_cache
        policy.reset()
        assert "emb" not in policy._obs_cache

    def test_predict_action_chunk(self, policy):
        batch = {
            "observation.image": torch.randn(1, 1, 3, 112, 112),
            "action": torch.randn(1, 1, 2),
        }
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)
        assert chunk.shape == (1, policy.config.horizon, 2)


class TestLeWMPolicyCheckpoint:
    def test_save_load(self, policy):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            policy.save_pretrained(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))

            # Load
            loaded = LeWMPolicy.from_pretrained(tmpdir)
            assert loaded.name == "lewm"

            # Forward pass should produce the same output
            batch = {
                "observation.image": torch.randn(1, 4, 3, 112, 112),
                "action": torch.randn(1, 4, 2),
            }
            orig_loss, _ = policy.forward(batch)
            loaded_loss, _ = loaded.forward(batch)
            assert torch.allclose(orig_loss, loaded_loss, atol=1e-4)
