"""Unit tests for JEPA world model (encode, predict, forward, rollout, get_cost)."""

import pytest
import torch

from lerobot_policy_lewm.jepa import JEPA


@pytest.fixture
def jepa():
    """Small JEPA for fast testing."""
    return JEPA(
        img_size=112,
        patch_size=14,
        embed_dim=128,
        encoder_depth=4,
        encoder_heads=4,
        num_frames=3,
        predictor_depth=2,
        predictor_heads=8,
        predictor_mlp_dim=512,
        predictor_dim_head=64,
        proj_hidden_dim=512,
        action_dim=2,
        action_emb_dim=128,
        sigreg_knots=5,
        sigreg_num_proj=64,
    )


class TestJEPAEncode:
    def test_encode_shape(self, jepa):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        info = jepa.encode(batch)
        assert info["emb"].shape == (2, 4, 128)
        assert info["act_emb"].shape == (2, 4, 128)

    def test_encode_with_pixels_key(self, jepa):
        batch = {
            "pixels": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        info = jepa.encode(batch)
        assert info["emb"].shape == (2, 4, 128)

    def test_encode_missing_image_raises(self, jepa):
        batch = {"action": torch.randn(2, 4, 2)}
        with pytest.raises(KeyError):
            jepa.encode(batch)


class TestJEPAPredict:
    def test_predict_shape(self, jepa):
        emb = torch.randn(2, 3, 128)
        act_emb = torch.randn(2, 3, 128)
        pred = jepa.predict(emb, act_emb)
        assert pred.shape == (2, 3, 128)

    def test_predict_differentiable(self, jepa):
        emb = torch.randn(2, 3, 128, requires_grad=True)
        act_emb = torch.randn(2, 3, 128)
        pred = jepa.predict(emb, act_emb)
        pred.sum().backward()
        assert emb.grad is not None


class TestJEPAForward:
    def test_forward_returns_loss(self, jepa):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        output = jepa(batch, num_preds=1)
        assert "loss" in output
        assert "pred_loss" in output
        assert "sigreg_loss" in output
        assert output["loss"].ndim == 0  # scalar
        assert output["loss"].item() > 0

    def test_forward_backward(self, jepa):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        output = jepa(batch, num_preds=1)
        output["loss"].backward()
        # Check gradients flow to encoder and predictor
        encoder_grad = sum(
            p.grad.norm().item() for p in jepa.encoder.parameters() if p.grad is not None
        )
        predictor_grad = sum(
            p.grad.norm().item() for p in jepa.predictor.parameters() if p.grad is not None
        )
        assert encoder_grad > 0
        assert predictor_grad > 0

    def test_no_nan(self, jepa):
        batch = {
            "observation.image": torch.randn(2, 4, 3, 112, 112),
            "action": torch.randn(2, 4, 2),
        }
        output = jepa(batch, num_preds=1)
        assert not torch.isnan(output["loss"])
        assert not torch.isinf(output["loss"])


class TestJEPARollout:
    def test_rollout_shape(self, jepa):
        # Setup: encode some frames
        batch = {
            "observation.image": torch.randn(1, 4, 3, 112, 112),
            "action": torch.randn(1, 4, 2),
        }
        info = jepa.encode(batch)
        info["emb"] = info["emb"][:, :3]  # 3 context frames

        # Action candidates: (B=1, S=5, H=3, A=2)
        candidates = torch.randn(1, 5, 3, 2)

        with torch.no_grad():
            jepa.eval()
            info = jepa.rollout(info, candidates, history_size=3)

        assert "predicted_emb" in info
        assert info["predicted_emb"].shape == (1, 5, 3, 128)

    def test_rollout_single_sample(self, jepa):
        batch = {
            "observation.image": torch.randn(1, 4, 3, 112, 112),
            "action": torch.randn(1, 4, 2),
        }
        info = jepa.encode(batch)
        info["emb"] = info["emb"][:, :3]

        candidates = torch.randn(1, 1, 3, 2)  # S=1

        with torch.no_grad():
            jepa.eval()
            info = jepa.rollout(info, candidates, history_size=3)

        assert info["predicted_emb"].shape == (1, 1, 3, 128)


class TestJEPAGetCost:
    def test_get_cost_shape(self, jepa):
        batch = {
            "observation.image": torch.randn(1, 4, 3, 112, 112),
            "action": torch.randn(1, 4, 2),
        }
        info = jepa.encode(batch)
        info["emb"] = info["emb"][:, :3]  # context
        info["goal_emb"] = info["emb"][:, -1:]  # last context frame as goal

        candidates = torch.randn(1, 10, 3, 2)  # (B=1, S=10, H=3, A=2)

        with torch.no_grad():
            jepa.eval()
            costs = jepa.get_cost(info, candidates)

        assert costs.shape == (1, 10)  # (B, S)

    def test_get_cost_values_reasonable(self, jepa):
        batch = {
            "observation.image": torch.randn(1, 4, 3, 112, 112),
            "action": torch.randn(1, 4, 2),
        }
        info = jepa.encode(batch)
        info["emb"] = info["emb"][:, :3]
        info["goal_emb"] = info["emb"][:, -1:]

        candidates = torch.randn(1, 20, 3, 2)

        with torch.no_grad():
            jepa.eval()
            costs = jepa.get_cost(info, candidates)

        assert not torch.isnan(costs).any()
        assert not torch.isinf(costs).any()
        assert costs.min() >= 0  # MSE cost should be non-negative
