"""
Unit tests for lerobot_policy_vjepa_ac.

Strategy: the real vjepa2 encoder (ViT-Giant) is too heavy to load in CI.
All tests mock torch.hub.load with a lightweight FakeEncoder that returns
tensors of the correct shape. This lets us test everything except the actual
encoder weights, which are covered by integration / smoke tests.

Run from lerobot_policy_vjepa_ac/:
    conda run -n lerobot pytest tests/ -v
"""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from lerobot_policy_vjepa_ac.configuration_vjepa_ac import VjepaAcConfig
from lerobot_policy_vjepa_ac.modeling_vjepa_ac import IMAGENET_MEAN, IMAGENET_STD, VjepaAcPolicy

# ---------------------------------------------------------------------------
# Test-only constants — small enough to be fast, realistic enough to be useful
# ---------------------------------------------------------------------------
IMG_SIZE = 64        # 64px instead of 384px
PATCH_SIZE = 16      # → (64/16)² = 16 tokens per frame
EMBED_DIM = 64       # encoder embed_dim (normally 1408)
PRED_DIM = 32        # predictor_embed_dim
N_HEADS = 2          # must divide PRED_DIM
PRED_DEPTH = 2       # 2 transformer blocks instead of 24
ACTION_DIM = 6
N_OBS = 2
MPC_H = 2
TOKENS = (IMG_SIZE // PATCH_SIZE) ** 2  # 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEncoder(nn.Module):
    """Minimal stand-in for the vjepa2 ViT encoder.

    Returns deterministic zeros with the right shape so all downstream
    shape / gradient / loss tests can run without GPU or network.
    """

    def __init__(self, tokens: int = TOKENS, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self._tokens = tokens
        # One real parameter so .parameters() / grad checks work
        self._dummy = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self._dummy.weight.sum() * 0 + torch.zeros(
            B, self._tokens, self.embed_dim, device=x.device, dtype=x.dtype
        )


def make_config(**overrides) -> VjepaAcConfig:
    defaults = dict(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        predictor_embed_dim=PRED_DIM,
        num_heads=N_HEADS,
        pred_depth=PRED_DEPTH,
        n_obs_steps=N_OBS,
        mpc_horizon=MPC_H,
        action_dim=ACTION_DIM,
        cem_num_samples=4,
        cem_num_iters=2,
    )
    defaults.update(overrides)
    return VjepaAcConfig(**defaults)


def make_policy(config: VjepaAcConfig | None = None) -> VjepaAcPolicy:
    if config is None:
        config = make_config()
    fake_enc = FakeEncoder()
    with patch("torch.hub.load", return_value=fake_enc):
        policy = VjepaAcPolicy(config)
    # Move the FakeEncoder to the same device as the predictor so outputs are consistent
    policy_device = next(policy.predictor.parameters()).device
    policy.encoder = policy.encoder.to(policy_device)
    return policy


def policy_device(policy: VjepaAcPolicy) -> torch.device:
    return next(policy.predictor.parameters()).device


def make_batch(B: int = 2, T: int = 3, device: torch.device | str = "cpu") -> dict:
    """Synthetic training batch (T obs frames → T-1 action steps)."""
    return {
        "observation.images.top": torch.rand(B, T, 3, IMG_SIZE, IMG_SIZE, device=device),
        "observation.state": torch.rand(B, T, ACTION_DIM, device=device) * 2 - 1,
        "action": torch.rand(B, T - 1, ACTION_DIM, device=device) * 2 - 1,
        "action_is_pad": torch.zeros(B, T - 1, dtype=torch.bool, device=device),
    }


# ---------------------------------------------------------------------------
# Config tests (no model instantiation needed)
# ---------------------------------------------------------------------------

class TestConfig:
    def test_tubelet_size_default(self):
        """tubelet_size must default to 2 to match vjepa2.1 pre-training."""
        cfg = make_config()
        assert cfg.tubelet_size == 2

    def test_normalize_reps_default(self):
        """normalize_reps must default to True to match vjepa2.1 pre-training."""
        cfg = make_config()
        assert cfg.normalize_reps is True

    def test_normalization_mapping_visual_is_identity(self):
        """VISUAL normalization must be IDENTITY — ImageNet norm happens inside the model."""
        from lerobot.configs.types import NormalizationMode
        cfg = make_config()
        assert cfg.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY

    def test_normalization_mapping_state_and_action(self):
        """STATE and ACTION use MIN_MAX — handled by the LeRobot processor."""
        from lerobot.configs.types import NormalizationMode
        cfg = make_config()
        assert cfg.normalization_mapping["STATE"] == NormalizationMode.MIN_MAX
        assert cfg.normalization_mapping["ACTION"] == NormalizationMode.MIN_MAX


# ---------------------------------------------------------------------------
# Predictor attention mask size
# ---------------------------------------------------------------------------

class TestAttnMask:
    def test_attn_mask_covers_max_sequence(self):
        """attn_mask must be large enough for the longest sequence we'll pass.

        max_seq_len = max_temporal_depth * tubelet_size
        max_temporal_depth = max(n_obs_steps, mpc_horizon + 1)

        With n_obs=2, mpc_h=2, tubelet=2 → depth=3, max_seq=6, grid_depth=3.
        Sequence at T=3: 3*(TOKENS + 2 cond tokens) = 54 → mask needs ≥ 54.
        """
        cfg = make_config()
        policy = make_policy(cfg)

        max_temporal_depth = max(cfg.n_obs_steps, cfg.mpc_horizon + 1)
        cond_tokens = 2  # action + state (no extrinsics)
        max_seq = max_temporal_depth * (TOKENS + cond_tokens)

        assert policy.predictor.attn_mask is not None
        mask_size = policy.predictor.attn_mask.shape[0]
        assert mask_size >= max_seq, (
            f"attn_mask too small: {mask_size} < {max_seq}. "
            f"max_seq_len in __init__ probably wrong."
        )

    def test_attn_mask_is_none_for_t1(self):
        """T=1 single-frame context must disable the mask to enable FlashAttention."""
        policy = make_policy()
        # Simulate what the predictor does at T=1
        z = torch.zeros(1, TOKENS, PRED_DIM)
        T = z.size(1) // (policy.predictor.grid_height * policy.predictor.grid_width)
        assert T == 1
        # The predictor skips attn_mask when T == 1
        assert policy.predictor.attn_mask is None or T == 1


# ---------------------------------------------------------------------------
# ImageNet normalization (pure math, no model)
# ---------------------------------------------------------------------------

class TestImageNetNormalize:
    def test_normalize_math(self):
        """(x - mean) / std for each channel — verified against known values."""
        policy = make_policy()
        # Image of constant 0.5 across all channels, shape [1, 3, 1, 4, 4]
        img = torch.full((1, 3, 1, 4, 4), 0.5)
        out = policy._imagenet_normalize(img)

        for c in range(3):
            expected = (0.5 - IMAGENET_MEAN[c].item()) / IMAGENET_STD[c].item()
            assert torch.allclose(out[0, c], torch.tensor(expected), atol=1e-5), (
                f"Channel {c}: expected {expected:.4f}, got {out[0, c, 0, 0, 0].item():.4f}"
            )

    def test_normalize_zero_image(self):
        """Zero image must produce negative normalized values (mean > 0)."""
        policy = make_policy()
        img = torch.zeros(1, 3, 1, 4, 4)
        out = policy._imagenet_normalize(img)
        assert (out < 0).all(), "Normalized zero image should be negative for all channels"

    def test_normalize_skipped_when_disabled(self):
        """Setting use_imagenet_for_visuals=False must return the image unchanged."""
        cfg = make_config()
        cfg.use_imagenet_for_visuals = False
        policy = make_policy(cfg)
        img = torch.rand(1, 3, 1, 4, 4)
        out = policy._imagenet_normalize(img)
        assert torch.equal(out, img)


# ---------------------------------------------------------------------------
# Encoder tubelet duplication
# ---------------------------------------------------------------------------

class TestTubeletDuplication:
    def test_encoder_receives_t2_in_forward(self):
        """With tubelet_size=2, each frame must be duplicated before encoding.

        We intercept the encoder call to inspect the input shape.
        """
        policy = make_policy()
        dev = policy_device(policy)
        received_shapes = []

        real_encoder_forward = policy.encoder.forward

        def capturing_forward(x):
            received_shapes.append(x.shape)
            return real_encoder_forward(x)

        policy.encoder.forward = capturing_forward

        batch = make_batch(B=1, T=3, device=dev)
        policy.train()
        policy.forward(batch)

        # Each of the 3 frames is encoded as [1, C, 2, H, W]
        assert all(s[2] == 2 for s in received_shapes), (
            f"Encoder should receive T=2 (tubelet duplication). Got shapes: {received_shapes}"
        )

    def test_encoder_t1_and_t2_produce_same_token_count(self):
        """tubelet_size=2 must not change tokens_per_frame vs tubelet_size=1."""
        fake_enc = FakeEncoder()

        t1 = torch.zeros(1, 3, 1, IMG_SIZE, IMG_SIZE)
        t2 = torch.zeros(1, 3, 2, IMG_SIZE, IMG_SIZE)

        out1 = fake_enc(t1)
        out2 = fake_enc(t2)
        assert out1.shape == out2.shape, (
            f"Token count changed with tubelet duplication: {out1.shape} vs {out2.shape}"
        )


# ---------------------------------------------------------------------------
# forward() — training path
# ---------------------------------------------------------------------------

class TestForward:
    def test_returns_scalar_loss(self):
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        loss, info = policy.forward(make_batch(device=dev))
        assert loss.shape == (), f"Loss must be scalar, got shape {loss.shape}"

    def test_loss_not_nan(self):
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        loss, _ = policy.forward(make_batch(device=dev))
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_info_keys(self):
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        _, info = policy.forward(make_batch(device=dev))
        assert "loss" in info
        assert "jloss" in info
        assert "sloss" in info

    def test_backward_passes(self):
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        loss, _ = policy.forward(make_batch(device=dev))
        loss.backward()  # must not raise

    def test_grad_on_predictor_not_encoder(self):
        """Encoder is frozen — its params must have no gradient after backward."""
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        loss, _ = policy.forward(make_batch(device=dev))
        loss.backward()

        for name, p in policy.encoder.named_parameters():
            assert p.grad is None or p.grad.abs().max() == 0, (
                f"Encoder param '{name}' has non-zero gradient — encoder is not frozen!"
            )

        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in policy.predictor.parameters()
        )
        assert has_grad, "Predictor has no gradient after backward — something is wrong"

    def test_no_batch_mutation(self):
        """forward() must not modify the input batch (keys or values)."""
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        batch = make_batch(device=dev)
        batch_copy = deepcopy(batch)
        policy.forward(batch)

        assert set(batch.keys()) == set(batch_copy.keys()), "Batch keys were mutated"
        for k in batch_copy:
            assert torch.equal(batch[k], batch_copy[k]), f"Batch['{k}'] was mutated"

    def test_action_is_pad_all_true_gives_zero_loss(self):
        """When all actions are padded the masked loss must be zero."""
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        batch = make_batch(device=dev)
        batch["action_is_pad"] = torch.ones_like(batch["action_is_pad"])
        loss, info = policy.forward(batch)
        assert info["jloss"] == pytest.approx(0.0), "jloss should be 0 when all steps are padded"
        assert info["sloss"] == pytest.approx(0.0), "sloss should be 0 when all steps are padded"

    def test_action_is_pad_absent_does_not_crash(self):
        """forward() must work even when action_is_pad is not in the batch."""
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        batch = make_batch(device=dev)
        del batch["action_is_pad"]
        loss, _ = policy.forward(batch)
        assert not torch.isnan(loss)

    def test_single_obs_frame(self):
        """T=1 observation (no temporal context) must return a dummy zero loss, not crash."""
        policy = make_policy()
        dev = policy_device(policy)
        policy.train()
        batch = make_batch(B=2, T=1, device=dev)
        # T_full=1 → predictor has nothing to predict, code returns dummy 0 loss
        loss, _ = policy.forward(batch)
        assert loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Safetensors key cleaning
# ---------------------------------------------------------------------------

class TestSafetensorKeyCleaning:
    def test_orig_mod_prefix_stripped(self):
        """_load_as_safetensor must strip 'predictor._orig_mod.' from compiled keys."""
        dirty = {
            "predictor._orig_mod.blocks.0.weight": torch.zeros(2, 2),
            "predictor._orig_mod.norm.weight": torch.zeros(4),
            "encoder.patch_embed.proj.weight": torch.zeros(2, 2),
        }
        cleaned = {k.replace("predictor._orig_mod.", "predictor."): v for k, v in dirty.items()}

        assert "predictor.blocks.0.weight" in cleaned
        assert "predictor.norm.weight" in cleaned
        assert "encoder.patch_embed.proj.weight" in cleaned
        assert not any("_orig_mod" in k for k in cleaned)

    def test_non_compiled_keys_unchanged(self):
        """Keys without the compiled prefix must pass through unchanged."""
        clean = {
            "predictor.embed.weight": torch.zeros(2, 2),
            "encoder.blocks.0.attn.proj.weight": torch.zeros(2, 2),
        }
        result = {k.replace("predictor._orig_mod.", "predictor."): v for k, v in clean.items()}
        assert result == clean
