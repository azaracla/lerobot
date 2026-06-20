#!/usr/bin/env python
"""End-to-end training smoke test with synthetic data.

Validates the full pipeline: config → policy → forward → backward → optimizer step.
Run: python scripts/train_smoke_test.py
"""

import sys
import torch
import numpy as np

from lerobot_policy_lewm import LeWMConfig, LeWMPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


def create_synthetic_batch(batch_size=2, n_obs_steps=4, img_size=224, action_dim=2):
    """Create a synthetic batch mimicking PushT data."""
    return {
        "observation.image": torch.randn(batch_size, n_obs_steps, 3, img_size, img_size),
        "action": torch.randn(batch_size, n_obs_steps, action_dim),
    }


def main():
    print("=" * 60)
    print("Training Smoke Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create config (smaller model for fast testing)
    config = LeWMConfig(
        n_obs_steps=4,
        num_preds=1,
        img_size=112,
        embed_dim=192,
        encoder_depth=4,
        encoder_heads=3,
        predictor_depth=2,
        predictor_heads=8,
        predictor_mlp_dim=512,
        proj_hidden_dim=512,
        action_emb_dim=192,
        sigreg_knots=5,
        sigreg_num_proj=64,
        device=device,
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 112, 112)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )

    # Create policy
    print("\n[1] Creating policy...")
    policy = LeWMPolicy(config)
    policy.to(device)
    policy.train()
    n_params = sum(p.numel() for p in policy.model.parameters())
    print(f"    Params: {n_params:,}")

    # Create optimizer
    print("\n[2] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
    )

    # Training loop
    print("\n[3] Running training loop...")
    num_steps = 50
    losses = []

    for step in range(num_steps):
        batch = create_synthetic_batch(batch_size=4, n_obs_steps=4, img_size=112, action_dim=2)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        loss, info = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"    Step {step:3d}: loss={loss.item():.6f}, "
                  f"pred_loss={info['pred_loss']:.6f}, "
                  f"sigreg_loss={info['sigreg_loss']:.6f}")

    # Check loss is decreasing
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    print(f"\n    Initial loss (avg first 5): {initial_loss:.6f}")
    print(f"    Final loss (avg last 5):   {final_loss:.6f}")
    print(f"    Loss ratio: {final_loss / initial_loss:.3f}")

    assert final_loss < initial_loss, (
        f"Loss should decrease during training! "
        f"Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
    )

    # Test inference after training
    print("\n[4] Testing inference after training...")
    policy.eval()
    batch_inf = {
        "observation.image": torch.randn(1, 1, 3, 112, 112, device=device),
        "action": torch.randn(1, 1, 2, device=device),
    }
    with torch.no_grad():
        action = policy.select_action(batch_inf)
    print(f"    Action: {action.cpu().numpy()[0]}")

    print("\n" + "=" * 60)
    print("Training smoke test passed!")
    print(f"Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
