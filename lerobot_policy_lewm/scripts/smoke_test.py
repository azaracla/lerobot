#!/usr/bin/env python
"""Smoke test for LeWM policy: instantiate, forward pass, select_action."""

import sys
import torch
import numpy as np

from lerobot_policy_lewm import LeWMConfig, LeWMPolicy, JEPA
from lerobot.configs.types import FeatureType, PolicyFeature


def main():
    print("=" * 60)
    print("LeWM Policy Smoke Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Create config
    print("\n[1] Creating LeWMConfig...")
    config = LeWMConfig(
        n_obs_steps=4,
        num_preds=1,
        device=device,
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )
    print(f"    Type: {config.type}")
    print(f"    History size: {config.history_size}")
    print(f"    n_obs_steps: {config.n_obs_steps}")

    # 2. Create policy
    print("\n[2] Creating LeWMPolicy...")
    policy = LeWMPolicy(config)
    policy.to(device)
    n_params = sum(p.numel() for p in policy.model.parameters())
    print(f"    Total params: {n_params:,}")

    # 3. Forward pass (training)
    print("\n[3] Running forward pass...")
    B, T = 2, 4  # batch=2, n_obs_steps=4
    batch = {
        "observation.image": torch.randn(B, T, 3, 224, 224, device=device),
        "action": torch.randn(B, T, 2, device=device),  # T actions (same as frames)
    }
    loss, info = policy.forward(batch)
    print(f"    Loss: {loss.item():.6f}")
    print(f"    pred_loss: {info.get('pred_loss', 'N/A'):.6f}")
    print(f"    sigreg_loss: {info.get('sigreg_loss', 'N/A'):.6f}")

    # 4. Backward pass
    print("\n[4] Running backward pass...")
    loss.backward()
    total_grad = sum(
        p.grad.norm().item()
        for p in policy.model.parameters()
        if p.grad is not None
    )
    print(f"    Total gradient norm: {total_grad:.4f}")

    # 5. select_action (inference)
    print("\n[5] Running select_action...")
    batch_inf = {
        "observation.image": torch.randn(1, 1, 3, 224, 224, device=device),
        "action": torch.randn(1, 1, 2, device=device),
    }
    with torch.no_grad():
        action = policy.select_action(batch_inf)
    print(f"    Action shape: {action.shape}")
    print(f"    Action: {action.cpu().numpy()}")

    # 6. Checkpoint save/load
    print("\n[6] Testing checkpoint save/load...")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        policy.save_pretrained(tmpdir)
        print(f"    Saved to {tmpdir}")
        for f in sorted(os.listdir(tmpdir)):
            print(f"      {f}")

        # Load
        loaded = LeWMPolicy.from_pretrained(tmpdir)
        loaded.to(device)
        print(f"    Loaded params: {sum(p.numel() for p in loaded.model.parameters()):,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
