#!/usr/bin/env python
"""
Final validation test for VJEPa AC implementation.

This script tests:
1. Loading the frozen ViT encoder from PyTorch Hub
2. Creating the VJEPa AC policy
3. Running a forward pass (training)
4. Running CEM inference
5. Testing custom DroidRandomResizedCrop transform
"""

import torch
import numpy as np
from lerobot_policy_vjepa_ac import VjepaAcPolicy, VjepaAcConfig
from lerobot_policy_vjepa_ac.transforms import DroidRandomResizedCrop


def test_encoder_loading():
    """Test 1: Load ViT encoder from PyTorch Hub"""
    print("=" * 60)
    print("Test 1: Loading ViT encoder from PyTorch Hub")
    print("=" * 60)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_output = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_giant_384")
        if isinstance(encoder_output, tuple):
            encoder = encoder_output[0]
        else:
            encoder = encoder_output
        print(f"✓ Encoder loaded successfully on {device}")
        print(f"  Embed dim: {encoder.embed_dim}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"✗ Failed to load encoder: {e}")
        return False


def test_policy_creation():
    """Test 2: Create VJEPa AC policy"""
    print("\n" + "=" * 60)
    print("Test 2: Creating VJEPa AC policy")
    print("=" * 60)
    try:
        config = VjepaAcConfig(
            model_name="vjepa2_1_vit_giant_384",
            encoder_repo_id="facebookresearch/vjepa2",
            action_dim=6,
            img_size=256,
            predictor_embed_dim=1024,
            pred_depth=24,
            num_heads=16,
            mpc_horizon=15,
            cem_num_samples=100,
            use_imagenet_for_visuals=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = VjepaAcPolicy(config, device=device)
        print(f"✓ Policy created successfully on {device}")
        print(f"  Total params: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")
        print(f"  Learnable params: {sum(p.numel() for p in policy.get_optim_params()) / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"✗ Failed to create policy: {e}")
        return False


def test_training_forward():
    """Test 3: Run forward pass (training)"""
    print("\n" + "=" * 60)
    print("Test 3: Running training forward pass")
    print("=" * 60)
    try:
        config = VjepaAcConfig(
            model_name="vjepa2_1_vit_giant_384",
            encoder_repo_id="facebookresearch/vjepa2",
            action_dim=6,
            img_size=256,
            predictor_embed_dim=1024,
            pred_depth=24,
            num_heads=16,
            use_imagenet_for_visuals=True,
            auto_steps=2,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = VjepaAcPolicy(config, device=device)

        B, T, C, H, W = 2, 4, 3, 256, 256
        batch = {
            "observation.images.up": torch.randn(B, C, T, H, W, device=device),
            "action": torch.randn(B, T - 1, config.action_dim, device=device),
            "observation.state": torch.randn(B, T, config.action_dim, device=device),
        }

        loss, info = policy(batch)
        print(f"✓ Forward pass successful")
        print(f"  Loss: {info['loss']:.6f}")
        print(f"  Joint loss (teacher forcing): {info['jloss']:.6f}")
        print(f"  Sequential loss (autoregressive): {info['sloss']:.6f}")
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cem_inference():
    """Test 4: Run CEM inference"""
    print("\n" + "=" * 60)
    print("Test 4: Running CEM inference")
    print("=" * 60)
    try:
        config = VjepaAcConfig(
            model_name="vjepa2_1_vit_giant_384",
            encoder_repo_id="facebookresearch/vjepa2",
            action_dim=6,
            img_size=256,
            predictor_embed_dim=1024,
            pred_depth=24,
            num_heads=16,
            mpc_horizon=5,  # Short horizon for faster test
            cem_num_samples=50,  # Few samples for faster test
            cem_num_iters=3,  # Few iterations for faster test
            use_imagenet_for_visuals=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = VjepaAcPolicy(config, device=device)

        B, C, H, W = 2, 3, 256, 256
        batch = {
            "observation.images.up": torch.randn(B, C, H, W, device=device),
            "observation.state": torch.randn(B, config.action_dim, device=device),
        }

        with torch.no_grad():
            actions = policy.select_action(batch)

        print(f"✓ CEM inference successful")
        print(f"  Action shape: {actions.shape}")
        print(f"  Expected: ({B}, {config.action_dim})")
        print(f"  Action mean: {actions.mean(dim=0)}")
        print(f"  Action std: {actions.std(dim=0)}")
        assert actions.shape == (B, config.action_dim), f"Wrong action shape: {actions.shape}"
        assert not torch.isnan(actions).any(), "Actions contain NaN!"
        return True
    except Exception as e:
        print(f"✗ CEM inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_custom_transform():
    """Test 5: Test DroidRandomResizedCrop"""
    print("\n" + "=" * 60)
    print("Test 5: Testing DroidRandomResizedCrop transform")
    print("=" * 60)
    try:
        transform = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)

        # Test with single image [C, H, W]
        img_single = torch.randn(3, 480, 640)
        out_single = transform(img_single)
        print(f"✓ Single image transform successful")
        print(f"  Input: {img_single.shape}")
        print(f"  Output: {out_single.shape}")
        assert out_single.shape == (3, 256, 256), f"Wrong output shape: {out_single.shape}"

        # Test with video [T, C, H, W]
        img_video = torch.randn(4, 3, 480, 640)
        out_video = transform(img_video)
        print(f"✓ Video transform successful")
        print(f"  Input: {img_video.shape}")
        print(f"  Output: {out_video.shape}")
        assert out_video.shape == (4, 3, 256, 256), f"Wrong output shape: {out_video.shape}"

        return True
    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("VJEPa AC Implementation Validation Test")
    print("=" * 60)

    results = []
    results.append(("Encoder loading", test_encoder_loading()))
    results.append(("Policy creation", test_policy_creation()))
    results.append(("Training forward pass", test_training_forward()))
    results.append(("CEM inference", test_cem_inference()))
    results.append(("Custom transform", test_custom_transform()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
