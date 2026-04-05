#!/usr/bin/env python3
"""
Test to verify ImageNet normalization is correctly applied to images before the vjepa2 encoder.

The vjepa2 encoder was pretrained on DROID with ImageNet normalization:
- mean = (0.485, 0.456, 0.406)
- std = (0.229, 0.224, 0.225)

This script tests:
1. What the dataset actually provides (wrong normalization)
2. What the encoder expects (ImageNet normalization)
3. How to fix it by applying correct normalization
"""

import sys
import os
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vjepa2"))

from lerobot_policy_vjepa_ac.configuration_vjepa_ac import VjepaAcConfig
from lerobot_policy_vjepa_ac.modeling_vjepa_ac import VjepaAcPolicy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DROID_TEST_DATA = "/home/arthur/Code/lerobot/vjepa2/droid_test_data"
EPISODE_DIR = f"{DROID_TEST_DATA}/Fri_Jul__7_09:42:23_2023"
MP4_FILE = f"{EPISODE_DIR}/recordings/MP4/22008760.mp4"


def load_frame_cv2(mp4_path, frame_idx=0):
    """Load a single frame from video."""
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception(f"Failed to read frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float()  # [H, W, C] in [0, 255]


def apply_imagenet_normalization_to_tensor(tensor_c_h_w):
    """Apply ImageNet normalization to a [C, H, W] tensor."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor_c_h_w = tensor_c_h_w / 255.0
    return (tensor_c_h_w - mean) / std


def apply_dataset_normalization_to_tensor(tensor_c_h_w):
    """Apply dataset-specific normalization (what LeRobot actually provides)."""
    return tensor_c_h_w / 255.0


def main():
    print("=" * 60)
    print("IMAGE NET NORMALIZATION TEST FOR VJEPA_AC")
    print("=" * 60)

    if not os.path.exists(MP4_FILE):
        print(f"❌ SKIPPED: Video file not found at {MP4_FILE}")
        return 1

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n1. Loading frame from DROID video...")
    frame = load_frame_cv2(MP4_FILE, frame_idx=0)  # [H, W, C] in [0, 255]
    H, W = frame.shape[:2]
    print(f"   Raw frame: {frame.shape}, range=[{frame.min():.1f}, {frame.max():.1f}]")

    print("\n2. Loading vjepa2 encoder...")
    config = VjepaAcConfig(
        model_name="vjepa2_1_vit_giant_384",
        encoder_repo_id="facebookresearch/vjepa2",
        img_size=384,
        patch_size=16,
    )
    policy = VjepaAcPolicy(config).to(device)
    policy.eval()
    print(f"   Encoder loaded successfully")

    print("\n3. Testing different normalization approaches...")

    frame_c_h_w = frame.permute(2, 0, 1)  # [C, H, W]

    test_cases = [
        ("RAW [0,255] (WRONG)", frame_c_h_w.clone()),
        ("Dataset [0,1] no ImageNet", apply_dataset_normalization_to_tensor(frame_c_h_w.clone())),
        ("ImageNet normalized [0,1]", apply_imagenet_normalization_to_tensor(frame_c_h_w.clone())),
    ]

    results = []
    for name, img_tensor in test_cases:
        # Add batch and temporal dimensions: [C, H, W] -> [B, C, T, H, W]
        img_5d = img_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, H, W]

        with torch.no_grad():
            try:
                latent = policy.encoder(img_5d)
                latent_cpu = latent.cpu()

                print(f"\n   {name}:")
                print(f"     Input range: [{img_5d.min():.4f}, {img_5d.max():.4f}]")
                print(f"     Input mean per channel: {img_5d[0].mean(dim=(1, 2))}")
                print(f"     Latent shape: {latent.shape}")
                print(f"     Latent range: [{latent.min():.6f}, {latent.max():.6f}]")
                print(f"     Latent mean: {latent.mean():.6f}")
                print(f"     Latent std: {latent.std():.6f}")
                print(f"     Has NaN: {torch.isnan(latent).any()}")
                print(f"     Has Inf: {torch.isinf(latent).any()}")

                results.append((name, latent_cpu, None))
            except Exception as e:
                print(f"\n   {name}:")
                print(f"     ❌ ERROR: {e}")
                results.append((name, None, str(e)))

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if all(r[1] is not None for r in results):
        raw_latent = results[0][1]
        wrong_latent = results[1][1]
        correct_latent = results[2][1]

        print("\nComparing latents:")

        # Compare ImageNet vs Dataset normalization
        diff = (correct_latent - wrong_latent).abs()
        print(f"\n  ImageNet vs Dataset [0,1]:")
        print(f"    Mean difference: {diff.mean():.6f}")
        print(f"    Max difference: {diff.max():.6f}")
        print(f"    This shows Dataset [0,1] is NOT equivalent to ImageNet normalization!")

        # Check if raw [0,255] without normalization produces reasonable output
        print(f"\n  Raw [0,255] vs ImageNet:")
        diff_raw = (correct_latent - raw_latent).abs()
        print(f"    Mean difference: {diff_raw.mean():.6f}")

        # Check latent magnitudes
        print(f"\n  Latent magnitude comparison:")
        for name, latent, _ in results:
            print(
                f"    {name}: mean={latent.mean():.4f}, std={latent.std():.4f}, |max|={latent.abs().max():.4f}"
            )

        # Compare to what LeRobot dataset provides
        print("\n" + "=" * 60)
        print("CHECKING LEROBOT DATASET (svla_so101_pickplace)")
        print("=" * 60)

        try:
            from lerobot.datasets import LeRobotDataset

            ds = LeRobotDataset("lerobot/svla_so101_pickplace")
            item = ds[0]
            lr_img = item["observation.images.up"]  # [C, H, W]
            print(f"\n  LeRobot dataset image:")
            print(f"    Shape: {lr_img.shape}")
            print(f"    Range: [{lr_img.min():.4f}, {lr_img.max():.4f}]")
            print(f"    Mean per channel: {lr_img.mean(dim=(1, 2))}")
            print(f"    This is similar to 'Dataset [0,1]' normalization, NOT ImageNet!")
        except Exception as e:
            print(f"  Could not load LeRobot dataset: {e}")

        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        print("""
The vjepa2 encoder was pretrained with ImageNet normalization.
Your LeRobot dataset provides images in [0,1] without proper ImageNet stats.

The current vjepa_ac policy has:
  - normalization_mapping.VISUAL = IDENTITY (no normalization)

This means images go directly from dataset to encoder WITHOUT ImageNet normalization.

FIX NEEDED: 
1. Either change normalization_mapping.VISUAL to apply ImageNet normalization
2. Or preprocess images in the policy before the encoder
3. Or ensure the dataset applies ImageNet normalization during loading
""")
    else:
        print("\n⚠️  Some tests failed - see errors above")

    return 0


if __name__ == "__main__":
    sys.exit(main())
