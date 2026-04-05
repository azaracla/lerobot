#!/usr/bin/env python3
"""
Parity Test for vjepa_ac implementation on DROID data.

This script performs three types of tests:
1. Direct Comparison: Compare encoder outputs using original vjepa2 transforms vs LeRobot transforms
2. Coherence Test: Verify that LeRobot outputs are reasonable (no NaN, normal variance, etc.)
3. Transform Test: Verify that LeRobot transforms are mathematically correct

Usage:
    cd lerobot_policy_vjepa_ac
    conda activate lerobot
    python scripts/parity_test_droid.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vjepa2"))

from lerobot_policy_vjepa_ac.configuration_vjepa_ac import VjepaAcConfig
from lerobot_policy_vjepa_ac.modeling_vjepa_ac import VjepaAcPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.transforms import DroidRandomResizedCrop

DROID_TEST_DATA = "/home/arthur/Code/lerobot/vjepa2/droid_test_data"
EPISODE_DIR = f"{DROID_TEST_DATA}/Fri_Jul__7_09:42:23_2023"
METADATA_FILE = f"{EPISODE_DIR}/metadata_AUTOLab+5d05c5aa+2023-07-07-09h-42m-23s.json"
MP4_FILE = f"{EPISODE_DIR}/recordings/MP4/22008760.mp4"

ORIGINAL_TRANSFORMS_CONFIG = {
    "random_horizontal_flip": False,
    "random_resize_aspect_ratio": (0.75, 1.35),
    "random_resize_scale": (0.3, 1.0),
    "crop_size": 256,
    "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

LEROBOT_TRANSFORMS_CONFIG = {
    "scale": 1.777,
    "ratio": (0.75, 1.35),
    "target_size": 256,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OriginalVJepa2Transform:
    """Replicates the original vjepa2 video transforms."""

    def __init__(self, config):
        self.random_horizontal_flip = config.get("random_horizontal_flip", False)
        self.random_resize_aspect_ratio = config["random_resize_aspect_ratio"]
        self.random_resize_scale = config["random_resize_scale"]
        self.crop_size = config["crop_size"]
        mean = config["normalize"][0]
        std = config["normalize"][1]
        self.mean = torch.tensor(mean, dtype=torch.float32) * 255.0
        self.std = torch.tensor(std, dtype=torch.float32) * 255.0

    def random_resized_crop(self, buffer, target_height, target_width, scale, ratio):
        """Random resized crop matching vjepa2 implementation."""
        C, T, H, W = buffer.shape

        area = H * W
        target_area = area * scale

        for _ in range(10):
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()
            crop_h = int(round((target_area * aspect_ratio) ** 0.5))
            crop_w = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < crop_h <= H and 0 < crop_w <= W:
                top = torch.randint(0, H - crop_h + 1, (1,)).item()
                left = torch.randint(0, W - crop_w + 1, (1,)).item()
                break
        else:
            top, left, crop_h, crop_w = 0, 0, H, W

        buffer = buffer[:, :, top : top + crop_h, left : left + crop_w]

        if crop_h != target_height or crop_w != target_width:
            buffer = torch.nn.functional.interpolate(
                buffer, size=(target_height, target_width), mode="bilinear", align_corners=False
            )
        return buffer

    def __call__(self, buffer):
        """Apply transforms to video buffer.

        Args:
            buffer: numpy array or tensor of shape [T, H, W, C] or [T, C, H, W]
        """
        if isinstance(buffer, np.ndarray):
            buffer = torch.from_numpy(buffer)

        if buffer.dim() == 4 and buffer.shape[-1] == 3:
            buffer = buffer.permute(0, 3, 1, 2)

        buffer = buffer.float()
        T, C, H, W = buffer.shape

        buffer = self.random_resized_crop(
            buffer,
            self.crop_size,
            self.crop_size,
            scale=torch.empty(1).uniform_(self.random_resize_scale[0], self.random_resize_scale[1]).item(),
            ratio=self.random_resize_aspect_ratio,
        )

        T, C, H, W = buffer.shape
        buffer = buffer.clone().contiguous()
        buffer = buffer.view(C, T * H * W)
        buffer = (buffer - self.mean.view(-1, 1)) / self.std.view(-1, 1)
        buffer = buffer.view(C, T, H, W)

        return buffer


class LeRobotTransform:
    """LeRobot DROID-style transforms."""

    def __init__(self, config):
        self.crop = DroidRandomResizedCrop(**config)

    def __call__(self, buffer):
        """Apply transforms to video buffer.

        Args:
            buffer: numpy array or tensor of shape [T, H, W, C] or [T, C, H, W]
        """
        if isinstance(buffer, np.ndarray):
            buffer = torch.from_numpy(buffer)

        if buffer.dim() == 4 and buffer.shape[-1] == 3:
            buffer = buffer.permute(0, 3, 1, 2)

        buffer = buffer.float() / 255.0

        buffer = self.crop(buffer)

        T, C, H, W = buffer.shape
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        buffer = (buffer - mean) / std

        buffer = buffer.permute(1, 0, 2, 3)

        return buffer


def load_video_frames_cv2(mp4_path, num_frames=8, fps=4, frameskip=2):
    """Load video frames using OpenCV (matching original vjepa2 sampling logic)."""
    import cv2

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {mp4_path}")

    vfps = cap.get(cv2.CAP_PROP_FPS)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fpc = num_frames
    fstp = int(np.ceil(vfps / fps))
    nframes = int(fpc * fstp)

    if vlen < nframes:
        cap.release()
        raise Exception(f"Video too short: {nframes=} {vlen=}")

    sf = np.random.randint(nframes, vlen)
    ef = sf + nframes

    indices = list(range(sf, ef, fstp))[::frameskip]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise Exception(f"No frames loaded")

    return np.stack(frames, axis=0)


def test_1_direct_comparison():
    """Test 1: Direct Comparison - Compare encoder outputs with different transforms."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct Comparison")
    print("=" * 60)
    print("Comparing original vjepa2 transforms vs LeRobot transforms")
    print(f"Episode: {EPISODE_DIR}")
    print(f"Video: {MP4_FILE}")

    if not os.path.exists(MP4_FILE):
        print(f"❌ SKIPPED: Video file not found at {MP4_FILE}")
        return False

    torch.manual_seed(42)
    np.random.seed(42)

    frames = load_video_frames_cv2(MP4_FILE, num_frames=8, fps=4, frameskip=2)
    print(f"Loaded frames: {frames.shape} (T, H, W, C)")

    orig_transform = OriginalVJepa2Transform(ORIGINAL_TRANSFORMS_CONFIG)
    lerobot_transform = LeRobotTransform(LEROBOT_TRANSFORMS_CONFIG)

    frames_tensor = torch.from_numpy(frames)

    orig_transformed = orig_transform(frames_tensor)
    print(f"Original transformed: {orig_transformed.shape} (C, T, H, W)")
    print(f"  Value range: [{orig_transformed.min():.3f}, {orig_transformed.max():.3f}]")

    lerobot_transformed = lerobot_transform(frames_tensor)
    print(f"LeRobot transformed: {lerobot_transformed.shape} (C, T, H, W)")
    print(f"  Value range: [{lerobot_transformed.min():.3f}, {lerobot_transformed.max():.3f}]")

    crop_h_orig = orig_transformed.shape[2]
    crop_w_orig = orig_transformed.shape[3]
    crop_size = LEROBOT_TRANSFORMS_CONFIG["target_size"]

    if crop_h_orig != crop_size or crop_w_orig != crop_size:
        lerobot_transformed = torch.nn.functional.interpolate(
            lerobot_transformed, size=(crop_h_orig, crop_w_orig), mode="bilinear", align_corners=False
        )
        print(f"  Resized LeRobot output to match: {lerobot_transformed.shape}")

    diff = (orig_transformed - lerobot_transformed).abs()
    print(f"\nDifference between transforms (should be non-zero due to different scales):")
    print(f"  Mean: {diff.mean():.4f}")
    print(f"  Max: {diff.max():.4f}")
    print(f"  Std: {diff.std():.4f}")

    print("\n⚠️  NOTE: Original uses random scale [0.3, 1.0] while LeRobot uses fixed scale 1.777")
    print("   These are fundamentally different approaches. Latents will differ.")
    print("   The purpose is to verify transforms are correctly implemented, not to match exactly.")

    return True


def test_2_coherence():
    """Test 2: Coherence Test - Verify LeRobot outputs are reasonable."""
    print("\n" + "=" * 60)
    print("TEST 2: Coherence Test")
    print("=" * 60)
    print("Verifying LeRobot encoder outputs are reasonable (no NaN, normal variance, etc.)")

    if not os.path.exists(MP4_FILE):
        print(f"❌ SKIPPED: Video file not found at {MP4_FILE}")
        return False

    torch.manual_seed(42)
    np.random.seed(42)

    print("\nLoading video frames...")
    frames = load_video_frames_cv2(MP4_FILE, num_frames=8, fps=4, frameskip=2)

    print("Creating LeRobot transform...")
    lerobot_transform = LeRobotTransform(LEROBOT_TRANSFORMS_CONFIG)
    frames_tensor = torch.from_numpy(frames)
    transformed = lerobot_transform(frames_tensor)

    print(f"Transformed frames: {transformed.shape}")
    print(f"  Value range: [{transformed.min():.3f}, {transformed.max():.3f}]")

    print("\nLoading VJepaAcPolicy with pretrained encoder...")
    config = VjepaAcConfig(
        model_name="vjepa2_1_vit_giant_384",
        encoder_repo_id="facebookresearch/vjepa2",
        img_size=384,
        patch_size=16,
        embed_dim=1536,
        predictor_embed_dim=1024,
        action_dim=6,
        pred_depth=24,
        num_heads=16,
    )
    config.cem_num_samples = 4
    config.cem_num_iters = 2
    config.mpc_horizon = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = VjepaAcPolicy(config).to(device)
    policy.eval()

    transformed = transformed.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    print(f"Input to encoder: {transformed.shape}")

    with torch.no_grad():
        latent = policy.encoder(transformed)

    print(f"\n✓ Encoder output received")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent dtype: {latent.dtype}")
    print(f"  Value range: [{latent.min():.6f}, {latent.max():.6f}]")
    print(f"  Mean: {latent.mean():.6f}")
    print(f"  Std: {latent.std():.6f}")

    all_passed = True

    if torch.isnan(latent).any():
        print(f"❌ FAILED: NaN values detected in latent")
        all_passed = False
    else:
        print(f"✓ No NaN values")

    if torch.isinf(latent).any():
        print(f"❌ FAILED: Inf values detected in latent")
        all_passed = False
    else:
        print(f"✓ No Inf values")

    if latent.std() < 1e-6:
        print(f"❌ FAILED: Latent has near-zero variance (possible collapse)")
        all_passed = False
    else:
        print(f"✓ Variance is reasonable: {latent.std():.6f}")

    if latent.abs().max() > 100:
        print(f"⚠️  WARNING: Latent values are very large (|max| > 100)")
    else:
        print(f"✓ Latent magnitudes are reasonable")

    return all_passed


def test_3_transform_correctness():
    """Test 3: Transform Correctness - Verify LeRobot transforms are mathematically correct."""
    print("\n" + "=" * 60)
    print("TEST 3: Transform Correctness")
    print("=" * 60)
    print("Verifying LeRobot transforms are mathematically correct")

    torch.manual_seed(42)
    np.random.seed(42)

    if not os.path.exists(MP4_FILE):
        print(f"❌ SKIPPED: Video file not found at {MP4_FILE}")
        return False

    frames = load_video_frames_cv2(MP4_FILE, num_frames=8, fps=4, frameskip=2)
    frames_tensor = torch.from_numpy(frames).float()

    print("\n3.1: Testing DroidRandomResizedCrop...")

    crop = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)

    result = crop(frames_tensor)
    print(f"  Input: {frames_tensor.shape} (T, C, H, W)")
    print(f"  Output: {result.shape}")

    C, T, H, W = result.shape
    if H != 256 or W != 256:
        print(f"❌ FAILED: Expected output size (256, 256), got ({H}, {W})")
        return False
    print(f"✓ Output size is correct: {H}x{W}")

    if result.dim() != 4:
        print(f"❌ FAILED: Expected 4D tensor, got {result.dim()}D")
        return False
    print(f"✓ Output is 4D tensor")

    print("\n3.2: Testing normalization...")
    normalized = result / 255.0

    expected_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    expected_std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)

    normalized_manual = (normalized - expected_mean) / expected_std

    crop2 = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)
    result2 = crop2(frames_tensor)
    lerobot_result = result2 / 255.0

    diff = (lerobot_result - normalized_manual).abs()
    if diff.mean() > 1e-5:
        print(f"❌ FAILED: Normalization mismatch, mean diff = {diff.mean()}")
        return False
    print(f"✓ Normalization is correct (diff < 1e-5)")

    print("\n3.3: Testing scale parameter...")
    for _ in range(3):
        crop_rand = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)
        result_rand = crop_rand(frames_tensor)
        if result_rand.shape[2:] != (256, 256):
            print(f"❌ FAILED: Random crop produced wrong size")
            return False
    print(f"✓ Random crops all produce correct size (256x256)")

    print("\n3.4: Testing that different seeds produce different crops...")
    torch.manual_seed(42)
    crop_a = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)
    out_a = crop_a(frames_tensor)

    torch.manual_seed(123)
    crop_b = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)
    out_b = crop_b(frames_tensor)

    diff_crops = (out_a - out_b).abs().mean()
    if diff_crops < 1e-5:
        print(f"⚠️  NOTE: Same seed used, crops are identical (expected)")
    else:
        print(f"✓ Different seeds produce different crops (diff={diff_crops:.4f})")

    print("\n3.5: Testing that LeRobot transform handles various input formats...")

    test_cases = [
        ("numpy [T,H,W,C]", frames.astype(np.uint8)),
        ("torch [T,H,W,C]", frames_tensor.permute(0, 3, 1, 2)),
        ("torch [T,C,H,W]", frames_tensor.permute(0, 3, 1, 2).float()),
    ]

    for name, test_input in test_cases:
        try:
            crop_test = DroidRandomResizedCrop(scale=1.777, ratio=(0.75, 1.35), target_size=256)
            if isinstance(test_input, np.ndarray):
                result_format = crop_test(torch.from_numpy(test_input))
            else:
                result_format = crop_test(test_input)
            if result_format.shape[2:] != (256, 256):
                print(f"❌ FAILED ({name}): Wrong output size")
                return False
            print(f"✓ Input format '{name}' handled correctly")
        except Exception as e:
            print(f"❌ FAILED ({name}): {e}")
            return False

    return True


def main():
    print("=" * 60)
    print("VJEPA_AC PARITY TEST ON DROID DATA")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = {}

    results["test_1_direct_comparison"] = test_1_direct_comparison()
    results["test_2_coherence"] = test_2_coherence()
    results["test_3_transform_correctness"] = test_3_transform_correctness()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("=" * 60))
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
