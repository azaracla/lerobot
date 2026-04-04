#!/usr/bin/env python3
"""
Parity test: can lerobot_policy_vjepa_ac's VisionTransformerPredictorAC
load the official Meta pretrained checkpoint vjepa2-ac-vitg.pt ?

Usage:
    python scripts/parity_test.py
    python scripts/parity_test.py --ckpt /path/to/vjepa2-ac-vitg.pt
"""

import argparse
import sys
import os
import urllib.request

import torch

# -- allow importing from the package without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lerobot_policy_vjepa_ac.ac_predictor_utils import VisionTransformerPredictorAC, vit_ac_predictor

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
DEFAULT_CKPT_PATH = os.path.expanduser("~/.cache/vjepa2/vjepa2-ac-vitg.pt")


def download_if_needed(url: str, dest: str) -> str:
    if os.path.exists(dest):
        print(f"✓ Checkpoint already cached at {dest}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} → {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"✓ Downloaded ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def inspect_checkpoint(ckpt: dict) -> dict:
    """Print checkpoint top-level keys and infer model config from predictor weights."""
    print("\n=== Checkpoint keys ===")
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    # -- strip DDP 'module.' prefix if present
    pred_state = ckpt.get("predictor", {})
    if hasattr(pred_state, "items"):
        pred_state = {k.replace("module.", ""): v for k, v in pred_state.items()}

    print("\n=== Predictor keys (first 20) ===")
    for i, (k, v) in enumerate(pred_state.items()):
        if i >= 20:
            print("  ...")
            break
        print(f"  {k}: {tuple(v.shape)}")

    # -- infer config from weights
    config = {}
    if "predictor_embed.weight" in pred_state:
        pred_dim, enc_dim = pred_state["predictor_embed.weight"].shape
        config["predictor_embed_dim"] = pred_dim
        config["embed_dim"] = enc_dim
        print(f"\n→ Inferred embed_dim         = {enc_dim}")
        print(f"→ Inferred predictor_embed_dim = {pred_dim}")

    if "action_encoder.weight" in pred_state:
        _, action_dim = pred_state["action_encoder.weight"].shape
        config["action_embed_dim"] = action_dim
        print(f"→ Inferred action_embed_dim  = {action_dim}")

    if "predictor_blocks.0.attn.qkv.weight" in pred_state:
        qkv_out, qkv_in = pred_state["predictor_blocks.0.attn.qkv.weight"].shape
        config["dim"] = qkv_in
        print(f"→ Inferred block dim (predictor_embed_dim) = {qkv_in}")

    n_blocks = sum(1 for k in pred_state if k.startswith("predictor_blocks.") and k.endswith(".norm1.weight"))
    config["depth"] = n_blocks
    print(f"→ Inferred depth             = {n_blocks}")

    return config, pred_state


def run_parity_test(pred_state: dict, config: dict):
    """Attempt to load the predictor weights into our implementation."""
    from functools import partial
    import torch.nn as nn

    print("\n=== Building model from inferred config ===")
    embed_dim = config.get("embed_dim", 1408)
    predictor_embed_dim = config.get("predictor_embed_dim", 1024)
    action_embed_dim = config.get("action_embed_dim", 7)
    depth = config.get("depth", 24)

    model = VisionTransformerPredictorAC(
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=depth,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        use_rope=True,
        use_silu=False,
        is_frame_causal=True,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print("\n=== Loading state dict ===")
    result = model.load_state_dict(pred_state, strict=False)

    missing = result.missing_keys
    unexpected = result.unexpected_keys

    if not missing and not unexpected:
        print("✅  PERFECT MATCH — all keys loaded successfully, no missing or unexpected keys.")
    else:
        if missing:
            print(f"⚠️  Missing keys ({len(missing)}):")
            for k in missing[:10]:
                print(f"    {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        if unexpected:
            print(f"⚠️  Unexpected keys ({len(unexpected)}):")
            for k in unexpected[:10]:
                print(f"    {k}")
            if len(unexpected) > 10:
                print(f"    ... and {len(unexpected) - 10} more")

    # -- forward pass smoke test
    print("\n=== Forward pass smoke test ===")
    T_grid = 16 // 2  # num_frames // tubelet_size
    H = W = 224 // 16  # 14
    B = 1

    x = torch.randn(B, T_grid * H * W, embed_dim)
    actions = torch.randn(B, T_grid, action_embed_dim)
    states = torch.randn(B, T_grid, action_embed_dim)

    model.eval()
    with torch.no_grad():
        out = model(x, actions, states)

    expected_shape = (B, T_grid * H * W, embed_dim)
    if out.shape == expected_shape:
        print(f"✅  Output shape {tuple(out.shape)} — correct!")
    else:
        print(f"❌  Output shape {tuple(out.shape)}, expected {expected_shape}")

    return len(missing) == 0 and len(unexpected) == 0


def main():
    parser = argparse.ArgumentParser(description="Parity test for vjepa2-ac-vitg.pt")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint (downloaded if not given)")
    parser.add_argument("--no-download", action="store_true", help="Fail if checkpoint not cached")
    args = parser.parse_args()

    ckpt_path = args.ckpt or DEFAULT_CKPT_PATH

    if not os.path.exists(ckpt_path):
        if args.no_download:
            print(f"❌ Checkpoint not found at {ckpt_path} and --no-download is set.")
            sys.exit(1)
        ckpt_path = download_if_needed(CHECKPOINT_URL, ckpt_path)

    print(f"\nLoading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config, pred_state = inspect_checkpoint(ckpt)
    success = run_parity_test(pred_state, config)

    print("\n" + ("=" * 50))
    if success:
        print("✅  PARITY TEST PASSED")
    else:
        print("❌  PARITY TEST FAILED — see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
