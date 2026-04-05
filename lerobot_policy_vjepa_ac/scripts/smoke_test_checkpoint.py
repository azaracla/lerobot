#!/usr/bin/env python3
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lerobot_policy_vjepa_ac.configuration_vjepa_ac import VjepaAcConfig
from lerobot_policy_vjepa_ac.modeling_vjepa_ac import VjepaAcPolicy
from lerobot.configs.policies import PreTrainedConfig

CHECKPOINT_PATH = "outputs/train/2026-04-04/20-53-43_vjepa_ac/checkpoints/001000/pretrained_model"

print("Loading config...")
config = PreTrainedConfig.from_pretrained(CHECKPOINT_PATH)
print(f"  Policy type: {config.type}")
print(f"  Model: {config.model_name}")
print(f"  Action dim: {config.action_dim}")
print(f"  Embed dim: {config.embed_dim}")

config.cem_num_samples = 16
config.cem_num_iters = 2
config.mpc_horizon = 3
print(
    f"\n  Reduced CEM params for smoke test: cem_samples={config.cem_num_samples}, cem_iters={config.cem_num_iters}, mpc_horizon={config.mpc_horizon}"
)

print("\nCreating policy...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = VjepaAcPolicy(config).to(device)
policy.eval()

print("\nLoading weights...")
policy = VjepaAcPolicy.from_pretrained(CHECKPOINT_PATH, config=config)
policy.to(device)
policy.eval()
print("Weights loaded successfully!")

batch_size = 2
H = 384
W = 384
C = 3

print("\nCreating dummy batch (T=2 for real loss computation)...")
batch = {
    "observation.image.up": torch.randn(batch_size, C, 2, H, W, device=device),
    "observation.image.side": torch.randn(batch_size, C, 2, H, W, device=device),
    "observation.state": torch.randn(batch_size, 2, 6, device=device),
    "action": torch.randn(batch_size, 1, 6, device=device),  # T-1 = 1
}

print("\n=== Testing forward() for training loss ===")
with torch.no_grad():
    loss, loss_dict = policy(batch)
print(f"Loss: {loss.item():.6f}")
print(f"Loss dict: {loss_dict}")

print("\n=== Testing select_action() for inference ===")
with torch.no_grad():
    actions = policy.select_action(batch)
print(f"Predicted actions shape: {actions.shape}")
print(f"Actions stats: mean={actions.mean().item():.4f}, std={actions.std().item():.4f}")

print("\n" + "=" * 50)
print("SMOKE TEST PASSED!")
print("=" * 50)
