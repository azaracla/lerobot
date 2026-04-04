#!/usr/bin/env python3
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lerobot_policy_vjepa_ac.configuration_vjepa_ac import VjepaAcConfig
from lerobot_policy_vjepa_ac.modeling_vjepa_ac import VjepaAcPolicy

config = VjepaAcConfig(
    embed_dim=1408,
    predictor_embed_dim=1024,
    pred_depth=6, # Make it small for smoke test
    num_heads=8,
    action_dim=7,
    auto_steps=2,
    cem_num_samples=16,
    cem_num_iters=2,
    mpc_horizon=3,
)

# We use an empty device for the fake policy, or cpu
device = torch.device("cpu")
policy = VjepaAcPolicy(config).to(device)
policy.eval()

# Dummy batch
batch_size = 2
T_batch = 4
H = 384
W = 384
C = 3

batch = {
    "observation.image": torch.randn(batch_size, C, T_batch, H, W, device=device),
    "action": torch.randn(batch_size, T_batch - 1, 7, device=device),
    "observation.state": torch.randn(batch_size, T_batch, 7, device=device),
    "goal.image": torch.randn(batch_size, C, H, W, device=device)
}

print("Running forward() for training loss...")
with torch.no_grad():
    loss, loss_dict = policy(batch)
print("Loss shapes/values:", loss_dict)

print("\nRunning select_action() for inference...")
with torch.no_grad():
    actions = policy.select_action(batch)
print("Predicted actions shape:", actions.shape)
print("Smoke test PASSED!")
