#!/usr/bin/env python
"""Test the full async inference pipeline locally without robot"""

import torch
from safetensors import safe_open
from lerobot_policy_vjepa_ac import VjepaAcPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline, PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

CKPT = "outputs/vjepa_ac/run_20260406_overfit_4/checkpoints/003000/pretrained_model"

print("Loading policy with dataset_stats...")
stats_file = f"{CKPT}/policy_preprocessor_step_3_normalizer_processor.safetensors"
with safe_open(stats_file, framework="pt") as f:
    state = {}
    for k in f.keys():
        v = f.get_tensor(k)
        state[k] = v.tolist() if v.numel() > 1 else v.item()

dataset_stats = {
    "observation.state": {"min": state["observation.state.min"], "max": state["observation.state.max"]}
}

policy = VjepaAcPolicy.from_pretrained(CKPT, dataset_stats=dataset_stats)
policy.to("cuda")
policy.eval()

postprocessor = PolicyProcessorPipeline.from_pretrained(
    CKPT,
    config_filename="policy_postprocessor.json",
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

print("Loaded! Simulating 5 timesteps with changing state...")

# Simulate robot state changing (as robot moves)
current_state = [0.0, 0.0, 45.0, 45.0, 20.0, 10.0]

for t in range(5):
    fake_image = torch.randn(1, 3, 720, 1280).cuda()
    fake_state = torch.tensor([current_state]).cuda()

    observation = {
        "observation.images.top": fake_image,
        "observation.state": fake_state,
    }

    with torch.no_grad():
        chunk = policy.predict_action_chunk(observation)
        action_norm = chunk[0, 0, :].cpu()
        result = postprocessor(action_norm.unsqueeze(0).unsqueeze(0).cuda())
        action = result[0, 0, :].cpu()

    print(f"T={t}: state={current_state}")
    print(f"  action (normalized): {action_norm.tolist()}")
    print(f"  action (joint pos):  {action.tolist()}")

    # Simulate robot moving (in reality, robot sends actual state)
    current_state = action.tolist()
