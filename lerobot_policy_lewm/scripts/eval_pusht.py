#!/usr/bin/env python
"""Evaluate LeWM on PushT using swm/PushT-v1 (stable-worldmodel env).

Matches the LeWM eval protocol:
- Uses swm PushT which provides image observations and goal images
- CEM-based MPC plans towards the env's actual goal configuration
- Success measured by the environment

Usage:
    python scripts/eval_pusht.py --checkpoint outputs/lewm_pusht_10k/final
"""

import argparse, os, sys, time, numpy as np, torch, torch.nn.functional as F
import gymnasium as gym

# Monkey-patch pymunk for backward compat
import pymunk
if not hasattr(pymunk.Space, 'on_collision'):
    def _on_collision(self, a, b, begin=None, pre_solve=None, post_solve=None, separate=None):
        h = self.add_collision_handler(a, b)
        if begin: h.begin = begin
        if pre_solve: h.pre_solve = pre_solve
        if post_solve: h.post_solve = post_solve
        if separate: h.separate = separate
        return h
    pymunk.Space.on_collision = _on_collision

import stable_worldmodel as swm  # noqa: E402 — registers swm/PushT-v1
from lerobot_policy_lewm import LeWMPolicy  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def image_to_input(rgb, device):
    """Convert numpy RGB image (H,W,C) to model input batch (1,1,C,H,W)."""
    img = torch.from_numpy(rgb).float() / 255.0  # (H, W, C)
    img = img.permute(2, 0, 1).unsqueeze(0)       # (1, C, H, W)
    img = (img - MEAN.view(1, 3, 1, 1)) / STD.view(1, 3, 1, 1)
    img = img.unsqueeze(1).to(device)              # (1, 1, C, H, W)
    return {"observation.image": img, "action": torch.zeros(1, 1, 2, device=device)}


def run_episode(env, policy, device, max_steps=300):
    """Run one MPC episode with proper goal conditioning."""
    obs, info = env.reset()
    policy.reset()

    # Get goal image from env (rendered at reset, shows goal configuration)
    goal_rgb = info["goal"]  # (H, W, C) uint8
    goal_tensor = image_to_input(goal_rgb, device)["observation.image"]
    policy.set_goal(goal_tensor)

    # Prime embedding cache with actual env observation
    img_rgb = env.render()  # (H, W, C) uint8
    batch = image_to_input(img_rgb, device)
    with torch.no_grad():
        emb_info = policy.model.encode(batch)
        policy._update_embedding_cache(emb_info["emb"])

    total_reward = 0.0
    for step in range(max_steps):
        img_rgb = env.render()
        batch = image_to_input(img_rgb, device)

        with torch.no_grad():
            action = policy.select_action(batch)
        action_np = action.cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward

        if terminated or truncated:
            return {"success": bool(info.get("success", False)),
                    "steps": step + 1, "total_reward": total_reward}

    return {"success": bool(info.get("success", False)),
            "steps": max_steps, "total_reward": total_reward}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--cem_samples", type=int, default=80)
    p.add_argument("--cem_iters", type=int, default=8)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading {args.checkpoint}...")
    policy = LeWMPolicy.from_pretrained(args.checkpoint)
    policy.to(device).eval()
    policy.config.cem_num_samples = args.cem_samples
    policy.config.cem_n_steps = args.cem_iters
    print(f"  Params: {sum(p.numel() for p in policy.model.parameters()):,}")
    print(f"  CEM: {args.cem_samples}s × {args.cem_iters}i")

    # Create env
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("swm/PushT-v1", render_mode=render_mode)
    print(f"  Action: {env.action_space}")

    # Eval
    print(f"\n{args.episodes} episodes (max {args.max_steps} steps)...\n")
    ok, steps_l, rewards_l, times = [], [], [], []

    for ep in range(args.episodes):
        t0 = time.time()
        r = run_episode(env, policy, device, args.max_steps)
        dt = time.time() - t0
        ok.append(r["success"]); steps_l.append(r["steps"]); rewards_l.append(r["total_reward"]); times.append(dt)
        s = "✅" if r["success"] else "❌"
        print(f"  Ep {ep+1:2d}: {s}  steps={r['steps']:3d}  reward={r['total_reward']:6.1f}  {dt:.1f}s")

    rate = sum(ok) / len(ok) * 100
    print(f"\n{'='*50}")
    print(f"Success: {rate:.1f}% ({sum(ok)}/{len(ok)})")
    print(f"Avg steps: {np.mean(steps_l):.0f}  Avg reward: {np.mean(rewards_l):.1f}  Avg time: {np.mean(times):.1f}s")
    print(f"{'='*50}")
    env.close()


if __name__ == "__main__":
    main()
