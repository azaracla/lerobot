#!/usr/bin/env python
"""Script to find episodes with sufficient length for VJEPA AC training.

This filters out episodes that are too short to provide the required
observation history (n_obs_steps frames at fps spacing).

Usage:
    python scripts/filter_long_episodes.py --repo_id lerobot/svla_so101_pickplace --n_obs_steps 8 --fps 4
"""

import argparse
import numpy as np
from lerobot.datasets import LeRobotDatasetMetadata


def compute_min_episode_length(n_obs_steps: int, fps: int, vfps: int = 30) -> int:
    """Compute minimum episode length needed for given observation params.

    With n_obs_steps=8 and fps=4 and vfps=30:
    - frame_step = round(30/4) = 8
    - indices = [0, -8, -16, -24, -32, -40, -48, -56]
    - Need at least 57 frames in episode
    """
    frame_step = round(vfps / fps)
    max_backward = abs(frame_step * (n_obs_steps - 1))
    return max_backward + 1


def filter_long_episodes(repo_id: str, n_obs_steps: int, fps: int, vfps: int = 30) -> list[int]:
    """Filter episodes that are long enough for training."""
    min_length = compute_min_episode_length(n_obs_steps, fps, vfps)
    print(f"Looking for episodes with at least {min_length} frames...")
    print(f"  (n_obs_steps={n_obs_steps}, fps={fps}, vfps={vfps})")

    meta = LeRobotDatasetMetadata(repo_id)
    ep_lengths = meta.episodes["length"]

    valid_mask = ep_lengths >= min_length
    valid_episodes = meta.episodes[valid_mask]["episode_index"].tolist()

    print(f"\nFound {len(valid_episodes)}/{len(ep_lengths)} valid episodes")

    if len(valid_episodes) > 0:
        print(f"\nEpisode indices to use in config:")
        print(f"  episodes: {valid_episodes[:50]}")  # Show first 50
        if len(valid_episodes) > 50:
            print(f"  ... and {len(valid_episodes) - 50} more")

        # Also show episode lengths for reference
        valid_lengths = ep_lengths[valid_mask].tolist()
        print(f"\nCorresponding lengths: {valid_lengths[:50]}")

    return valid_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter episodes by length for VJEPA AC training")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo ID")
    parser.add_argument("--n_obs_steps", type=int, default=8, help="Number of observation steps")
    parser.add_argument("--fps", type=int, default=4, help="Target FPS for temporal sampling")
    parser.add_argument("--vfps", type=int, default=30, help="Original video FPS")

    args = parser.parse_args()

    valid_episodes = filter_long_episodes(args.repo_id, args.n_obs_steps, args.fps, args.vfps)
