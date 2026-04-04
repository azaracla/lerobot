#!/usr/bin/env python
"""
Convert HuggingFaceVLA/community_dataset_v1 from v2.1 to v3.0 format.

This script:
1. Downloads the community_dataset_v1 from HuggingFace
2. Converts it to LeRobot v3.0 format
3. Saves to the specified local path

Usage:
    python scripts/convert_community_dataset.py \
        --output_dir /data/datasets/community_so100_v3

The converted dataset can then be used for VJEPA AC training with:
    --dataset.repo_id=local --dataset.root=/data/datasets/community_so100_v3
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_community_dataset(
    output_dir: str,
    hf_token: str | None = None,
    push_to_hub: bool = False,
):
    """
    Convert community_dataset_v1 from v2.1 to v3.0.

    Args:
        output_dir: Local directory to save the converted dataset
        hf_token: HuggingFace token for authentication (if needed)
        push_to_hub: Whether to push the converted dataset to Hub
    """
    output_path = Path(output_dir)

    source_repo_id = "HuggingFaceVLA/community_dataset_v1"

    logger.info(f"Starting conversion of {source_repo_id}")
    logger.info(f"Output directory: {output_path}")

    if output_path.exists():
        logger.warning(f"Output directory already exists: {output_path}")
        response = input("This will overwrite existing data. Continue? [y/N] ")
        if response.lower() != "y":
            logger.info("Conversion cancelled.")
            return
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "lerobot.scripts.convert_dataset_v21_to_v30",
        "--repo-id",
        source_repo_id,
        "--root",
        str(output_path),
        "--push-to-hub=false",
    ]

    if hf_token:
        import os

        os.environ["HF_TOKEN"] = hf_token

    import subprocess

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(f"Conversion failed with return code {result.returncode}")
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    logger.info(f"Conversion complete! Dataset saved to: {output_path}")

    logger.info("\nVerifying converted dataset...")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local", root=str(output_path))
        logger.info(f"  Frames: {dataset.num_frames}")
        logger.info(f"  Episodes: {dataset.num_episodes}")
        logger.info(f"  Cameras: {dataset.camera_keys}")
        logger.info(f"  Features: {list(dataset.features.keys())[:5]}...")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

    logger.info("\nTo use this dataset for training, add to your config:")
    logger.info(f"  dataset_repo_id: 'local'")
    logger.info(f"  dataset_root: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Convert community_dataset_v1 from v2.1 to v3.0 format")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/nas/datasets/community_so100_v3",
        help="Output directory for converted dataset (default: /mnt/nas/datasets/community_so100_v3)",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for authentication")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push converted dataset to HuggingFace Hub"
    )

    args = parser.parse_args()

    convert_community_dataset(
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
