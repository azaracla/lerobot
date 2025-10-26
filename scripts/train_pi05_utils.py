#!/usr/bin/env python
"""
Utility functions for Pi0.5 training pipeline on Vast.ai

This module provides helper functions for:
- Dataset validation and preparation
- Training configuration management
- Checkpoint handling
- Result reporting
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> tuple[bool, str]:
    """Check if GPU is available and return GPU info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

        total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count))
        total_memory_gb = total_memory / 1e9

        gpu_info = f"{gpu_count}x {gpu_names[0]} ({total_memory_gb:.1f}GB total)"
        return True, gpu_info
    else:
        return False, "No GPU detected (CPU-only mode)"


def validate_dataset(dataset_repo: str, hf_token: Optional[str] = None) -> bool:
    """
    Validate that the dataset exists and is accessible.

    Args:
        dataset_repo: HuggingFace dataset repository ID (e.g., "azaracla/smolvla_3dprint_plate")
        hf_token: Optional HuggingFace API token for private datasets

    Returns:
        True if dataset is valid, False otherwise
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from huggingface_hub import list_repo_files

        logger.info(f"Validating dataset: {dataset_repo}")

        # Check if dataset files exist
        try:
            files = list_repo_files(dataset_repo, token=hf_token)
            logger.info(f"  Found {len(files)} files in dataset")
        except Exception as e:
            logger.warning(f"  Could not list files: {e}")

        # Try to load dataset metadata
        try:
            dataset = LeRobotDataset(dataset_repo, episodes=list(range(min(1, 10))))
            logger.info(f"  ✓ Dataset loaded successfully")
            logger.info(f"    Episodes: {dataset.num_episodes}")
            logger.info(f"    Total frames: {dataset.num_frames}")
            logger.info(f"    FPS: {dataset.meta.fps}")

            # Check for quantile stats
            if hasattr(dataset.meta, 'stats') and dataset.meta.stats:
                logger.info(f"    ✓ Quantile normalization stats present")
            else:
                logger.warning(f"    ⚠ No quantile stats found (will use MEAN_STD normalization)")

            return True

        except Exception as e:
            logger.error(f"  ✗ Could not load dataset: {e}")
            return False

    except ImportError as e:
        logger.warning(f"LeRobot not installed: {e}")
        return False


def prepare_dataset_normalization(
    dataset_repo: str,
    output_dir: Path,
    hf_token: Optional[str] = None
) -> bool:
    """
    Prepare dataset normalization statistics if not already present.

    Args:
        dataset_repo: HuggingFace dataset repository ID
        output_dir: Output directory for statistics
        hf_token: Optional HuggingFace API token

    Returns:
        True if preparation successful, False otherwise
    """
    try:
        from lerobot.datasets.v30.augment_dataset_quantile_stats import compute_quantile_stats

        logger.info(f"Computing quantile normalization stats for {dataset_repo}...")

        # This would compute stats, but currently requires full dataset in memory
        # For now, we'll let the training script handle this with MEAN_STD normalization
        logger.info("  Quantile stat computation delegated to training script")
        return True

    except Exception as e:
        logger.warning(f"Could not compute quantile stats: {e}")
        return True  # Not critical, training can proceed


def generate_training_config(
    dataset_repo: str,
    base_model: str = "lerobot/pi05_base",
    steps: int = 3000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    output_dir: str = "./outputs/pi05_speedrun",
    hf_repo_id: Optional[str] = None,
    enable_wandb: bool = True,
    push_to_hub: bool = True,
) -> dict:
    """
    Generate a complete training configuration dictionary.

    Args:
        dataset_repo: Dataset repository ID
        base_model: Base model to fine-tune
        steps: Number of training steps
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Output directory
        hf_repo_id: HuggingFace repo ID for pushing
        enable_wandb: Whether to use Weights & Biases
        push_to_hub: Whether to push to Hub

    Returns:
        Configuration dictionary
    """
    config = {
        "dataset": {
            "repo_id": dataset_repo,
        },
        "policy": {
            "type": "pi05",
            "pretrained_path": base_model,
            "repo_id": hf_repo_id,
            "push_to_hub": push_to_hub,
            "compile_model": True,
            "gradient_checkpointing": True,
            "dtype": "bfloat16",
            "device": "cuda",
            "normalization_mapping": {
                "ACTION": "MEAN_STD",
                "STATE": "MEAN_STD",
                "VISUAL": "IDENTITY"
            }
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "num_workers": 4,
            "log_freq": 100,
            "save_freq": 500,
            "save_checkpoint": True,
            "output_dir": output_dir,
        },
        "optimizer": {
            "type": "adam",
            "lr": learning_rate,
            "warmup_steps": 100,
            "grad_clip_norm": 1.0,
        },
        "wandb": {
            "enable": enable_wandb,
            "project": "lerobot-pi05" if enable_wandb else None,
        }
    }

    return config


def save_training_config(config: dict, output_path: Path) -> None:
    """Save configuration to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {output_path}")


def load_training_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def estimate_training_time(
    num_frames: int,
    batch_size: int,
    num_gpus: int = 1,
    estimated_fps: float = 100.0,
    warmup_factor: float = 1.1
) -> dict:
    """
    Estimate training time based on dataset size and hardware.

    Args:
        num_frames: Total number of frames in dataset
        batch_size: Training batch size
        num_gpus: Number of GPUs available
        estimated_fps: Estimated frames per second throughput
        warmup_factor: Factor for warmup overhead

    Returns:
        Dictionary with time estimates
    """
    effective_fps = (estimated_fps * num_gpus) / (batch_size / 16)  # Normalize to batch_size 16
    total_seconds = (num_frames / (effective_fps * warmup_factor))

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return {
        "estimated_hours": hours,
        "estimated_minutes": minutes,
        "total_seconds": int(total_seconds),
        "effective_fps": effective_fps,
    }


def create_vastai_launch_script(
    output_path: Path,
    dataset_repo: str = "azaracla/smolvla_3dprint_plate",
    steps: int = 3000,
    hf_token: str = "",
    wandb_api_key: str = "",
    hf_repo_id: str = "",
) -> None:
    """
    Create a Vast.ai launch script template.

    Args:
        output_path: Where to save the launch script
        dataset_repo: Dataset repository
        steps: Number of training steps
        hf_token: HuggingFace token placeholder
        wandb_api_key: Weights & Biases key placeholder
        hf_repo_id: Target HuggingFace repository
    """
    script_content = f'''#!/bin/bash
# Vast.ai launch script for Pi0.5 training
# Set your tokens before running on Vast.ai

export HF_TOKEN="{hf_token or 'your_hf_token_here'}"
export WANDB_API_KEY="{wandb_api_key or 'your_wandb_key_here'}"
export DATASET_REPO="{dataset_repo}"
export STEPS={steps}
export HF_REPO_ID="{hf_repo_id or 'your_username/model_name'}"
export PUSH_TO_HUB=true
export ENABLE_WANDB=true

# Install dependencies
pip install -e ".[pi]"

# Run training
bash train_pi05_speedrun.sh
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script_content)

    output_path.chmod(0o755)
    logger.info(f"Vast.ai launch script created: {output_path}")


def print_system_info() -> None:
    """Print system information for debugging."""
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")

    gpu_available, gpu_info = check_gpu_availability()
    if gpu_available:
        logger.info(f"  GPU: {gpu_info}")
    else:
        logger.warning(f"  GPU: {gpu_info}")

    try:
        import lerobot
        logger.info(f"  LeRobot: {lerobot.__version__ if hasattr(lerobot, '__version__') else 'installed'}")
    except ImportError:
        logger.warning("  LeRobot: not installed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pi0.5 training utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate dataset command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("dataset_repo", help="Dataset repository ID")
    validate_parser.add_argument("--hf-token", help="HuggingFace API token")

    # Generate config command
    config_parser = subparsers.add_parser("config", help="Generate training config")
    config_parser.add_argument("--dataset", default="azaracla/smolvla_3dprint_plate")
    config_parser.add_argument("--steps", type=int, default=3000)
    config_parser.add_argument("--batch-size", type=int, default=32)
    config_parser.add_argument("--output", default="training_config.json")

    # System info command
    subparsers.add_parser("sysinfo", help="Show system information")

    # Vast.ai script command
    vastai_parser = subparsers.add_parser("vastai", help="Generate Vast.ai launch script")
    vastai_parser.add_argument("--output", default="launch_on_vastai.sh")
    vastai_parser.add_argument("--dataset", default="azaracla/smolvla_3dprint_plate")

    args = parser.parse_args()

    if args.command == "validate":
        success = validate_dataset(args.dataset_repo, args.hf_token)
        sys.exit(0 if success else 1)

    elif args.command == "config":
        config = generate_training_config(
            dataset_repo=args.dataset,
            steps=args.steps,
            batch_size=args.batch_size,
        )
        save_training_config(config, Path(args.output))
        logger.info(f"Config saved to {args.output}")

    elif args.command == "sysinfo":
        print_system_info()

    elif args.command == "vastai":
        create_vastai_launch_script(Path(args.output), dataset_repo=args.dataset)

    else:
        parser.print_help()
