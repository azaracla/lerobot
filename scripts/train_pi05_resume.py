#!/usr/bin/env python
"""
Checkpoint management and training resumption utilities for Pi0.5 training

This module provides utilities to:
- Find the latest checkpoint
- Resume training from a specific checkpoint
- Manage and organize checkpoints
- Create training snapshots for analysis
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class CheckpointManager:
    """Manage training checkpoints and resumption."""

    def __init__(self, output_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Base output directory for training
        """
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.backups_dir = self.output_dir / "backups"

    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint directory.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoints_dir.exists():
            logger.warning(f"No checkpoints directory found at {self.checkpoints_dir}")
            return None

        # Look for step_* directories
        checkpoints = sorted(
            [d for d in self.checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0,
            reverse=True
        )

        if checkpoints:
            logger.info(f"Found {len(checkpoints)} checkpoints")
            latest = checkpoints[0]
            logger.info(f"Latest checkpoint: {latest.name}")
            return latest

        return None

    def find_checkpoint_by_step(self, step: int) -> Optional[Path]:
        """
        Find a specific checkpoint by step number.

        Args:
            step: Training step number

        Returns:
            Path to checkpoint or None
        """
        checkpoint_path = self.checkpoints_dir / f"step_{step:06d}"
        if checkpoint_path.exists():
            logger.info(f"Found checkpoint at step {step}")
            return checkpoint_path
        else:
            logger.warning(f"No checkpoint found at step {step}")
            return None

    def list_all_checkpoints(self) -> list[tuple[int, Path]]:
        """
        List all checkpoints with their step numbers.

        Returns:
            List of (step, path) tuples sorted by step number
        """
        if not self.checkpoints_dir.exists():
            return []

        checkpoints = []
        for d in self.checkpoints_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step = int(d.name.split("_")[1])
                    checkpoints.append((step, d))
                except (ValueError, IndexError):
                    pass

        return sorted(checkpoints, key=lambda x: x[0])

    def get_checkpoint_info(self, checkpoint_path: Path) -> dict:
        """
        Get information about a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with checkpoint metadata
        """
        info = {
            "path": str(checkpoint_path),
            "step": None,
            "size_mb": 0,
            "created": None,
            "training_state_exists": False,
        }

        # Extract step number
        if checkpoint_path.name.startswith("step_"):
            try:
                info["step"] = int(checkpoint_path.name.split("_")[1])
            except (ValueError, IndexError):
                pass

        # Calculate size
        total_size = 0
        for item in checkpoint_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        info["size_mb"] = total_size / (1024 * 1024)

        # Check modification time
        info["created"] = datetime.fromtimestamp(
            checkpoint_path.stat().st_mtime
        ).isoformat()

        # Check for training state
        info["training_state_exists"] = (checkpoint_path / "training_state.json").exists()

        return info

    def create_training_snapshot(self, checkpoint_path: Path, backup_name: Optional[str] = None) -> Path:
        """
        Create a backup snapshot of a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint to backup
            backup_name: Name for the backup (default: auto-generated)

        Returns:
            Path to backup directory
        """
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        if backup_name is None:
            step = int(checkpoint_path.name.split("_")[1]) if "_" in checkpoint_path.name else 0
            backup_name = f"backup_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.backups_dir / backup_name
        logger.info(f"Creating backup snapshot: {backup_name}")

        shutil.copytree(checkpoint_path, backup_path, dirs_exist_ok=True)
        logger.info(f"Backup created at {backup_path}")

        return backup_path

    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_all_checkpoints()
        if len(checkpoints) <= keep_last_n:
            logger.info(f"Only {len(checkpoints)} checkpoints exist, keeping all")
            return

        to_remove = checkpoints[:-keep_last_n]
        logger.info(f"Removing {len(to_remove)} old checkpoints, keeping last {keep_last_n}")

        for step, path in to_remove:
            logger.info(f"  Removing checkpoint at step {step}")
            shutil.rmtree(path)


def create_resume_command(checkpoint_path: Path) -> str:
    """
    Generate a command to resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Bash command string to resume training
    """
    return f"""
# Resume training from checkpoint
export RESUME_FROM_CHECKPOINT=true
export CHECKPOINT_PATH="{checkpoint_path}"
bash train_pi05_speedrun.sh
"""


def analyze_training_progress(output_dir: Path) -> dict:
    """
    Analyze training progress from logs and checkpoints.

    Args:
        output_dir: Training output directory

    Returns:
        Dictionary with training statistics
    """
    stats = {
        "total_checkpoints": 0,
        "latest_step": 0,
        "latest_checkpoint": None,
        "total_size_mb": 0,
        "checkpoint_sizes": {},
    }

    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return stats

    manager = CheckpointManager(output_dir)
    checkpoints = manager.list_all_checkpoints()

    stats["total_checkpoints"] = len(checkpoints)

    if checkpoints:
        latest_step, latest_path = checkpoints[-1]
        stats["latest_step"] = latest_step
        stats["latest_checkpoint"] = str(latest_path)

    for step, path in checkpoints:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
        stats["checkpoint_sizes"][f"step_{step:06d}"] = f"{size:.2f}MB"
        stats["total_size_mb"] += size

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint management for Pi0.5 training")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List checkpoints command
    list_parser = subparsers.add_parser("list", help="List all checkpoints")
    list_parser.add_argument("output_dir", help="Training output directory")

    # Find latest command
    latest_parser = subparsers.add_parser("latest", help="Find latest checkpoint")
    latest_parser.add_argument("output_dir", help="Training output directory")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get checkpoint information")
    info_parser.add_argument("checkpoint_path", help="Path to checkpoint")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create checkpoint backup")
    backup_parser.add_argument("checkpoint_path", help="Path to checkpoint")
    backup_parser.add_argument("--name", help="Backup name")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old checkpoints")
    cleanup_parser.add_argument("output_dir", help="Training output directory")
    cleanup_parser.add_argument("--keep", type=int, default=3, help="Number of checkpoints to keep")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Generate resume command")
    resume_parser.add_argument("output_dir", help="Training output directory")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training progress")
    analyze_parser.add_argument("output_dir", help="Training output directory")

    args = parser.parse_args()

    if args.command == "list":
        manager = CheckpointManager(args.output_dir)
        checkpoints = manager.list_all_checkpoints()
        if checkpoints:
            print("Available checkpoints:")
            for step, path in checkpoints:
                info = manager.get_checkpoint_info(path)
                print(f"  Step {step:06d}: {info['size_mb']:.1f}MB ({info['created']})")
        else:
            print("No checkpoints found")

    elif args.command == "latest":
        manager = CheckpointManager(args.output_dir)
        latest = manager.find_latest_checkpoint()
        if latest:
            info = manager.get_checkpoint_info(latest)
            print(f"Latest checkpoint: {latest.name}")
            print(f"  Step: {info['step']}")
            print(f"  Size: {info['size_mb']:.1f}MB")
            print(f"  Created: {info['created']}")
        else:
            print("No checkpoints found")

    elif args.command == "info":
        manager = CheckpointManager(Path(args.checkpoint_path).parent.parent)
        info = manager.get_checkpoint_info(Path(args.checkpoint_path))
        print(f"Checkpoint: {info['path']}")
        print(f"  Step: {info['step']}")
        print(f"  Size: {info['size_mb']:.1f}MB")
        print(f"  Created: {info['created']}")
        print(f"  Training state: {info['training_state_exists']}")

    elif args.command == "backup":
        manager = CheckpointManager(Path(args.checkpoint_path).parent.parent)
        backup = manager.create_training_snapshot(Path(args.checkpoint_path), args.name)
        print(f"Backup created: {backup}")

    elif args.command == "cleanup":
        manager = CheckpointManager(args.output_dir)
        manager.cleanup_old_checkpoints(args.keep)
        print(f"Cleanup complete, kept last {args.keep} checkpoints")

    elif args.command == "resume":
        manager = CheckpointManager(args.output_dir)
        latest = manager.find_latest_checkpoint()
        if latest:
            print(create_resume_command(latest))
        else:
            print("No checkpoints found to resume from")

    elif args.command == "analyze":
        stats = analyze_training_progress(Path(args.output_dir))
        print(f"Training Progress Analysis:")
        print(f"  Total checkpoints: {stats['total_checkpoints']}")
        print(f"  Latest step: {stats['latest_step']}")
        print(f"  Total size: {stats['total_size_mb']:.1f}MB")
        if stats['latest_checkpoint']:
            print(f"  Latest: {stats['latest_checkpoint']}")

    else:
        parser.print_help()
