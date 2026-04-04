#!/usr/bin/env python
"""
Aggregate converted community dataset sub-datasets into one dataset.

This script:
1. Finds all converted v3.0 sub-datasets in the community_dataset_v3 directory
2. Aggregates them into a single dataset
3. Saves to the specified output directory

Usage:
    python scripts/aggregate_community_datasets.py \
        --input_dir /mnt/nas/datasets/community_dataset_v3 \
        --output_dir /mnt/nas/datasets/community_aggregated

Requirements:
    - All sub-datasets must be converted to v3.0 format first
    - All sub-datasets must have compatible features (same robot type, fps, etc.)
"""

import argparse
import logging
import os
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_converted_subdatasets(input_dir: Path) -> list[tuple[str, str]]:
    """Find all converted v3.0 sub-datasets in the input directory.

    Returns:
        List of (repo_id, root_path) tuples
    """
    subdatasets = []

    for contributor in sorted(os.listdir(input_dir)):
        contributor_path = input_dir / contributor
        if not contributor_path.is_dir() or contributor.startswith("."):
            continue

        for dataset_name in sorted(os.listdir(contributor_path)):
            dataset_path = contributor_path / dataset_name
            if not dataset_path.is_dir():
                continue

            # Check if it's a v3.0 dataset
            info_path = dataset_path / "meta" / "info.json"
            if info_path.exists():
                import json

                with open(info_path) as f:
                    info = json.load(f)

                version = info.get("codebase_version", "unknown")
                if version == "v3.0":
                    repo_id = f"{contributor}/{dataset_name}"
                    subdatasets.append((repo_id, str(input_dir)))
                    logger.info(f"  Found: {repo_id} (v{version})")
                else:
                    logger.warning(f"  Skipping {contributor}/{dataset_name}: v{version} (not v3.0)")
            else:
                logger.warning(f"  Skipping {contributor}/{dataset_name}: no info.json")

    return subdatasets


def main():
    parser = argparse.ArgumentParser(description="Aggregate community dataset sub-datasets into one")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/nas/datasets/community_dataset_v3",
        help="Directory containing converted sub-datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/nas/datasets/community_aggregated",
        help="Output directory for aggregated dataset",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    logger.info(f"Scanning for sub-datasets in: {input_path}")

    subdatasets = find_converted_subdatasets(input_path)

    if not subdatasets:
        logger.error("No v3.0 sub-datasets found!")
        logger.error(f"Make sure to convert the datasets first using convert_dataset_v21_to_v30.py")
        return

    logger.info(f"\nFound {len(subdatasets)} sub-datasets to aggregate")

    repo_ids = [s[0] for s in subdatasets]
    roots = [s[1] for s in subdatasets]

    logger.info(f"Aggregating to: {output_path}")

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id="community_aggregated",
        roots=roots,
        aggr_root=output_path,
    )

    logger.info("\nAggregation complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"\nTo use for training, set in config:")
    logger.info(f"  dataset_repo_id: 'local'")
    logger.info(f"  dataset_root: '{output_path}'")


if __name__ == "__main__":
    main()
