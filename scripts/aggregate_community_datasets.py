#!/usr/bin/env python
"""
Aggregate converted community dataset sub-datasets into one dataset.

This script:
1. Finds all converted v3.0 sub-datasets in the input directory
2. Selects the most common feature configuration
3. Filters to compatible datasets
4. Aggregates them into a single dataset
5. Saves to the specified output directory

Usage:
    python scripts/aggregate_community_datasets.py \
        --input_dir /mnt/nas/datasets/community_dataset_v1 \
        --output_dir /mnt/nas/datasets/community_aggregated

Requirements:
    - All sub-datasets must be converted to v3.0 format first
    - All sub-datasets must have compatible features (same robot type, fps, etc.)
"""

import argparse
import logging
import os
from collections import Counter
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_converted_subdatasets(input_dir: Path):
    """Find all converted v3.0 sub-datasets in the input directory.

    Returns:
        List of dicts with repo_id, root_path, and features info
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

            info_path = dataset_path / "meta" / "info.json"
            if info_path.exists():
                import json

                with open(info_path) as f:
                    info = json.load(f)

                    version = info.get("codebase_version", "unknown")
                if version == "v3.0":
                    features = info.get("features", {})
                    image_keys = tuple(sorted(k for k in features if features[k]["dtype"] == "video"))
                    action_shape = tuple(features.get("action", {}).get("shape", None))
                    action_names = tuple(features.get("action", {}).get("names", []))
                    fps = info.get("fps")
                    robot_type = info.get("robot_type")
                    codec = None
                    resolutions = None
                    if image_keys:
                        codec = features[image_keys[0]].get("info", {}).get("video.codec", None)
                        resolutions = tuple(
                            (
                                features[k].get("info", {}).get("video.height", None),
                                features[k].get("info", {}).get("video.width", None),
                            )
                            for k in image_keys
                        )
                    feature_keys = frozenset(features.keys())

                    subdatasets.append(
                        {
                            "repo_id": f"{contributor}/{dataset_name}",
                            "root": str(dataset_path),
                            "image_keys": image_keys,
                            "action_shape": action_shape,
                            "action_names": action_names,
                            "fps": fps,
                            "robot_type": robot_type,
                            "codec": codec,
                            "resolutions": resolutions,
                            "feature_keys": feature_keys,
                        }
                    )
                    logger.info(f"  Found: {contributor}/{dataset_name} (v{version})")
                else:
                    logger.warning(f"  Skipping {contributor}/{dataset_name}: v{version} (not v3.0)")
            else:
                logger.warning(f"  Skipping {contributor}/{dataset_name}: no info.json")

    return subdatasets


def select_compatible_datasets(subdatasets):
    """Select datasets compatible with the most common feature configuration.

    Returns:
        Tuple of (selected_datasets, excluded_datasets)
    """
    config_counter = Counter()
    for ds in subdatasets:
        config = (
            ds["image_keys"],
            ds["action_shape"],
            ds["action_names"],
            ds["fps"],
            ds["robot_type"],
            ds["codec"],
            ds["resolutions"],
            ds["feature_keys"],
        )
        config_counter[config] += 1

    most_common_config = config_counter.most_common(1)[0][0]
    (
        target_image_keys,
        target_action_shape,
        target_action_names,
        target_fps,
        target_robot_type,
        target_codec,
        target_resolutions,
        target_feature_keys,
    ) = most_common_config

    logger.info(f"\nMost common config:")
    logger.info(f"  image_keys: {target_image_keys}")
    logger.info(f"  action_shape: {target_action_shape}")
    logger.info(f"  action_names: {target_action_names}")
    logger.info(f"  fps: {target_fps}")
    logger.info(f"  robot_type: {target_robot_type}")
    logger.info(f"  codec: {target_codec}")
    logger.info(f"  resolutions: {target_resolutions}")
    logger.info(f"  feature_keys: {target_feature_keys}")

    selected = []
    excluded = []

    for ds in subdatasets:
        config = (
            ds["image_keys"],
            ds["action_shape"],
            ds["action_names"],
            ds["fps"],
            ds["robot_type"],
            ds["codec"],
            ds["resolutions"],
            ds["feature_keys"],
        )
        if config == most_common_config:
            selected.append(ds)
        else:
            excluded.append(ds)

    logger.info(f"\nCompatible datasets: {len(selected)}")
    logger.info(f"Excluded datasets: {len(excluded)}")

    if excluded:
        logger.info("\nExcluded datasets:")
        for ds in excluded:
            logger.info(
                f"  - {ds['repo_id']}: image_keys={ds['image_keys']}, codec={ds['codec']}, resolutions={ds['resolutions']}"
            )

    return selected, excluded


def main():
    parser = argparse.ArgumentParser(description="Aggregate community dataset sub-datasets into one")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/nas/datasets/community_dataset_v1",
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

    logger.info(f"\nFound {len(subdatasets)} sub-datasets total")

    selected, excluded = select_compatible_datasets(subdatasets)

    if not selected:
        logger.error("No compatible datasets found!")
        return

    repo_ids = [s["repo_id"] for s in selected]
    roots = [s["root"] for s in selected]

    logger.info(f"\nAggregating {len(selected)} compatible datasets to: {output_path}")

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
