#!/usr/bin/env python
"""
List all sub-datasets in the community_dataset_v1 directory.
Useful for debugging or generating the dataset list for multi-dataset training.

Usage:
    python scripts/list_subdatasets.py
    python scripts/list_subdatasets.py --root /mnt/nas/datasets/community_dataset_v1 --format json
"""

import argparse
import json
import os
from pathlib import Path


def find_subdatasets(root: str | Path) -> list[str]:
    """Find all v3.0 sub-datasets in the root directory.

    Args:
        root: Root directory containing contributor/dataset_name structure

    Returns:
        List of dataset paths in 'contributor/dataset_name' format
    """
    root = Path(root)
    datasets = []

    for contributor in sorted(os.listdir(root)):
        contributor_path = root / contributor
        if not contributor_path.is_dir() or contributor.startswith("."):
            continue

        for dataset_name in sorted(os.listdir(contributor_path)):
            dataset_path = contributor_path / dataset_name
            if not dataset_path.is_dir():
                continue

            info_path = dataset_path / "meta" / "info.json"
            if info_path.exists():
                datasets.append(f"{contributor}/{dataset_name}")
            else:
                print(f"Warning: {contributor}/{dataset_name} missing meta/info.json, skipping")

    return datasets


def main():
    parser = argparse.ArgumentParser(description="List all v3.0 sub-datasets")
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/nas/datasets/community_dataset_v1",
        help="Root directory of the community dataset",
    )
    parser.add_argument(
        "--format", type=str, choices=["text", "json", "bash"], default="text", help="Output format"
    )
    args = parser.parse_args()

    datasets = find_subdatasets(args.root)

    if args.format == "text":
        for ds in datasets:
            print(ds)
    elif args.format == "json":
        print(json.dumps(datasets, indent=2))
    elif args.format == "bash":
        # Format for bash array: ("a" "b" "c")
        formatted = " ".join(f'"{ds}"' for ds in datasets)
        print(f"({formatted})")

    print(f"\nTotal: {len(datasets)} sub-datasets", file=os.sys.stderr)


if __name__ == "__main__":
    main()
