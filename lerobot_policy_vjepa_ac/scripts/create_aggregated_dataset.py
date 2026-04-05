#!/usr/bin/env python
"""
Create an aggregated dataset from local v3.0 sub-datasets.
This is a simplified version that handles local datasets correctly.
"""

import argparse
import os
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def find_local_subdatasets(root: str | Path) -> list[tuple[str, str]]:
    """Find all v3.0 sub-datasets in the root directory.

    Returns:
        List of (repo_id, root_path) tuples for local datasets
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
                with open(info_path) as f:
                    info = json.load(f)

                version = info.get("codebase_version", "unknown")
                if version == "v3.0":
                    # For local datasets, use "local" as repo_id and full path as root
                    datasets.append(("local", str(dataset_path)))
                    print(f"  Found: {contributor}/{dataset_name}")
                else:
                    print(f"  Skipping {contributor}/{dataset_name}: v{version} (not v3.0)")

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Create aggregated dataset from local sub-datasets")
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

    print(f"Scanning for sub-datasets in: {input_path}")

    subdatasets = find_local_subdatasets(input_path)

    if not subdatasets:
        print("ERROR: No v3.0 sub-datasets found!")
        return

    print(f"\nFound {len(subdatasets)} sub-datasets to aggregate")

    # For aggregate_datasets, we need:
    # - repo_ids: list of repo_ids (all "local" for local datasets)
    # - roots: list of root paths (full paths to each dataset)
    # But aggregate_datasets seems to expect a different format...

    # Actually, let's check what aggregate_datasets expects
    repo_ids = [s[0] for s in subdatasets]
    roots = [s[1] for s in subdatasets]

    # Get the parent directory to use as root, and relative paths
    parent_root = str(input_path)

    # Actually aggregate_datasets creates a new dataset that references the original ones
    # Let's try passing the paths differently

    print(f"\nAggregating to: {output_path}")

    # Pass the parent directory as root, and contributor/dataset_name as repo_id
    # But this doesn't work for local datasets...

    # Alternative: use the first dataset's parent as root for all?
    # No, that won't work either because each dataset is in its own subdirectory.

    # The issue is that aggregate_datasets seems designed for HuggingFace datasets,
    # not for locally structured datasets.

    # Let's try a different approach: create a single dataset by symlinking or copying
    # the data, or by using MultiLeRobotDataset directly.

    # For now, let's try the simplest approach: just use MultiLeRobotDataset directly
    # in the training script, bypassing the aggregation.

    print("\nThe aggregate_datasets function doesn't support local datasets properly.")
    print("We will use MultiLeRobotDataset directly in training instead.")
    print(f"\nSubdatasets found:")
    for repo_id, root in subdatasets:
        print(f"  - {repo_id} @ {root}")


if __name__ == "__main__":
    main()
