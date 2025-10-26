#!/usr/bin/env python3
"""
Downscale video resolution in a LeRobot dataset to reduce memory usage during training.

Usage:
    python scripts/downscale_dataset_videos.py \
        --input_repo_id lerobot/aloha_sim_insertion_human \
        --output_repo_id lerobot/aloha_sim_insertion_human_480p \
        --target_height 480

    # Or with more options:
    python scripts/downscale_dataset_videos.py \
        --input_repo_id lerobot/aloha_sim_insertion_human \
        --output_repo_id lerobot/aloha_sim_insertion_human_360p \
        --target_height 360 \
        --push_to_hub
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import av
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_new_video_dimensions(frame: np.ndarray, target_height: int) -> tuple[int, int]:
    """
    Calculate new video dimensions maintaining aspect ratio.

    Args:
        frame: A sample video frame.
        target_height: The target height for the new video.

    Returns:
        A tuple (new_width, new_height) with even dimensions.
    """
    h, w = frame.shape[:2]
    new_h = target_height
    new_w = int(w * (new_h / h))

    # Ensure dimensions are even (required by some codecs, e.g., H.264)
    new_h = new_h + 1 if new_h % 2 != 0 else new_h
    new_w = new_w + 1 if new_w % 2 != 0 else new_w

    return new_w, new_h


def resize_frame(frame: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Resize a frame to the target dimensions.
    This function is a placeholder for a more optimized resizing implementation.
    """
    # This is a basic resizing implementation. For better performance and quality,
    # consider using a library like OpenCV (cv2.resize) with appropriate interpolation.
    # For this script, we'll rely on PyAV's built-in scaling, which is efficient.
    # This function is kept for clarity on the resizing step.
    pass  # PyAV handles resizing internally


def process_video_file(input_path: Path, output_path: Path, target_height: int) -> tuple[int, int]:
    """
    Process a single video file by resizing all its frames.

    Args:
        input_path: Path to the source video file.
        output_path: Path to save the downscaled video file.
        target_height: The target height for the new video.

    Returns:
        A tuple (new_width, new_height) of the resized video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(input_path)) as input_container:
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = "AUTO"

        # Decode a single frame to calculate new dimensions
        first_frame = next(input_container.decode(video=0))
        new_width, new_height = get_new_video_dimensions(first_frame.to_ndarray(format="rgb24"), target_height)

        # Reset stream to the beginning
        input_container.seek(0)

        with av.open(str(output_path), "w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=input_stream.average_rate)
            output_stream.thread_type = "AUTO"
            output_stream.width = new_width
            output_stream.height = new_height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": "23", "preset": "medium"}

            for frame in input_container.decode(video=0):
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)

            # Flush remaining packets
            for packet in output_stream.encode():
                output_container.mux(packet)

    return new_width, new_height


def copy_and_update_metadata(src_dataset: LeRobotDataset, local_dir: Path, new_video_shapes: dict):
    """
    Copy metadata files and update them with new video dimensions.
    """
    logger.info("Copying and updating metadata files...")
    meta_dir = local_dir / "meta"
    meta_dir.mkdir(exist_ok=True, parents=True)

    src_meta_dir = Path(src_dataset.root) / "meta"
    if src_meta_dir.exists():
        shutil.copytree(src_meta_dir, meta_dir, dirs_exist_ok=True)

    # Update info.json with new video dimensions
    info_file = meta_dir / "info.json"
    if info_file.exists():
        with open(info_file) as f:
            info_data = json.load(f)

        if "features" in info_data:
            for key, (h, w, c) in new_video_shapes.items():
                if key in info_data["features"]:
                    info_data["features"][key]["shape"] = [h, w, c]
                    if "info" in info_data["features"][key]:
                        info_data["features"][key]["info"]["video.height"] = h
                        info_data["features"][key]["info"]["video.width"] = w

        with open(info_file, "w") as f:
            json.dump(info_data, f, indent=2)


def downscale_dataset(
    input_repo_id: str,
    output_repo_id: str,
    target_height: int = 480,
    push_to_hub: bool = False,
    local_dir: Path | None = None,
):
    """
    Create a new dataset with downscaled videos.
    """
    logger.info(f"Loading dataset: {input_repo_id}")
    src_dataset = LeRobotDataset(input_repo_id)

    if local_dir is None:
        local_dir = HF_LEROBOT_HOME / output_repo_id
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating new dataset at: {local_dir}")

    # Copy non-video data (parquet files)
    logger.info("Copying dataset metadata and parquet files...")
    src_data_dir = Path(src_dataset.root) / "data"
    if src_data_dir.exists():
        shutil.copytree(src_data_dir, local_dir / "data", dirs_exist_ok=True)

    # Process videos
    logger.info(f"Processing {len(src_dataset.meta.video_keys)} video streams...")
    videos_dir = local_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    new_video_shapes = {}

    for video_key in src_dataset.meta.video_keys:
        logger.info(f"\nProcessing video key: {video_key}")

        src_video_dir = Path(src_dataset.root) / "videos" / video_key
        dest_video_dir = videos_dir / video_key

        if not src_video_dir.exists():
            logger.warning(f"Video directory not found: {src_video_dir}")
            continue

        video_files = sorted(list(src_video_dir.rglob("*.mp4")))
        logger.info(f"Found {len(video_files)} video files")

        new_w, new_h = None, None
        for video_file in tqdm(video_files, desc=f"Videos for {video_key}"):
            relative_path = video_file.relative_to(src_video_dir)
            output_file = dest_video_dir / relative_path
            new_w, new_h = process_video_file(video_file, output_file, target_height)

        if new_w and new_h:
            new_video_shapes[video_key] = (new_h, new_w, 3)

    copy_and_update_metadata(src_dataset, local_dir, new_video_shapes)

    # Copy other metadata files
    for metadata_file in ["stats.safetensors"]:
        src_file = Path(src_dataset.root) / metadata_file
        if src_file.exists():
            shutil.copy2(src_file, local_dir / metadata_file)

    logger.info("\n✓ Dataset downscaled successfully!")
    logger.info(f"  Source: {input_repo_id}")
    logger.info(f"  Output: {output_repo_id}")
    logger.info(f"  Location: {local_dir}")

    if push_to_hub:
        logger.info(f"\nPushing dataset to HuggingFace Hub: {output_repo_id}")
        new_dataset = LeRobotDataset(output_repo_id, root=local_dir)
        new_dataset.push_to_hub(output_repo_id)
        logger.info("✓ Dataset pushed to Hub")


def main():
    parser = argparse.ArgumentParser(
        description="Downscale video resolution in a LeRobot dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_repo_id",
        type=str,
        required=True,
        help="Input dataset repository ID (e.g., 'lerobot/aloha_sim_insertion_human')",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="Output dataset repository ID (e.g., 'lerobot/aloha_sim_insertion_human_480p')",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=480,
        help="Target video height in pixels (default: 480). Width is scaled proportionally.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the downscaled dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--local_dir",
        type=Path,
        help="Local directory to save the dataset (optional).",
    )

    args = parser.parse_args()

    try:
        downscale_dataset(
            input_repo_id=args.input_repo_id,
            output_repo_id=args.output_repo_id,
            target_height=args.target_height,
            push_to_hub=args.push_to_hub,
            local_dir=args.local_dir,
        )
        logger.info("\n✓ Done!")
    except Exception as e:
        logger.error(f"\n✗ An error occurred: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()