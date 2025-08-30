#!/usr/bin/env python

"""
Quick benchmark to test video decoding speed across different backends.
"""

import time
from pathlib import Path
import torch

def test_video_backend(video_path, backend_name, num_frames=10):
    """Test video decoding speed for a specific backend."""
    try:
        from lerobot.datasets.video_utils import decode_video_frames
        
        # Create timestamps for first N frames
        fps = 30  # Assume 30fps, adjust if needed
        timestamps = [i / fps for i in range(num_frames)]
        
        # Time the decoding
        start_time = time.perf_counter()
        frames = decode_video_frames(video_path, timestamps, tolerance_s=1e-4, backend=backend_name)
        decode_time = time.perf_counter() - start_time
        
        frames_decoded = frames.shape[1] if frames.dim() > 1 else frames.shape[0]
        ms_per_frame = (decode_time * 1000) / max(frames_decoded, 1)
        
        print(f"✅ {backend_name:12} | {decode_time*1000:6.1f}ms total | {ms_per_frame:6.1f}ms/frame | {frames_decoded} frames")
        return decode_time, frames_decoded
        
    except Exception as e:
        print(f"❌ {backend_name:12} | ERROR: {str(e)[:50]}...")
        return float('inf'), 0

def main():
    print("📦 Downloading dataset to get video file locations...")
    
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Download the dataset - this will tell us exactly where it's stored
        dataset = LeRobotDataset("kenmacken/record-test-2", download_videos=True)
        
        print(f"✅ Dataset downloaded to: {dataset.root}")
        print(f"   Video keys: {dataset.meta.video_keys}")
        print(f"   Total episodes: {dataset.meta.total_episodes}")
        
        # Get actual video file paths from the dataset
        video_files = []
        for ep_idx in range(min(2, dataset.meta.total_episodes)):  # Test first 2 episodes max
            for vid_key in dataset.meta.video_keys:
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
                if video_path.exists():
                    video_files.append(video_path)
                    break  # Just need one video file for testing
            if video_files:
                break
        
        if not video_files:
            print("❌ No video files found after download!")
            return
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        # Fallback to manual search
        possible_paths = [
            Path.home() / ".cache/huggingface/lerobot/kenmacken/record-test-2",
            Path("/tmp/huggingface/lerobot/kenmacken/record-test-2"),
            Path("./datasets/record-test-2"),
        ]
        
        video_files = []
        print("Trying fallback search...")
        for path in possible_paths:
            print(f"  Checking: {path}")
            if path.exists():
                files = list(path.rglob("*.mp4"))
                if files:
                    video_files = files
                    print(f"  ✅ Found {len(files)} video files!")
                    break
        
        if not video_files:
            print("❌ No video files found!")
            return
        
    test_video = video_files[0]
    print(f"Testing video: {test_video.name}")
    print(f"File size: {test_video.stat().st_size / 1024 / 1024:.1f} MB")
    print("-" * 60)
    
    backends = ["torchcodec", "pyav", "video_reader"]
    results = {}
    
    for backend in backends:
        decode_time, frames = test_video_backend(test_video, backend)
        results[backend] = (decode_time, frames)
    
    print("-" * 60)
    print("RECOMMENDATION:")
    
    # Find fastest backend
    valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1][0])
        print(f"🚀 Use '{fastest[0]}' - fastest backend!")
        print(f"   Add to your config: video_backend: \"{fastest[0]}\"")
        
        slowest_time = max(valid_results.values())[0]
        speedup = slowest_time / fastest[1][0]
        print(f"   Speedup vs slowest: {speedup:.1f}x faster")
    else:
        print("❌ No backends worked!")

if __name__ == "__main__":
    main()
