#!/usr/bin/env python
"""
Test script to verify LeRobot streaming works with DROID dataset.

This tests the StreamingLeRobotDataset class with the DROID dataset
on HuggingFace Hub.

Usage:
    cd /home/arthur/Code/lerobot
    conda activate lerobot
    python plans/test_lerobot_streaming_droid.py
"""

import torch
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Dataset repo ID for LeRobot DROID
REPO_ID = "lerobot/droid_1.0.1"


def test_streaming_basic():
    """Test basic streaming functionality."""
    print(f"Testing StreamingLeRobotDataset with {REPO_ID}")
    print("=" * 60)
    
    # Create streaming dataset
    dataset = StreamingLeRobotDataset(
        repo_id=REPO_ID,
        streaming=True,
        buffer_size=100,
        shuffle=True,
    )
    
    print(f"Dataset loaded successfully!")
    print(f"  - Total frames: {dataset.num_frames}")
    print(f"  - Total episodes: {dataset.num_episodes}")
    print(f"  - FPS: {dataset.fps}")
    print(f"  - Video keys: {dataset.meta.video_keys}")
    print(f"  - Features: {list(dataset.meta.info['features'].keys())}")
    print()
    
    return dataset


def test_iteration(dataset, num_samples=5):
    """Test iterating over the streaming dataset."""
    print(f"Iterating over {num_samples} samples:")
    print("-" * 60)
    
    for i, sample in enumerate(dataset):
        print(f"\nSample {i}:")
        print(f"  Episode index: {sample['episode_index'].item()}")
        print(f"  Frame index: {sample['frame_index'].item()}")
        
        # Print state/action shapes
        if 'observation.state' in sample:
            print(f"  observation.state shape: {sample['observation.state'].shape}")
        if 'action' in sample:
            print(f"  action shape: {sample['action'].shape}")
        
        # Print image keys and shapes
        for key, value in sample.items():
            if 'image' in key and isinstance(value, torch.Tensor):
                print(f"  {key} shape: {value.shape}")
        
        if i >= num_samples - 1:
            break
    
    print("\n" + "=" * 60)
    print("Iteration test completed successfully!")


def test_video_frame_loading(dataset, num_samples=3):
    """Test video frame loading specifically."""
    print(f"\nTesting video frame loading ({num_samples} samples):")
    print("-" * 60)
    
    for i, sample in enumerate(dataset):
        for key, value in sample.items():
            if 'image' in key and isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        if i >= num_samples - 1:
            break
    
    print("\nVideo frame loading test completed!")


def test_camera_extrinsics(dataset):
    """Test camera extrinsics loading."""
    print("\nTesting camera extrinsics:")
    print("-" * 60)
    
    # Check if camera_extrinsics features exist
    features = dataset.meta.info.get('features', {})
    extrinsics_keys = [k for k in features.keys() if 'extrinsics' in k]
    print(f"  Camera extrinsics keys: {extrinsics_keys}")
    
    # Get a sample to check actual data
    sample = next(iter(dataset))
    for key in extrinsics_keys:
        if key in sample:
            print(f"  {key}: shape={sample[key].shape}")


def test_parquet_data():
    """Test loading parquet data directly (without video)."""
    print("\nTesting parquet data loading:")
    print("-" * 60)
    
    import numpy as np
    from datasets import load_dataset
    
    # Load just the data parquet files directly
    ds = load_dataset('lerobot/droid_1.0.1', split='train', streaming=True)
    ds = ds.with_format('pandas')
    
    sample = next(iter(ds))
    
    # Get actual array shapes
    print('=== STATE/ACTION ARRAY SHAPES ===')
    
    obs_state = np.array(sample['observation.state'])
    print(f'observation.state: shape={obs_state.shape}, dtype={obs_state.dtype}')
    
    action = np.array(sample['action'])
    print(f'action: shape={action.shape}, dtype={action.dtype}')
    
    obs_cart = np.array(sample['observation.state.cartesian_position'])
    print(f'observation.state.cartesian_position: shape={obs_cart.shape}')
    
    obs_gripper = sample['observation.state.gripper_position']
    print(f'observation.state.gripper_position: {obs_gripper}')
    
    obs_joint = np.array(sample['observation.state.joint_position'])
    print(f'observation.state.joint_position: shape={obs_joint.shape}')
    
    action_cart = np.array(sample['action.cartesian_position'])
    print(f'action.cartesian_position: shape={action_cart.shape}')
    
    action_gripper = sample['action.gripper_position']
    print(f'action.gripper_position: {action_gripper}')
    
    # Camera extrinsics
    cam_ext = np.array(sample['camera_extrinsics.exterior_1_left'])
    print(f'camera_extrinsics.exterior_1_left: shape={cam_ext.shape}')
    
    print("\nParquet data test completed!")


if __name__ == "__main__":
    print("LeRobot DROID Streaming Test")
    print("=" * 60)
    
    # Test basic loading
    dataset = test_streaming_basic()
    
    # Test parquet data (no video needed)
    test_parquet_data()
    
    # Note: Video streaming has a known bug with DROID dataset
    # See: https://github.com/huggingface/lerobot/issues/XXX
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("\nTo use with V-JEPA, you need to:")
    print("1. Sample video clips at the target FPS (e.g., 4 FPS)")
    print("2. Convert joint-space states to cartesian (or use cartesian_position)")
    print("3. Select appropriate camera view (exterior_1_left)")
