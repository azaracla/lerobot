import torch
import logging
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

repo_id = "azaracla/community_dataset_v1_aggregated"
tolerance_s = 1e-4  # The PROBLEMATIC value to reproduce the error

print(f"Testing {repo_id} with tolerance_s={tolerance_s} (repro mode)...")

# Configuration matching your training run
delta_timestamps = {
    "observation.images.image2": [-0.8, -0.5333, -0.2666, 0.0]
}

try:
    dataset = StreamingLeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=tolerance_s,
        streaming=True,
        shuffle=False, # Process linearly to find the problematic timestamp faster
    )

    print("Dataset initialized. Searching for timestamps > 1020s...")
    
    found = False
    for i, item in enumerate(dataset):
        # The key in the item includes the camera name
        ts_key = "observation.images.image2_timestamps"
        if ts_key in item:
            current_ts = item[ts_key][-1]
            
            if current_ts > 1020:
                print(f"Step {i}: Successfully loaded frames at timestamp {current_ts:.4f}s")
                found = True
            
            if current_ts > 1025:
                print("Reached target window. Test SUCCESSFUL (unexpected if reproducing failure).")
                break
                
            if i % 100 == 0 and not found:
                print(f"Progress: Step {i}, current timestamp: {current_ts:.2f}s")
        
        if i > 50000: # Increased limit to ensure we hit 1024s
            print("Reached 50000 steps without hitting 1020s.")
            break

except Exception as e:
    print(f"\nTEST FAILED (SUCCESS for reproduction): {e}")
    # Don't print full traceback to keep it clean, the message is enough
else:
    print("\nNo FrameTimestampError encountered in the target range.")
