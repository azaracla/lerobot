
import torch
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

repo_id = "azaracla/community_dataset_v1_aggregated"

print(f"Inspecting dataset: {repo_id}")

try:
    dataset = StreamingLeRobotDataset(
        repo_id,
        streaming=True,
        revision="main",
    )

    # Get the first item
    it = iter(dataset)
    item = next(it)
    
    print("\nKeys available in item:")
    print(list(item.keys()))
    
    print("\nValues for the first item:")
    for key in ["index", "episode_index", "timestamp", "frame_index"]:
        if key in item:
            print(f"{key}: {item[key]}")
        else:
            print(f"{key}: NOT FOUND")

    # Check if index is global or per-episode
    # Let's skip to near the end of the first episode if possible or just check a few items
    print("\nChecking first 5 items:")
    for i in range(5):
        if i > 0: item = next(it)
        print(f"Step {i}: index={item['index']}, episode={item['episode_index']}, " + 
              (f"timestamp={item['timestamp']}" if 'timestamp' in item else "no timestamp"))

except Exception as e:
    print(f"Error: {e}")
