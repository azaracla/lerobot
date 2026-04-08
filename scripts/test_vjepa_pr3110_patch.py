
import torch
import logging
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
import lerobot.datasets.streaming_dataset as streaming_module

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# --- MONKEY PATCH (PR #3110 logic) ---
def patched_make_frame(self, dataset_iterator):
    """Version patchée de make_frame utilisant item['timestamp'] ou item['frame_index']"""
    from lerobot.datasets.io_utils import item_to_torch
    
    item = next(dataset_iterator)
    item = item_to_torch(item)
    updates = []
    ep_idx = item["episode_index"]

    # LOGIQUE DE LA PR #3110:
    # On utilise le timestamp de l'item s'il existe, sinon frame_index relatif à l'épisode
    if "timestamp" in item:
        current_ts = item["timestamp"]
        if hasattr(current_ts, "item"):
            current_ts = current_ts.item()
    elif "frame_index" in item:
        current_ts = item["frame_index"]
        if hasattr(current_ts, "item"):
            current_ts = current_ts.item()
        current_ts /= self.fps
    else:
        # Fallback sur l'ancienne logique (problématique)
        current_ts = item["index"]
        if hasattr(current_ts, "item"):
            current_ts = current_ts.item()
        current_ts /= self.fps
    
    episode_boundaries_ts = {
        key: (
            self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
            self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
        )
        for key in self.meta.video_keys
    }

    if self.delta_indices is not None:
        query_result, padding = self._get_delta_frames(dataset_iterator, item)
        updates.append(query_result)
        updates.append(padding)

    if len(self.meta.video_keys) > 0:
        original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
        query_timestamps = self._get_query_timestamps(
            current_ts, self.delta_indices, episode_boundaries_ts
        )
        video_frames = self._query_videos(query_timestamps, ep_idx)

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                video_frames[cam] = self.image_transforms(video_frames[cam])
        updates.append(video_frames)

        if self.delta_indices is not None:
            padding_mask = self._get_video_frame_padding_mask(
                video_frames, query_timestamps, original_timestamps
            )
            updates.append({"padding_mask": padding_mask})

    for update in updates:
        item.update(update)
    
    yield item

# Application du patch
StreamingLeRobotDataset.make_frame = patched_make_frame
print("DEBUG: StreamingLeRobotDataset.make_frame has been monkey-patched with PR #3110 logic.")

# --- TEST ---
repo_id = "azaracla/community_dataset_v1_aggregated"
try:
    dataset = StreamingLeRobotDataset(
        repo_id,
        streaming=True,
        revision="main",
        shuffle=False, # On lit dans l'ordre pour voir si on atteint les épisodes lointains
    )

    print(f"Dataset {repo_id} loaded. Starting iteration...")
    
    # On itère sur un certain nombre d'items pour voir si l'erreur survient
    # L'erreur arrivait vers le step 20k (ep 181)
    # Pour le test, on va juste vérifier les 1000 premiers ou essayer de sauter plus loin
    for i, item in enumerate(dataset):
        if i % 100 == 0:
            print(f"Step {i}: ep_idx={item['episode_index']}, ts={item['timestamp']:.4f}")
        
        if i >= 1000: # On s'arrête après 1000 pour ce test rapide
            print("Reached 1000 steps without error with the patch.")
            break

except Exception as e:
    print(f"\nTEST FAILED even with patch: {e}")
    import traceback
    traceback.print_exc()
else:
    print("\nTest finished successfully (patch seems to work or error was not reached).")
