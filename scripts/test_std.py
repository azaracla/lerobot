import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tqdm

ds = LeRobotDataset('azaracla/community_dataset_v1_aggregated')

# Calcul intra-episode sur 100 episodes
all_deltas = []
n_episodes = min(100, ds.meta.total_episodes)

for ep_idx in tqdm.tqdm(range(n_episodes)):
    ep = ds.meta.episodes[ep_idx]
    from_idx = ep['dataset_from_index']
    to_idx = ep['dataset_to_index']
    states = torch.stack([ds[i]['observation.state'] for i in range(from_idx, to_idx)])
    deltas = states[1:] - states[:-1]
    all_deltas.append(deltas)

all_deltas = torch.cat(all_deltas, dim=0)
print(f'Episodes: {n_episodes}, Frames: {len(all_deltas)}')
print(f'Delta mean: {all_deltas.mean(0).numpy().round(4)}')
print(f'Delta std:  {all_deltas.std(0).numpy().round(4)}')
print(f'Delta |max|: {all_deltas.abs().max(0).values.numpy().round(4)}')