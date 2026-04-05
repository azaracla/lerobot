# Consolidation du Community Dataset v1

## Commande pour agréger tous les sous-datasets en un seul dataset unifié

```bash
python /home/arthur/Code/lerobot/scripts/aggregate_community_datasets.py \
    --input_dir /mnt/nas/datasets/community_dataset_v1 \
    --output_dir /mnt/nas/datasets/community_aggregated
```

## Utilisation du dataset consolidé

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local",
    root="/mnt/nas/datasets/community_aggregated"
)
print(f"Total episodes: {len(dataset.episode_indices)}")
print(f"Total frames: {len(dataset)}")
```

## Pour l'entraînement avec SmolVLA

```bash
accelerate launch --config_file accelerate_configs/multi_gpu.yaml \
    src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --policy.repo_id=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --dataset.repo_id="local" \
    --dataset.root="/mnt/nas/datasets/community_aggregated" \
    --dataset.video_backend=pyav \
    --dataset.features_version=2 \
    --output_dir="./outputs/training" \
    --batch_size=8 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"
```
