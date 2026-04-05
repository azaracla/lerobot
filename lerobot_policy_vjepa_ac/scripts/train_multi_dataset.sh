#!/bin/bash
# Multi-dataset training for vjepa_ac on community_dataset_v1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="/mnt/nas/datasets/community_dataset_v1"
CONFIG="$PROJECT_DIR/configs/policy/vjepa_ac_community.yaml"
OUTPUT="$PROJECT_DIR/outputs/train_multi_dataset/$(date +%Y-%m-%d_%H-%M-%S)_vjepa_ac"

echo "Scanning for datasets..."
DATASETS=$(python -c "
import os, json
root = '$DATASET_ROOT'
ds = []
for c in sorted(os.listdir(root)):
    p = os.path.join(root, c)
    if os.path.isdir(p) and not c.startswith('.'):
        for d in sorted(os.listdir(p)):
            if os.path.exists(os.path.join(p, d, 'meta', 'info.json')):
                ds.append(f'{c}/{d}')
print(json.dumps(ds))
")

echo "Found $(python -c "import json; print(len(json.loads('$DATASETS')))") datasets"

echo "Starting training..."
lerobot-train \
    --config_path="$CONFIG" \
    --dataset.repo_id="$DATASETS" \
    --dataset.root="$DATASET_ROOT" \
    --output_dir="$OUTPUT" \
    --save_checkpoint=true \
    --wandb.enable=true \
    --wandb.project="vjepa_ac_community"
