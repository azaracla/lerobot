# VJEPA AC Training on Community Dataset - DROID Scale

## Objective

Replicate DROID-scale VJEPA AC training using the HuggingFaceVLA/community_dataset_v1 converted to LeRobot v3.0 format.

Target: ~5M frames from 128 datasets, DROID-style training parameters.

---

## 1. Dataset Conversion

### 1.1 Source Dataset

| Item | Value |
|------|-------|
| Name | HuggingFaceVLA/community_dataset_v1 |
| Format | LeRobot v2.0/v2.1 (legacy) |
| Size | 119.3 GB |
| Frames | 5,105,808 |
| Episodes | 11,132 |
| Contributors | 55 |
| Robot | SO-100 |

### 1.2 Dataset Structure

The `community_dataset_v1` contains **128 sub-datasets** from 55 contributors:
```
community_dataset_v1/
├── 00ri/so100_battery/
├── 356c/so100_duck_reposition_1/
├── AndrejOrsula/lerobot_double_ball_stacking_random/
├── ... (128 total)
```

Each sub-dataset is in v2.1 format and must be converted to v3.0.

### 1.3 Workflow

```
1. Download community_dataset_v1 from HF
2. Convert each sub-dataset from v2.1 to v3.0
3. Aggregate all v3.0 sub-datasets into one dataset
4. Train on aggregated dataset
```

### 1.4 Target Paths

| Step | Path |
|------|------|
| Download (raw v2.1) | `/mnt/nas/datasets/community_dataset_v1/` |
| Converted (v3.0) | `/mnt/nas/datasets/community_dataset_v3/` |
| Aggregated (final) | `/mnt/nas/datasets/community_aggregated/` |

### 1.5 NAS Performance Considerations

The dataset is stored on NAS (NFS4 mount at `/mnt/nas`) for storage constraints:
- **NFS4 mount**: 128KB rsize/wsize, tcp, hard mount
- **Throughput**: ~100-125 MB/s (gigabit ethernet)
- **Latency**: Higher than local SSD for random access

**Optimization for network storage:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_workers` | 16 | High count to overlap I/O across workers |
| `prefetch_factor` | 2 | Prefetch batches while GPU computes |
| `persistent_workers` | true | Avoid worker restart overhead |
| `pin_memory` | true | Faster GPU transfer |

### 1.6 Conversion Workflow

Since `community_dataset_v1` contains 128 sub-datasets, conversion is done per-sub-dataset:

```bash
# Convert each sub-dataset individually (example for one)
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id=local \
    --root=/mnt/nas/datasets/community_dataset_v3/00ri/so100_battery \
    --push-to-hub=false

# Repeat for all 128 sub-datasets...
```

A script is provided to convert all sub-datasets in batch:
```bash
python scripts/convert_community_dataset.py \
    --output_dir /mnt/nas/datasets/community_dataset_v3
```

### 1.7 Aggregation

After conversion, aggregate all sub-datasets into one:
```bash
python scripts/aggregate_community_datasets.py \
    --input_dir /mnt/nas/datasets/community_dataset_v3 \
    --output_dir /mnt/nas/datasets/community_aggregated
```

This creates a single aggregated dataset with:
- ~5M frames
- ~11k episodes
- All cameras merged

### 1.8 Verification

After aggregation, verify with:
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(repo_id="local", root="/mnt/nas/datasets/community_aggregated")
print(f"Frames: {dataset.num_frames}, Episodes: {dataset.num_episodes}")
print(f"Cameras: {dataset.camera_keys}")
print(f"Features: {list(dataset.features.keys())}")
```

---

## 2. Training Configuration

### 2.1 Config File

Create: `lerobot_policy_vjepa_ac/configs/policy/vjepa_ac_community.yaml`

### 2.2 Parameter Comparison

| Parameter | Current (SO-101 small) | DROID Original | Target (Community) |
|-----------|------------------------|----------------|---------------------|
| dataset | 1 dataset (~10k frames) | DROID (27.6M frames) | community_so100_v3 (5.1M frames) |
| batch_size | 256 | 256 (distributed) | 256 |
| learning_rate | 1e-4 | 4.25e-4 | 4.25e-4 |
| weight_decay | 1e-4 | 0.04 | 0.04 |
| warmup_steps | 500 | 4500 (15 epochs) | 4500 |
| total_steps | 100,000 | 94,500 (315 epochs) | 94,500 |
| scheduler | cosine | WSD | WSD |
| num_frames | 1 | 8 | 8 |
| tubelet_size | 1 | 2 | 2 |
| crop_size | 384 | 256 | 256 |
| normalize_reps | false | true | true |
| auto_steps | 1 | 2 | 2 |

### 2.3 WSD Scheduler (Warmup-Stable-Decay)

```
Steps 0-4500:        warmup (linear increase start_lr -> lr)
Steps 4500-90000:     stable (constant at lr)
Steps 90000-94500:    anneal (cosine decrease lr -> final_lr)
```

### 2.4 Data Augmentation (from DROID)

```yaml
augmentation:
  random_resized_crop:
    enabled: true
    scale: [1.777, 1.777]       # Fixed scale
    ratio: [0.75, 1.35]          # Aspect ratio variation
  horizontal_flip: false
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### 2.5 Camera Selection

Use top camera (overhead view) for context:
- Camera key: `observation.images.top`
- Delta timestamps for 8 frames at 4fps:
  ```python
  delta_timestamps = {
      "observation.images.top": [-1.75, -1.0, -0.5, -0.25, 0.0]  # 8 frames
  }
  ```

---

## 3. Implementation Tasks

### Task 1: WSD Scheduler

**File**: `lerobot/optim/schedulers.py`

Add `WSDScheduler` class:
```python
class WSDScheduler:
    """
    Warmup-Stable-Decay scheduler.
    
    Linear warmup, constant middle, cosine decay.
    """
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        anneal_steps: int,
        start_lr: float = 7.5e-5,
        ref_lr: float = 4.25e-4,
        final_lr: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.anneal_steps = anneal_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Warmup: linear increase
            return self.start_lr + (self.ref_lr - self.start_lr) * step / self.warmup_steps
        elif step < self.warmup_steps + self.stable_steps:
            # Stable: constant
            return self.ref_lr
        else:
            # Anneal: cosine decay
            progress = (step - self.warmup_steps - self.stable_steps) / self.anneal_steps
            progress = min(progress, 1.0)
            return self.ref_lr + (self.final_lr - self.ref_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

### Task 2: Update VjepaAcConfig

**File**: `lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/configuration_vjepa_ac.py`

Changes:
```python
# Training - change defaults to DROID values
num_frames: int = 8              # was 1
tubelet_size: int = 2             # was 1
normalize_reps: bool = True       # was False
auto_steps: int = 2               # was 1
optimizer_lr: float = 4.25e-4     # was 1e-4
optimizer_weight_decay: float = 0.04  # was 1e-4
scheduler_warmup_steps: int = 4500 # was 500
scheduler_name: str = "wsd"       # was "cosine"
scheduler_stable_steps: int = 85500  # NEW
scheduler_anneal_steps: int = 4500   # NEW
```

### Task 3: Add DROID-Style Augmentation

**File**: `lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/processor_vjepa_ac.py`

Add `RandomResizedCrop` transform:
```python
class RandomResizedCrop:
    """
    DROID-style random resized crop.
    
    Args:
        scale: Fixed scale tuple (1.777, 1.777)
        ratio: Aspect ratio range (0.75, 1.35)
        target_size: Output size (256)
    """
    def __init__(self, scale=(1.777, 1.777), ratio=(0.75, 1.35), target_size=256):
        self.scale = scale
        self.ratio = ratio
        self.target_size = target_size
    
    def __call__(self, image):
        # image: [T, C, H, W] tensor
        # Sample crop box and apply
        ...
```

### Task 4: Create Training Config YAML

**File**: `lerobot_policy_vjepa_ac/configs/policy/vjepa_ac_community.yaml`

```yaml
# @package _global_

seed: 42

dataset_repo_id: "local"
dataset_root: "/mnt/nas/datasets/community_aggregated"

override_dataset_stats:
  observation.images.top:
    mean: [[[0.485]], [[0.456]], [[0.406]]]
    std: [[[0.229]], [[0.224]], [[0.225]]]

training:
  offline_steps: 94500  # 315 epochs * 300 IPE (DROID scale)
  batch_size: 256
  eval_freq: 10000
  save_freq: 20000
  log_freq: 100
  save_checkpoint: true

eval:
  n_episodes: 50
  batch_size: 10

policy:
  name: vjepa_ac
  model_name: "vjepa2_1_vit_giant_384"
  repo_id: "facebookresearch/vjepa2"
  action_dim: 7  # SO-100: 6 joints + gripper
  img_size: 256  # DROID: 256 (not 384)
  predictor_embed_dim: 1024
  pred_depth: 24
  num_heads: 16
  mpc_horizon: 15
  cem_num_samples: 800
  n_action_steps: 1
  n_obs_steps: 4
  num_frames: 8       # DROID: 8 frames per clip
  tubelet_size: 2     # DROID: tubelet_size=2
  normalize_reps: true   # DROID
  auto_steps: 2         # DROID: 2 AR steps
  use_extrinsics: false

optimizer:
  name: adamw
  learning_rate: 4.25e-4  # DROID: 4.25e-4
  weight_decay: 0.04       # DROID: 0.04
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  name: wsd
  warmup_steps: 4500      # 15 epochs * 300 IPE
  stable_steps: 85500    # 285 epochs * 300 IPE
  anneal_steps: 4500     # 15 epochs * 300 IPE
  start_lr: 7.5e-5       # DROID start_lr
  final_lr: 0.0          # DROID final_lr

augmentation:
  random_resized_crop:
    enabled: true
    scale: [1.777, 1.777]
    ratio: [0.75, 1.35]
  horizontal_flip: false
```

---

## 4. Training Command

```bash
cd /home/arthur/Code/lerobot

# Single GPU
python -m lerobot.scripts.train \
    --config.path=lerobot_policy_vjepa_ac/configs/policy/vjepa_ac_community.yaml \
    --output_dir=./outputs/vjepa_ac_community

# Multi-GPU (e.g., 8 GPUs)
torchrun --nproc_per_node=8 -m lerobot.scripts.train \
    --config.path=lerobot_policy_vjepa_ac/configs/policy/vjepa_ac_community.yaml \
    --output_dir=./outputs/vjepa_ac_community
```

---

## 5. Expected Training Time

| GPUs | Batch Size | Steps | Est. Time |
|------|------------|-------|-----------|
| 1x A100 80GB | 256 | 94,500 | ~7 days |
| 8x A100 80GB | 2048 | 94,500 | ~1 day |

---

## 6. Monitoring

### Weights & Biases

Enable wandb in config:
```yaml
wandb:
  enable: true
  project: "vjepa-ac-community"
```

### Key Metrics to Monitor

- `train/loss` - Total loss (jloss + sloss)
- `train/jloss` - Teacher forcing loss
- `train/sloss` - Auto-regressive loss
- `train/lr` - Learning rate (should follow WSD curve)
- `eval/success_rate` - If evaluation environment available

### WSD LR Curve

Expected LR schedule:
```
Step 0:      7.5e-5 (start_lr)
Step 4500:   4.25e-4 (peak lr)
Step 90000:  4.25e-4 (stable)
Step 94500:  0.0 (final)
```

---

## 7. Files to Modify

| File | Change |
|------|--------|
| `lerobot/optim/schedulers.py` | Add WSDScheduler class |
| `lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/configuration_vjepa_ac.py` | Update defaults to DROID values |
| `lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/processor_vjepa_ac.py` | Add RandomResizedCrop augmentation |
| `lerobot_policy_vjepa_ac/configs/policy/vjepa_ac_community.yaml` | New config file (create) |

---

## 8. Verification Checklist

Before starting training:

- [ ] Verify converted dataset loads correctly
- [ ] Verify camera `observation.images.top` exists
- [ ] Verify action dim is 7 (6 joints + gripper)
- [ ] Verify WSD scheduler produces correct LR curve
- [ ] Test training for 100 steps as dry run
- [ ] Verify batch size fits in GPU memory
- [ ] Check disk space for checkpoints (~100GB)

---

## 9. References

| Resource | Path/URL |
|----------|----------|
| DROID Training Config | `vjepa2/configs/train/vitl16/droid-256px-8f.yaml` |
| DROID Data Loader | `vjepa2/app/vjepa_droid/droid.py` |
| DROID Training Script | `vjepa2/app/vjepa_droid/train.py` |
| DROID Transforms | `vjepa2/app/vjepa_droid/transforms.py` |
| LeRobot Dataset v3 Docs | https://huggingface.co/docs/lerobot/lerobot-dataset-v3 |
| VJEPA AC Policy | `lerobot_policy_vjepa_ac/` |
