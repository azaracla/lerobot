# V-JEPA 2 + LeRobot Integration Plan

> **Important**: LeRobot uses a plugin-based architecture for policies. Policies must be packaged as `lerobot_policy_{name}` with specific naming conventions.

---

## Policy Integration Architecture (per LeRobot docs)

LeRobot policies must be installed as packages with this structure:

```
lerobot_policy_vjepa/
├── pyproject.toml
└── src/
    └── lerobot_policy_vjepa/
        ├── __init__.py
        ├── configuration_vjepa.py
        ├── modeling_vjepa.py
        └── processor_vjepa.py
```

**Key Integration Points:**
1. `@PreTrainedConfig.register_subclass("vjepa")` - registers the policy type
2. `make_vjepa_pre_post_processors()` - processor function (naming is strict!)
3. Inherit from `PreTrainedPolicy` base class
4. Implement: `forward()`, `predict_action_chunk()`, `select_action()`, `reset()`

---

## Current State

| Component | Status |
|-----------|--------|
| V-JEPA 2 code | ✅ Ready in `vjepa2/` |
| V-JEPA 2 ViT-L pretrained weights | ✅ You have them locally |
| DROID in V-JEPA 2 format | ❌ Not downloaded |
| LeRobot DROID (Hub) | ✅ Available at `lerobot/droid_1.0.1` (streaming supported) |

---

## Understanding the Challenge

### V-JEPA 2 DROID Format (Original)

The V-JEPA 2 training code expects DROID in a specific format:

```
trajectory_dir/
├── trajectory.h5              # h5py with robot states, actions, extrinsics
├── metadata.json              # contains camera paths (left_mp4_path, right_mp4_path)
└── recordings/MP4/
    ├── 0000.mp4               # video recording
    └── ...
```

**Data loading** ([`vjepa2/app/vjepa_droid/droid.py`](vjepa2/app/vjepa_droid/droid.py)):
- Reads `trajectory.h5` using h5py
- Extracts: `observation/camera_extrinsics`, `observation/robot_state/cartesian_position`, `observation/robot_state/gripper_position`
- Uses decord `VideoReader` for video loading
- Computes actions as pose differences
- Expects a CSV file listing trajectory directories

### LeRobot DROID Format

LeRobot v3.0 uses Parquet + MP4 with episode metadata:

```
droid/
├── meta/
│   ├── info.json              # schema, features, fps, splits
│   ├── stats.json             # normalization stats
│   └── episodes/chunk-XXX/file-XXX.parquet
├── data/chunk-XXX/file-XXX.parquet   # frame data (states, actions, etc.)
└── videos/
    └── observation.images.wrist_left/
        └── chunk-XXX/file-XXX.mp4
```

**Key features in LeRobot DROID**:
- `observation.state`: [joint_0..joint_6, gripper] (8D)
- `action`: [joint_0..joint_6, gripper] (8D)  
- `observation.images.wrist_left`, `exterior_1_left`, `exterior_2_left`
- `camera_extrinsics.wrist_left`, etc.
- `observation.state.cartesian_position`: [x, y, z, roll, pitch, yaw] (6D)

---

## Recommended Execution Plan

### Step 1: Quick Test with Original V-JEPA 2 (Optional)

Since you have pretrained weights and DROID data isn't available in original format:

```bash
# Verify V-JEPA environment works
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
cd /home/arthur/Code/vjepa2
pip install -e .

# Verify pretrained weights are loadable
python -c "
import torch
model = torch.load('models/vitl.pt')
print('Loaded successfully')
"
```

---

### Step 2: Test LeRobot Streaming Works

```bash
# In lerobot environment
conda activate lerobot  # or your lerobot conda env

python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/droid_1.0.1')
print(f'Dataset: {ds}')
sample = ds[0]
print('Keys:', list(sample.keys()))
print('State shape:', sample['observation.state'].shape)
print('Action shape:', sample['action'].shape)
print('Image keys:', [k for k in sample.keys() if 'image' in k])
"
```

---

### Step 3: Build LeRobotDataset Adapter for V-JEPA

Create a new file at `vjepa2/src/datasets/lerobot_dataset.py`:

```python
"""
LeRobotDataset Adapter for V-JEPA 2 Training

This adapter loads LeRobot-format DROID and returns samples
compatible with the V-JEPA training pipeline.
"""

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class LeRobotDroidDataset(torch.utils.data.Dataset):
    """
    Adapter that loads LeRobot-format DROID and returns
    samples compatible with V-JEPA training.
    
    Args:
        repo_id: LeRobot dataset repo_id (default: "lerobot/droid_1.0.1")
        frames_per_clip: Number of frames per video clip (default: 8)
        fps: Target FPS for sampling (default: 4)
        crop_size: Image crop size (default: 256)
        camera_view: Which camera to use (default: "exterior_1_left")
        transform: Video transforms (default: None)
    """
    
    def __init__(
        self,
        repo_id: str = "lerobot/droid_1.0.1",
        frames_per_clip: int = 8,
        fps: int = 4,
        crop_size: int = 256,
        camera_view: str = "exterior_1_left",
        transform=None,
    ):
        self.lerobot_ds = LeRobotDataset(repo_id)
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.crop_size = crop_size
        self.camera_view = camera_view
        self.transform = transform
        
        # Get dataset info
        self.fps_original = self.lerobot_ds.meta.fps  # likely 15 or 30
        self.frameskip = int(self.fps_original / fps)
        
        # Build episode index
        self.episodes = self._build_episode_index()
        
    def _build_episode_index(self):
        """Build index of all episodes with their frame ranges."""
        episodes = []
        # Access episode boundaries from metadata
        # Each episode has: start_frame, end_frame, num_frames
        ...
        return episodes
        
    def __getitem__(self, index):
        # Select random episode and temporal window
        episode_idx, start_frame = self._sample_episode_window(index)
        
        # Load video frames from LeRobot video
        frames = self._load_frames(episode_idx, start_frame)
        
        # Load robot states and compute actions
        states = self._load_states(episode_idx, start_frame)
        actions = self._compute_actions(states)
        
        # Load camera extrinsics if needed
        extrinsics = self._load_extrinsics(episode_idx, start_frame)
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
            
        return frames, actions, states, extrinsics
        
    def __len__(self):
        return len(self.episodes)
```

---

### Step 4: Key Mappings to Implement

| V-JEPA 2 (Original) | LeRobot DROID | Implementation Note |
|---------------------|---------------|----------------------|
| `trajectory.h5` + metadata.json | Parquet chunks + video MP4s | Use `LeRobotDataset[index]` |
| `states = [cartesian_pos(6D) + gripper(1D)]` | `observation.state(8D)` or `observation.state.cartesian_position` + `observation.state.gripper_position` | Need cartesian conversion |
| `actions = poses_to_diffs(states)` | `action` (joint-space) | V-JEPA uses cartesian delta actions |
| Camera views: `left_mp4_path` | `observation.images.exterior_1_left` | Map camera names |
| `camera_extrinsics` | `camera_extrinsics.exterior_1_left` | Extract from LeRobot features |
| CSV file listing trajectories | Episodes metadata | Use `meta/episodes/` |

---

### Step 5: Integrate V-JEPA as LeRobot Policy

Create policy structure in `src/lerobot/policies/vjepa/`:

```
src/lerobot/policies/vjepa/
├── __init__.py
├── configuration_vjepa.py      # Config class
├── modeling_vjepa.py           # V-JEPA encoder + AC predictor
└── preprocessing.py           # Image/video preprocessing
```

---

## Phase 1: Verify Original V-JEPA 2 Training (Optional)

If you want to first understand V-JEPA training with original format:

1. Download DROID samples using gsutil:
```bash
./download_droid_test_samples.sh
```

2. Edit config to point to your local data:
```yaml
# configs/train/vitl16/droid-256px-8f.yaml
data:
  datasets:
    - /home/arthur/Code/vjepa2/droid_test_index.csv
  dataset_fpcs:
    - 8
```

3. Run training:
```bash
python -m app.main --fname configs/train/vitl16/droid-256px-8f.yaml \
  --devices cuda:0
```

---

## Phase 2: Create LeRobotDataset Adapter

### Core Implementation Details

1. **Video Frame Loading**
   - LeRobot stores videos in MP4 chunks
   - Use LeRobot's video decoding utilities
   - Sample frames at target FPS with temporal window

2. **State Format Conversion**
   - LeRobot: `observation.state` = [joint_pos(7D), gripper(1D)] 
   - V-JEPA expects: `states` = [cartesian_pos(6D), gripper(1D)]
   - Need forward kinematics to convert joint → cartesian
   - OR use `observation.state.cartesian_position` directly if available

3. **Action Computation**
   - V-JEPA computes: `actions = poses_to_diffs(states)`
   - LeRobot already has `action` field
   - Need to verify if LeRobot actions match V-JEPA's expected format

4. **Camera Selection**
   - V-JEPA uses single camera (typically `exterior_1_left`)
   - LeRobot has: `wrist_left`, `exterior_1_left`, `exterior_2_left`

---

## Phase 3: Integrate V-JEPA as LeRobot Policy

### Configuration Class

```python
# src/lerobot/policies/vjepa/configuration_vjepa.py
@dataclass
class VJEPAPretrainingConfig:
    """For pre-training V-JEPA encoder on video."""
    model_name: str = "vit_large"
    pretrained_path: str = "/path/to/vitl.pt"
    image_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    frames_per_clip: int = 8
    learning_rate: float = 4.25e-4
    epochs: int = 315
    
@dataclass  
class VJEPAActionConditionedConfig:
    """For post-training (action-conditioned)."""
    pretrained_encoder: str = "/path/to/vitl.pt"
    model_name: str = "vit_large"
    pred_depth: int = 24
    pred_embed_dim: int = 1024
    learning_rate: float = 4.25e-4
    ...
```

### Model Implementation

```python
# src/lerobot/policies/vjepa/modeling_vjepa.py
class VJEPAPretrainingPolicy(BasePolicy):
    """V-JEPA encoder for pre-training."""
    
    def __init__(self, config):
        # Load pretrained V-JEPA encoder from vjepa2
        self.encoder = load_vjepa_encoder(config.pretrained_path)
        
    def forward(self, batch):
        # Extract V-JEPA features from video frames
        ...
        
class VJEPAActionConditionedPolicy(BasePolicy):
    """V-JEPA with action-conditioned predictor for robot control."""
    
    def __init__(self, config):
        # Load pretrained encoder + AC predictor from V-JEPA 2-AC
        self.encoder, self.predictor = load_vjepa2_ac(config.pretrained_path)
        
    def forward(self, batch):
        # Encode observation images
        # Predict actions
        ...
```

---

## Phase 4: Support Post-Training on Any LeRobot Dataset

Once integrated as a LeRobot policy:

```bash
# Training example
python -m lerobot.train \
    --policy=vjepa \
    --config.vjepa.pretrained_encoder=/path/to/vitl.pt \
    --config.vjepa.learning_rate=4.25e-4 \
    --config.vjepa.epochs=315 \
    --dataset.repo_id="lerobot/droid_1.0.1"
```

---

## Key Technical Decisions Needed

1. **Video Backend**: Use decord (V-JEPA) or torchcodec (LeRobot)?
   - Recommendation: Create a unified video loader interface

2. **Action Space**: V-JEPA 2-AC predicts in latent space
   - LeRobot uses joint-space actions
   - Need to verify: Does V-JEPA 2-AC output actions directly or latent features?

3. **Camera Selection**: Which camera to use?
   - V-JEPA was trained with single camera view
   - LeRobot DROID has: wrist_left, exterior_1_left, exterior_2_left

---

## Resources

- [V-JEPA 2 Paper](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2 Blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)
- [LeRobot Dataset v3.0 Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [DROID Dataset](https://droid-dataset.github.io/)
- [LeRobot DROID on HuggingFace](https://huggingface.co/datasets/lerobot/droid_1.0.1)
- V-JEPA 2 pretrained weights: `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`

---

## Summary: Your Path Forward

1. **Today**: 
   - Test LeRobot streaming works: `LeRobotDataset('lerobot/droid_1.0.1')`
   - Understand LeRobot dataset structure

2. **Next**:
   - Create `LeRobotDroidDataset` adapter in vjepa2
   - Implement video frame loading from LeRobot format
   - Implement state/action format conversion

3. **Goal**:
   - V-JEPA post-training on LeRobot-format DROID
   - V-JEPA as a LeRobot policy for any dataset

---

## Step 1: Test LeRobot Streaming with DROID

### Script to Run

```bash
cd /home/arthur/Code/lerobot
conda activate vjepa2-312

python -c "
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Create streaming dataset (this will stream from HuggingFace Hub)
dataset = StreamingLeRobotDataset(
    repo_id='lerobot/droid_1.0.1',
    streaming=True,
    buffer_size=100,
    shuffle=True,
)

print('Dataset loaded!')
print(f'  Total frames: {dataset.num_frames}')
print(f'  Total episodes: {dataset.num_episodes}')
print(f'  FPS: {dataset.fps}')
print(f'  Video keys: {dataset.meta.video_keys}')
print(f'  Features: {list(dataset.meta.info[\"features\"].keys())}')

# Get a few samples
for i, sample in enumerate(dataset):
    print(f'\\nSample {i}:')
    print(f'  episode_index: {sample[\"episode_index\"].item()}')
    print(f'  frame_index: {sample[\"frame_index\"].item()}')
    if 'observation.state' in sample:
        print(f'  observation.state shape: {sample[\"observation.state\"].shape}')
    if 'action' in sample:
        print(f'  action shape: {sample[\"action\"].shape}')
    for key, value in sample.items():
        if 'image' in key:
            print(f'  {key} shape: {value.shape}')
    if i >= 4:
        break
"
```

### Key Classes for Streaming

The correct class to use is `StreamingLeRobotDataset` from:
```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
```

**Key features:**
- Streams data from HuggingFace Hub without downloading entire dataset
- Uses `Backtrackable` iterator for temporal access (look back/look ahead)
- Video decoding is cached to avoid re-initializing decoders
- Supports `delta_timestamps` for querying frames at different time offsets

### What to Look For

After running the test, verify:
1. ✅ Dataset loads from Hub
2. ✅ `num_frames` and `num_episodes` are populated
3. ✅ `observation.state` has shape `[8]` (7 joints + gripper)
4. ✅ `action` has shape `[8]` (joint-space actions)
5. ✅ Video images load correctly

### Next: Understand the Video Structure

The DROID dataset has 3 cameras:
- `observation.images.wrist_left` (180x320)
- `observation.images.exterior_1_left` (180x320) - **main camera used by V-JEPA**
- `observation.images.exterior_2_left` (180x320)

V-JEPA expects:
- Single camera view (typically `exterior_1_left`)
- Cartesian-space robot states
- Actions computed as pose differences

---

## Step 1: Test Results

### Streaming Works
```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
dataset = StreamingLeRobotDataset(repo_id='lerobot/droid_1.0.1', streaming=True)
# Result: ✅ Dataset loads, 27,618,651 frames, 95,617 episodes, FPS=15
```

### ⚠️ Video Streaming Bug
The `StreamingLeRobotDataset` has a bug with video frame indexing:
```
IndexError: Invalid frame index=1572624 for streamIndex=0; must be less than 145682
```
This is a bug in the LeRobot code when handling episode boundaries with video timestamps.

### Parquet Data Structure (Verified)

**LeRobot DROID Format:**
| Feature | Shape | Description |
|---------|-------|-------------|
| `observation.state` | (8D) | [joint_0..joint_6, gripper] joint space |
| `observation.state.cartesian_position` | (6D) | [x, y, z, roll, pitch, yaw] |
| `observation.state.gripper_position` | scalar | gripper position (0.0 = open) |
| `observation.state.joint_position` | (7D) | [joint_0..joint_6] |
| `action` | (8D) | same as observation.state |
| `action.cartesian_position` | (6D) | cartesian position |
| `action.gripper_position` | scalar | gripper position |
| `action.joint_position` | (7D) | joint position |
| `camera_extrinsics.exterior_1_left` | (6D) | [x, y, z, roll, pitch, yaw] |

**V-JEPA 2 Format:**
| Feature | Shape | Description |
|---------|-------|-------------|
| `states` | (7D) | [cartesian_pos(6D), gripper(1D)] |
| `actions` | computed | `poses_to_diffs(states)` |

### Key Mapping

| LeRobot → V-JEPA | Conversion |
|------------------|------------|
| `observation.state.cartesian_position` + `observation.state.gripper_position` | → `states` |
| LeRobot `action` (joint-space) | ≠ V-JEPA `actions` (cartesian deltas) |
| `observation.images.exterior_1_left` | → Camera view |

### Next Step: Build Adapter

The adapter needs to:
1. Sample video clips at target FPS (e.g., 4 FPS for V-JEPA)
2. Extract cartesian states from LeRobot format
3. Compute action differences (or use LeRobot's cartesian actions)
4. Handle camera extrinsics for coordinate transforms

---

## Phase 3: Integrate V-JEPA as LeRobot Policy

### Policy Package Structure

LeRobot uses plugin-based architecture for policies. Create:

```
lerobot_policy_vjepa/
├── pyproject.toml
└── src/
    └── lerobot_policy_vjepa/
        ├── __init__.py
        ├── configuration_vjepa.py
        ├── modeling_vjepa.py
        └── processor_vjepa.py
```

### Configuration Class

```python
# configuration_vjepa.py
from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("vjepa")
@dataclass
class VJEPAPolicyConfig(PreTrainedConfig):
    """Configuration for V-JEPA action-conditioned policy."""
    
    # Model
    pretrained_path: str = "facebook/vjepa2-vitg-fpc64-256"
    model_name: str = "vit_large"
    image_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    
    # Training
    horizon: int = 8  # action chunk length
    learning_rate: float = 4.25e-4
    weight_decay: float = 0.04
    
    def validate_features(self):
        if not self.image_features:
            raise ValueError("VJEPAPolicy requires image features")
        if self.action_feature is None:
            raise ValueError("VJEPAPolicy requires action feature")
    
    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.learning_rate, weight_decay=self.weight_decay)
```

### Model Class

```python
# modeling_vjepa.py
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

class VJEPAPolicy(PreTrainedPolicy):
    config_class = VJEPAPolicyConfig
    name = "vjepa"
    
    def __init__(self, config, dataset_stats=None):
        super().__init__(config, dataset_stats)
        # Load V-JEPA encoder from vjepa2
        # Load action-conditioned predictor
        self.model = ...
    
    def reset(self):
        # Reset episode state
        pass
    
    def forward(self, batch):
        # Training: compute loss
        ...
        return {"loss": loss}
    
    def predict_action_chunk(self, batch):
        # Return action chunk for training
        ...
    
    def select_action(self, batch):
        # Return single action for inference
        ...
```

### Processors (Naming is Strict!)

```python
# processor_vjepa.py
def make_vjepa_pre_post_processors(config, dataset_stats):
    # Function name MUST be exactly: make_{policy_name}_pre_post_processors
    preprocessor = ...
    postprocessor = ...
    return preprocessor, postprocessor
```

### Installation and Usage

```bash
cd lerobot_policy_vjepa
pip install -e .

# Use with LeRobot
lerobot-train \
    --policy.type vjepa \
    --policy.vjepa.pretrained_path=/path/to/vitl.pt \
    --dataset.repo_id=lerobot/droid_1.0.1
```

---

## Real Example: DiTFlow Policy

A concrete example is the [DiTFlow policy](https://github.com/danielsanjosepro/lerobot_policy_ditflow) which follows the same architecture.

### Package Structure
```
lerobot_policy_ditflow/
├── src/lerobot_policy_ditflow/
│   ├── __init__.py
│   ├── configuration_ditflow.py
│   ├── modeling_ditflow.py
│   └── processor_ditflow.py
```

### Key Implementation Details from DiTFlow

**1. Configuration:**
```python
@PreTrainedConfig.register_subclass("ditflow")
@dataclass
class DiTFlowConfig(PreTrainedConfig):
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    vision_backbone: str = "resnet18"
    hidden_dim: int = 512
```

**2. Policy Class:**
```python
class DiTFlowPolicy(PreTrainedPolicy):
    config_class = DiTFlowConfig
    name = "ditflow"
    
    def __init__(self, config, dataset_stats=None):
        super().__init__(config, dataset_stats)
        # Load vision encoder, noise net, etc.
        
    def forward(self, batch):
        # Training: compute diffusion loss
        return {"loss": loss}
        
    def generate_actions(self, batch):
        # Inference: sample actions
        ...
```

**3. Key Methods Required (from PreTrainedPolicy):**
- `forward(batch)` → returns `{"loss": tensor}`
- `predict_action_chunk(batch)` → returns action chunk
- `select_action(batch)` → returns single action
- `reset()` → clear caches

---

## Implementation Order for V-JEPA

### Step 1: Adapter (LeRobot → V-JEPA data format)
Before integrating as a policy, create an adapter to verify data compatibility:
- Sample video clips at 4 FPS
- Extract `observation.state.cartesian_position` + `observation.state.gripper_position`
- Match V-JEPA's expected output format

### Step 2: Policy Package
1. Create `lerobot_policy_vjepa` package
2. Implement `VJEPAPolicyConfig` with V-JEPA 2-AC parameters
3. Implement `VJEPAPolicy` class
4. Add `make_vjepa_pre_post_processors()`
5. Install and test with `lerobot-train`

### Step 3: Training
```bash
lerobot-train \
    --policy.type vjepa \
    --policy.vjepa.pretrained_path=/path/to/vitl.pt \
    --policy.vjepa.image_size=256 \
    --dataset.repo_id=lerobot/droid_1.0.1 \
    --steps 100000
```
