# LeRobot Policy: VJEPA-AC

This component (the `vjepa_ac` policy) is an experimental implementation for [LeRobot](https://github.com/huggingface/lerobot), integrating Meta's **V-JEPA (Video Joint Embedding Predictive Architecture)** visual foundation model (`facebookresearch/vjepa2`) for robot learning tasks.

Original vjepa2 code in vjepa2 directory.

## Execution

To start a demo training session on CPU/GPU with a standard LeRobot dataset:
```bash
conda activate lerobot
lerobot-train --policy.type=vjepa_ac --dataset.repo_id=lerobot/svla_so101_pickplace --batch_size=1 --steps=100000
```

## Architecture & Implementation

The policy revolves around two primary components:

### 1. Image Encoder (Pre-trained & Frozen V-JEPA)
- The architecture is powered by **`vjepa2_1_vit_giant_384`** model (ViT-Giant).
- **RAM / VRAM Optimization**: The `torch.hub.load()` loader is encapsulated within a PyTorch device context manager (`with torch.device(...)`) to load tensors directly into GPU memory. This entirely avoids system RAM Out-Of-Memory (OOM) crashes.
- The encoder weights are strictly frozen during policy training (`requires_grad = False`).

### 2. Action-Conditioned (AC) Latent Predictor
This trainable Transformer module (`VisionTransformerPredictorAC` inside `ac_predictor_utils.py`) aims to learn the environment dynamics—predicting future V-JEPA features conditioned on the current image observation, the current physical state, and a proposed sequence of temporal actions.
- **Spatial RoPE Attention**: Builds customized attention dimensions (width, height, and depth/time) using block matrix rotations that precisely respect the ViT-Giant dimensionality and the model's `num_heads`.
- **Modern Architecture**: Incorporates layers such as *SwiGLUFFN*, Stochastic Depth (`DropPath`), temporal Causal Attention masking, and built-in support for activation checkpointing.
- **Training Strategy**: It uses an `L1 Loss` objective to directly align its temporal predictions with the latent outputs of the V-JEPA foundation model (which acts as the teacher by encoding the unmasked future trajectory).

### 3. Model Predictive Control (MPC) via Cross-Entropy Method (CEM)
Instead of generating actions in a standard "feed-forward" manner, the policy executes an iterative Model Predictive Control strategy natively:
- The agent simulates `cem_num_samples` (e.g., 800) randomized action trajectories iteratively (`cem_num_iters`) using Gaussian distributions to sample short-term rollout proposals.
- It then evaluates the quality of these rollouts via the **AC Predictor** and keeps a percentage of elite performers (`cem_elite_ratio`) to compute an optimal moving average of the trajectory to actually execute.

### Recent Fixes / Updates:
- Fixed a buggy parameter overriding where `push_to_hub` attempts defaulted to overwriting the official Meta/Facebook repository instead of the user's namespace.
- Resolved dataset integration by filtering for 4D/5D multi-dimensional arrays inside the `batch` mapping and assigning the required LeRobot specific `NormalizationMode` metrics.