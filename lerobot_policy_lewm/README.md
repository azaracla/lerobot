# LeWM Policy Plugin for LeRobot

LeWM (LeWorldModel) JEPA world model policy for LeRobot.

A lightweight JEPA (Joint Embedding Predictive Architecture) that trains
end-to-end from pixels with only 2 loss terms:
1. **Next-embedding prediction** (MSE) — predict future latent states
2. **SIGReg** (Gaussian regularization) — keep latent embeddings well-behaved

~15M parameters, trains on a single GPU, no frozen encoder needed.

Reference: [LeWM paper](https://arxiv.org/abs/2603.19312) (MIT License)
Based on code from [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)

## Architecture

```
Images → ViT Encoder → CLS tokens → Projector → Latent embeddings
Actions → Embedder → Action embeddings ─┐
                                          ├→ ARPredictor → Predicted embeddings
Latent embeddings ────────────────────────┘

Training: MSE(predicted, target) + λ * SIGReg(embeddings)
Inference: CEM-based MPC over predicted future latents
```

## Quick Start

```bash
pip install -e .
python scripts/smoke_test.py
```

## Run Tests

```bash
pytest tests/ -v  # 56 tests
```

## Usage with LeRobot

```bash
# Train on PushT dataset
lerobot-train \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=lewm \
    --policy.n_obs_steps=4 \
    --batch_size=64 \
    --steps=50000
```

## Differences from VJEPA-AC

| Aspect | VJEPA-AC | LeWM |
|---|---|---|
| Encoder | ViT-Giant frozen (1.4B) | ViT-Tiny trainable (5M) |
| Total params | ~1.4B | ~18M |
| Attention | Custom RoPE 3D | Standard AdaLN |
| Losses | Implicit 3+ terms | 2 terms (pred + SIGReg) |
| Actions | Delta (state diffs) | Absolute |
| Code size | ~5000 lines | ~800 lines |
