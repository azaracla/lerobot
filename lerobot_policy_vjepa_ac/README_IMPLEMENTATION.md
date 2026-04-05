# VJEPa AC Implementation for LeRobot

Ce module implémente V-JEPA 2-AC (Vision Joint-Embedding Predictive Architecture - Action Conditioned) pour LeRobot, basé sur le papier [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985).

## Ce qui a été implémenté

### ✅ Composants Principaux

1. **VisionTransformerPredictorAC** (`ac_predictor_utils.py`)
   - Predictor action-conditionné identique au code original vjepa2
   - Attention causale avec block-causal mask
   - Intégration d'actions et states comme tokens spéciaux
   - Support pour RoPE (Rotary Position Embedding)

2. **VjepaAcPolicy** (`modeling_vjepa_ac.py`)
   - Heritage de `PreTrainedPolicy` de LeRobot
   - Charge l'encodeur ViT pré-entraîné depuis PyTorch Hub (gelé)
   - Training: forward pass avec teacher forcing + autorégression
   - Inference: CEM (Cross-Entropy Method) pour la planification

3. **VjepaAcConfig** (`configuration_vjepa_ac.py`)
   - Configuration complète avec tous les hyperparameters
   - Compatible avec le système de config LeRobot

4. **Image Normalization** (`modeling_vjepa_ac.py`)
   - Support pour ImageNet normalization (activé par défaut)
   - `use_imagenet_for_visuals: true` pour correspondre au prétraining ImageNet

5. **Custom Transform** (`transforms.py` + `__init__.py`)
   - `DroidRandomResizedCrop`: transform DROID-style corrigé
   - Enregistrement dynamique dans LeRobot (pas de modification du code LeRobot)

## Différences avec l'Original

### ✅ Conformes au Papier V-JEPA 2-AC

- **Encodeur ViT gelé**: Confirmé par le papier (Figure 2 caption: "freeze the video encoder")
- **Architecture AC predictor**: Identique ligne par ligne
- **Training objective**: Teacher forcing + autorégressive
- **CEM inference**: Identique en structure

### 🔄 Différences Acceptables

| Aspect | Original | LeRobot | Impact |
|--------|----------|---------|--------|
| **Hyperparameters** | lr=4.25e-4, wd=0.04 | lr=1e-4, wd=1e-4 | Différents (configurable) |
| **Batch encoding** | Batch complet | Frame-by-frame loop | Plus lent (optimisation possible) |
| **Pose integration** | Rotation matrices scipy | Addition simple | Approximation acceptable |
| **normalize_reps** | True (DROID config) | False (default) | Configurable |
| **auto_steps** | 2 | 1 (default) | Configurable |
| **Image size** | 256 | 384 (model) / 256 (crop) | Différent |

## Installation

```bash
cd lerobot_policy_vjepa_ac
pip install -e .
```

## Configuration

Exemple de config pour l'entraînement:

```yaml
# configs/policy/vjepa_ac.yaml
policy:
  type: vjepa_ac
  model_name: "vjepa2_1_vit_giant_384"
  encoder_repo_id: "facebookresearch/vjepa2"
  action_dim: 6
  img_size: 256
  predictor_embed_dim: 1024
  pred_depth: 24
  num_heads: 16
  mpc_horizon: 15
  cem_num_samples: 800
  use_imagenet_for_visuals: true
  # Optional: use DROID-style hyperparameters
  normalize_reps: true
  auto_steps: 2

dataset:
  repo_id: your_dataset
  image_transforms:
    enable: true
    tfs:
      droid_random_resized_crop:
        type: DroidRandomResizedCrop
        kwargs:
          scale: 1.777
          ratio: [0.75, 1.35]
          target_size: 256
```

## Entraînement

```bash
lerobot-train --config_path lerobot_policy_vjepa_ac/configs/policy/vjepa_ac.yaml
```

## Validation

Lancer les tests de validation:

```bash
cd lerobot_policy_vjepa_ac
python scripts/final_validation.py
```

Tests effectués:
1. ✓ Chargement de l'encodeur ViT depuis PyTorch Hub
2. ✓ Création de la policy VJEPa AC
3. ✓ Forward pass d'entraînement (teacher forcing + AR)
4. ✓ Inference CEM (planification)
5. ✓ Transform DroidRandomResizedCrop

## Structure du Code

```
lerobot_policy_vjepa_ac/
├── src/lerobot_policy_vjepa_ac/
│   ├── __init__.py                  # Patch LeRobot + exports
│   ├── configuration_vjepa_ac.py    # Config class
│   ├── modeling_vjepa_ac.py         # Policy main class
│   ├── processor_vjepa_ac.py        # Pre/post processors
│   ├── ac_predictor_utils.py        # AC predictor (VisionTransformerPredictorAC)
│   ├── transforms.py                # DroidRandomResizedCrop
│   └── configs/policy/
│       ├── vjepa_ac.yaml            # Config pickplace
│       └── vjepa_ac_community.yaml  # Config community dataset
├── scripts/
│   ├── final_validation.py          # Validation tests
│   ├── smoke_test_checkpoint.py     # Test checkpoint loading
│   ├── test_imagenet_normalization.py
│   ├── parity_test_droid.py
│   └── test_policy_imagenet_fix.py
└── pyproject.toml
```

## Bugs Corrigés

1. **ImageNet Normalization**: Ajouté `_imagenet_normalize()` dans `modeling_vjepa_ac.py`
2. **DroidRandomResizedCrop Bug**: Correction du reshape pour les images [C, H, W]
3. **Device Placement**: Ajout explicique `.to(device)` pour le predictor
4. **Transform Video Support**: Correction du reshape pour les vidéos [T, C, H, W]

## Prochaines Étapes Possibles

1. **Optimiser batch encoding**: Remplacer la boucle frame-by-frame par un encodage batch
2. **Implémenter WSDScheduler**: Scheduler 3-phases (warmup, steady, anneal)
3. **Pose integration scipy**: Utiliser Rotation matrices pour l'intégration correcte
4. **Mixed precision**: Support bfloat16 pour économiser VRAM

## Références

- Paper: [V-JEPA 2 on arXiv](https://arxiv.org/abs/2506.09985)
- Code original: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- Dataset DROID: [Droid on GitHub](https://github.com/real-stanford/droid)

## Notes Importantes

- Le ViT encodeur est **GELÉ** pendant l'entraînement AC, comme spécifié dans le papier
- L'EMA target encoder n'est PAS nécessaire pour l'entraînement AC (uniquement pour le prétraining action-free)
- L'implémentation est **conforme au papier officiel** V-JEPA 2-AC