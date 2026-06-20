# LeWM Plugin — Rapport technique

## Contexte

On avait un plugin `lerobot_policy_vjepa_ac` (~5000 lignes) qui réimplémentait VJEPA2 de Meta dans
l'écosystème LeRobot. Trop complexe : ViT-Giant gelé (1.4B params), attention RoPE 3D custom,
delta actions, calibration CEM per-joint.

Depuis, deux implémentations récentes de world models sont sorties :

- **[LeWM](https://github.com/lucas-maes/le-wm)** (MIT) — JEPA ~15M params entraîné de zéro avec
  **2 loss terms** (MSE prédiction + régularisation Gaussienne), pas d'encodeur gelé
- **[Stable-Worldmodel](https://github.com/galilai-group/stable-worldmodel)** — framework avec
  solvers CEM/iCEM/MPPI, environnements standardisés, adapter LeRobot

## Ce qui a été fait

**Nouveau plugin `lerobot_policy_lewm`** (~2500 lignes code + tests) :

| Fichier | Lignes | Contenu |
|---|---|---|
| `modules.py` | 550 | ViTEncoder, Transformer, ARPredictor, ConditionalBlock, SIGReg, MLP, Embedder |
| `jepa.py` | 304 | JEPA: encode, predict, rollout, criterion, get_cost, forward (training loss) |
| `solver.py` | 288 | CEMSolver, ICEMSolver (portés depuis stable-worldmodel) |
| `configuration_lewm.py` | 141 | LeWMConfig (PreTrainedConfig, enregistré via `@register_subclass("lewm")`) |
| `modeling_lewm.py` | 292 | LeWMPolicy (PreTrainedPolicy) — forward training + select_action CEM |
| `processor_lewm.py` | 141 | Pre/post processors (resize 224, ImageNet norm, z-score norm) |
| Tests (4 fichiers) | 764 | 56 tests : modules, JEPA, policy, solvers |

### Architecture

```
Images → ViT Encoder (timm, ~5M params) → CLS tokens → Projector → Latent embeddings
Actions → Embedder → Action embeddings ─┐
                                          ├→ ARPredictor (AdaLN) → Predicted embeddings
Latent embeddings ────────────────────────┘

Training:
  pred_loss = MSE(predicted_emb, target_emb)
  sigreg_loss = SIGReg(embeddings)  ← Gaussian regularizer
  loss = pred_loss + 0.09 * sigreg_loss

Inference (MPC):
  select_action() → CEM sur get_cost() = MSE(rollout_final_latent, goal_latent)
```

### Simplifications vs VJEPA-AC

| | VJEPA-AC | LeWM Plugin |
|---|---|---|
| Encodeur | ViT-Giant gelé (1.4B) | ViT-Tiny entraînable (5M) |
| Total params | ~1.4B | ~18M |
| Attention | RoPE spatial 3D custom | Transformer standard avec AdaLN-zero |
| Losses | 3+ implicites | 2 explicites (pred MSE + SIGReg) |
| Actions | Delta (state differences) | Absolues |
| Code source | ~5000 lignes | ~2500 lignes |
| Tests | 0 | 56 tests unitaires |
| Dépendances | torch.hub, Meta encoder | timm uniquement |

### Résultats des tests

- **56/56 tests passent** (forward, backward, CEM convergence, checkpoint save/load)
- **Training smoke test** : loss 0.18 → 0.0025 en 50 steps sur données synthétiques (99% réduction)
- **Inférence CEM** : `select_action()` fonctionnel, embedding cache maintenu, pas de NaN
- **Checkpoint roundtrip** : save → load → même output

### À venir

- Training sur `lerobot/pusht` (206 épisodes, 25K frames) — en cours
- Évaluation MPC sur environnement PushT
- Comparaison avec les résultats LeWM publiés (~80% success rate)
- Extension à TwoRoom, Cube, données robot réelles
