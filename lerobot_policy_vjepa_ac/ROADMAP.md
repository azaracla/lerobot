# Roadmap VJEPa AC - LeRobot

## ✅ Terminé

### 1. Correction observation_delta_indices avec FPS
- [x] Ajout de `vfps` (video FPS) et `fps` (target FPS) dans `configuration_vjepa_ac.py`
- [x] Calcul dynamique: `frame_step = round(vfps / fps)`
- [x] `vjepa_ac.yaml`: `n_obs_steps: 8`, `fps: 4`, `vfps: 30`
- [x] `vjepa_ac_96gb.yaml`: `n_obs_steps: 8`, `fps: 4`, `vfps: 30`

### 2. Correction hyperparamètres (Linear Scaling)
- [x] `vjepa_ac.yaml`: `lr: 3.32e-6` (scaled from 4.25e-4 × 2/256)
- [x] `vjepa_ac.yaml`: `weight_decay: 0.04`, `num_warmup_steps: 4500`, `scheduler_anneal_steps: 4500`
- [x] `vjepa_ac_96gb.yaml`: `lr: 1.33e-5` (scaled from 4.25e-4 × 8/256)
- [x] `vjepa_ac_96gb.yaml`: `num_warmup_steps: 4500`, `scheduler_anneal_steps: 4500`

---

## 📊 État actuel des configs

### vjepa_ac.yaml (RTX 3090/5070 Ti)
| Paramètre | Valeur | Papier | Verdict |
|-----------|--------|--------|---------|
| batch_size | 2 | 256 | ⚠️ Petit |
| lr | 3.32e-6 | 4.25e-4 | ✅ Scaled |
| weight_decay | 0.04 | 0.04 | ✅ |
| n_obs_steps | 8 | 8 | ✅ |
| fps | 4 | 4 | ✅ |
| vfps | 30 | N/A | ✅ |
| frame_step | 8 | 8 | ✅ |

### vjepa_ac_96gb.yaml (A100/H100)
| Paramètre | Valeur | Papier | Verdict |
|-----------|--------|--------|---------|
| batch_size | 8 | 256 | ⚠️ Petit |
| lr | 1.33e-5 | 4.25e-4 | ✅ Scaled |
| weight_decay | 0.04 | 0.04 | ✅ |
| n_obs_steps | 8 | 8 | ✅ |
| fps | 4 | 4 | ✅ |
| vfps | 30 | N/A | ✅ |
| frame_step | 8 | 8 | ✅ |

---

## 📋 Prochaines étapes

1. **Relancer l'entraînement** avec les nouvelles configs
2. **Vérifier** que la loss converge correctement
3. **Monitorer** les gradients (grdn dans les logs)

---

## Notes

- Training original (à arrêter): 100K steps, ~24h, loss ~0.8
- Dataset: 30fps, 1.9M frames, 4340 episodes
- Objectif: Matcher le comportement DROID (4fps, 8 frames espacées)
