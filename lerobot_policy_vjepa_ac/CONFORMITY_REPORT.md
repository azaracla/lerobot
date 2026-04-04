# Rapport de conformité : `lerobot_policy_vjepa_ac` vs `vjepa2/app/vjepa_droid`

> Analyse comparative de l'implémentation du predictor AC et de la boucle d'entraînement.
> Date : 2026-04-04

---

## 1. Fichiers comparés

| Composant | `lerobot_policy_vjepa_ac` | `vjepa2` |
|-----------|--------------------------|----------|
| Predictor | `src/lerobot_policy_vjepa_ac/ac_predictor_utils.py` | `src/models/ac_predictor.py` + `src/models/utils/modules.py` |
| Entraînement | `src/lerobot_policy_vjepa_ac/modeling_vjepa_ac.py` | `app/vjepa_droid/train.py` |

---

## 2. Architecture du predictor — ✅ Conforme

Le fichier `ac_predictor_utils.py` est une copie fidèle du predictor original.
Tous les composants suivants sont **identiques** :

| Composant | Statut |
|-----------|--------|
| `VisionTransformerPredictorAC` (signature, `__init__`, `forward`) | ✅ Identique |
| `ACBlock` (attention + MLP + drop path) | ✅ Identique |
| `ACRoPEAttention` (RoPE 3D spatio-temporel + action tokens) | ✅ Identique |
| `build_action_block_causal_attention_mask` | ✅ Identique |
| `rotate_queries_or_keys` (avec bug intentionnel `.repeat()` pour compat. poids pré-entraînés) | ✅ Identique |
| `SwiGLUFFN`, `MLP`, `DropPath` | ✅ Identiques |
| `_rescale_blocks` (rescaling des proj weights par `sqrt(2 * layer_id)`) | ✅ Identique |
| `vit_ac_predictor()` factory function | ✅ Identique |

### Divergences mineures (impact nul)

| Aspect | `lerobot` | `vjepa2` | Impact |
|--------|-----------|----------|--------|
| `trunc_normal_` | `torch.nn.init.trunc_normal_` | `src.utils.tensors.trunc_normal_` (copie ancienne PyTorch) | ✅ Aucun — fonctionnellement équivalent |
| `drop_path` | `from timm.layers` (nouvelle API) | `from timm.models.layers` (ancienne API) | ✅ Aucun — même fonction |

> **Note RoPE** : Le bug intentionnel dans `rotate_queries_or_keys` (`.repeat()` au lieu de `.repeat_interleave()`) est **présent dans les deux implémentations**. Il est conservé intentionnellement pour maintenir la compatibilité avec les poids pré-entraînés. Cf. commentaire dans `modules.py` et [PR #15](https://github.com/facebookresearch/vjepa2/pull/15).

---

## 3. Boucle d'entraînement et Inférence — ✅ Conforme (Corrigé)

Toutes les divergences identifiées précédemment ont été **corrigées** le **2026-04-04**. L'implémentation est désormais alignée sur Meta `vjepa_droid`.

### Rapport de correction

| Aspect | vjepa2/train.py | modeling_vjepa_ac.py | État |
|--------|-------------------|------------------------|----------|
| **Encoder** | Frozen (statique) | Frozen (statique) | ✅ Conforme |
| **Loss** | jloss + sloss (AR) | jloss + sloss (AR) | ✅ Corrigé |
| **Rollouts AR** | ✅ Oui (auto_steps) | ✅ Oui (auto_steps) | ✅ Corrigé |
| **Norm Latents** | LayerNorm (opt) | LayerNorm (opt) | ✅ Corrigé |
| **select_action** | N/A | CEM avec rollouts réels | ✅ Corrigé |

### Nouvelles configurations disponibles dans `VjepaAcConfig`

- `loss_exp` (float) : default 1.0. Exponent utilisé pour la perte `mean(abs**exp)`.
- `auto_steps` (int) : nombre de rollouts récursifs pour la loss (sloss).
- `normalize_reps` (bool) : applique LayerNorm sur les latents avant calcul de perte.
- `use_extrinsics` (bool) : support du 3ème token de conditionnement.


---

---

## 4. Vérification de conformité — Résultats réels

### Test de parité des poids
Le script [`scripts/parity_test.py`](scripts/parity_test.py) a validé le chargement du checkpoint officiel [`vjepa2-ac-vitg.pt`](https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt) (~11.8 GB) :
- **0 clé manquante / 0 clé inattendue.**
- Forward pass validé avec shape `(1, 1568, 1408)`.

### Test fonctionnel (Smoke Test)
Le script [`scripts/smoke_test.py`](scripts/smoke_test.py) confirme que :
- Le `forward()` calcule bien `jloss` (teacher forcing) et `sloss` (rollouts AR).
- Le `select_action()` effectue une recherche CEM réelle avec rollouts du predictor et distance au but.

```bash
# Lancer la vérification
conda run -n lerobot python scripts/smoke_test.py
```

**0 erreur détectée.** La politique est prête pour l'entraînement et l'inférence.

