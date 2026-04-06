# Rapport de comparaison : Réimplémentation V-JEPA 2-AC dans LeRobot

**Date :** 2026-04-06  
**Encodeur :** V-JEPA 2.1 ViT-Giant 384 (`facebookresearch/vjepa2`, `vjepa2_1_vit_giant_384`)  
**Référence originale :** `vjepa2/app/vjepa_droid/train.py` + `notebooks/utils/mpc_utils.py` + `world_model_wrapper.py`  
**Réimplémentation :** `lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/`  
**Dataset cible :** `azaracla/so101_pickup` (robot SO-101, 6 DOF)

---

## 1. Architecture du Predictor

### Fidélité : quasi-identique ✅

Le predictor `VisionTransformerPredictorAC` dans `ac_predictor_utils.py` est une copie directe de `vjepa2/src/models/ac_predictor.py`. Les composants copiés comprennent :

- `ACRoPEAttention` — attention avec RoPE 3D (depth, height, width), séparation des action tokens
- `ACBlock` — bloc transformer avec DropPath, choix GELU/SwiGLU
- `SwiGLUFFN` — FFN gated avec alignement hidden_dim sur 8
- `build_action_block_causal_attention_mask` — masque causal par blocs temporels
- `VisionTransformerPredictorAC` — architecture complète avec `predictor_embed`, `action_encoder`, `state_encoder`, normalisation finale, rescaling des blocs

**Seule différence structurelle** : l'original importe `ACBlock` depuis `src.models.utils.modules` ; la réimplémentation l'a copié localement dans `ac_predictor_utils.py`. L'initialisation des poids (`trunc_normal_`, `_rescale_blocks`) est identique.

### Optimisation inférence : ajout non présent dans l'original ✅

La réimplémentation ajoute un skip du masque d'attention quand `T=1` :

```python
# ac_predictor_utils.py:510
if T == 1 or self.attn_mask is None:
    attn_mask = None
```

L'original applique toujours `attn_mask` :

```python
# vjepa2/src/models/ac_predictor.py:156
attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)
```

Ce skip permet d'activer FlashAttention (via `scaled_dot_product_attention` sans masque), ce qui était le principal goulot d'étranglement (45s → 10s par action CEM). **C'est une optimisation correcte** : à T=1, le masque causal serait une matrice pleine de toute façon.

---

## 2. Encodage des frames

### Différence critique : tubelet_size ⚠️

**Original (`train.py:410`) :**
```python
c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
# [B, C, T, H, W] → [B*T, C, 1, H, W] → [B*T, C, 2, H, W]
```
Chaque frame est **dupliquée** pour créer un clip de 2 frames temporelles (tubelet_size=2 de l'encodeur).

**Réimplémentation (`modeling_vjepa_ac.py:205-209`) :**
```python
img_flat = img_seq.permute(0, 2, 1, 3, 4).reshape(B_s * T_obs, C, 1, H, W)
latents_flat = self.encoder(img_flat)  # T=1 directement
```
Chaque frame est encodée avec T=1, sans duplication.

**Impact :** L'encodeur V-JEPA 2.1 a été pré-entraîné avec des clips vidéo de tubelet_size=2. Encoder avec T=1 est potentiellement hors-distribution par rapport au pré-entraînement — bien que l'encodeur accepte T=1 techniquement (les tokens sont les mêmes en count), les activations internes diffèrent. À surveiller si les performances sont dégradées.

---

## 3. CEM (Cross-Entropy Method)

### Différences substantielles : adaptation SO-101 🔄

#### 3.1 Espace d'action

**Original (Droid/Franka, 7D) :** Le CEM original ne planifie que sur **4 dimensions** — xyz (3D) + gripper (1D) — avec la rotation forcée à 0 :
```python
# mpc_utils.py:98-105
action_samples = torch.cat([
    action_samples[:, :3],              # xyz
    torch.zeros((len(action_samples), 3), device=mean.device),  # rotation = 0
    action_samples[:, -1:],             # gripper
], dim=-1)[:, None]
```

**Réimplémentation (SO-101, 6D) :** Le CEM planifie sur **les 6 dimensions complètes** sans contraindre la rotation :
```python
# modeling_vjepa_ac.py:255
actions = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(N, H, self.config.action_dim, device=device)
actions[..., -1:] = torch.clamp(actions[..., -1:], 0.0, 1.0)
```
Cohérent avec le SO-101 qui utilise des commandes de joints (pas end-effector cartésien).

#### 3.2 Momentum gripper

**Original :** Momentum séparé pour la dimension gripper (plus conservateur) :
```python
momentum_mean=0.25, momentum_mean_gripper=0.15
momentum_std=0.95,  momentum_std_gripper=0.15
```

**Réimplémentation :** Momentum uniforme pour toutes les dimensions :
```python
momentum_mean=0.25, momentum_std=0.95  # identique pour tout
```
La distinction n'est pas critique pour le SO-101 puisque l'espace d'action est différent, mais pourrait bénéficier d'un tuning spécifique.

#### 3.3 Clamp des actions

**Original :** Clamp xyz sur `[-maxnorm, +maxnorm]`, gripper sur `[-0.75, 0.75]`, puis `round_small_elements(gripper, 0.25)` à la sortie.

**Réimplémentation :** Clamp uniquement sur la dernière dimension (gripper) : `[0, 1]`. Pas de `round_small_elements`. Adapté à la convention SO-101 où le gripper est dans `[0, 1]` et non `[-1, 1]`.

#### 3.4 Tracking de la pose (end-effector vs joints)

**Original :** `compute_new_pose()` maintient une pose cartésienne EE (xyz + euler + gripper) en composition de rotations SO(3). Le predictor reçoit à chaque pas la nouvelle pose calculée.

**Réimplémentation :** Pas de calcul de pose EE. La "state trajectory" dans le CEM rolling est simplifiée :
```python
_s = actions[:, :1]  # le state au step h = l'action précédente
```
Ce raccourci est fonctionnellement différent de l'original mais cohérent avec l'entraînement LeRobot où les états sont des positions de joints normalisées.

#### 3.5 Contexte d'observation multi-frames

**Original :** Contexte T=1 uniquement (`context_frame: [B=1, T=1, HW, D]`).

**Réimplémentation :** Contexte T=4 (`n_obs_steps=4`) pour le step 0, puis rolling T=1 pour les pas 1..H-1. Le contexte multi-frames enrichit la prédiction et correspond au training setup du predictor.

---

## 4. Loss d'entraînement

### Fidélité : identique ✅

La double loss teacher-forcing + auto-régressive est correctement reproduite :

**Original (`train.py:427-441`) :**
```python
z_tf = _step_predictor(z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1])
# + rollouts auto-régressifs
sloss = loss_fn(z_ar, h)
loss = jloss + sloss
```

**Réimplémentation (`modeling_vjepa_ac.py:382-401`) :**
```python
z_ctxt = target_latents[:, :-tokens_per_frame]
z_tf = _step_predictor(z_ctxt, actions[:, :n_action_steps], states[:, :-1], extrinsics)
# + rollouts auto-régressifs
loss = jloss + sloss
```

La fonction de loss `torch.mean(|z_pred - target|^loss_exp) / loss_exp` est identique.

**Différence mineure :** L'original dispose toujours des extrinsics caméra (Droid dataset les fournit). La réimplémentation passe `extrinsics=None` (pas disponible dans les datasets LeRobot standard), ce qui désactive `use_extrinsics` dans le predictor.

---

## 5. Normalisation des représentations

**Original :** `normalize_reps=True` par défaut dans le code de référence (encode et loss).

**Réimplémentation :** `normalize_reps=False` par défaut dans la config. Peut être activé mais non testé sur SO-101.

---

## 6. Pipeline d'entraînement

### Différences d'infrastructure

| Aspect | Original (vjepa2) | Réimplémentation (LeRobot) |
|--------|-------------------|---------------------------|
| Framework | PyTorch custom + SLURM | LeRobot `train.py` |
| Données | Droid dataset (RLDS/GCS) | HuggingFace datasets |
| Matériel | Multi-GPU A100 | Single GPU RTX 4000 16GB |
| Batch size | Configurable (dist.) | 16 (overfit config) |
| Compile | `encoder.compile()` + `predictor.compile()` pendant training | `torch.compile` appliqué APRÈS `load_state_dict` à l'inférence uniquement |
| DataLoader | Custom avec transforms vidéo | LeRobot dataloader avec DroidRandomResizedCrop |
| Extrinsics | Oui (DROID fournit les extrinsics caméra) | Non (indisponible) |
| AMP | `torch.cuda.amp.autocast(dtype=bfloat16)` | `torch.amp.autocast("cuda", dtype=bfloat16)` |

**Note sur le compile :** L'original compile l'encodeur et le predictor PENDANT l'entraînement. La réimplémentation compile uniquement le predictor au chargement des poids (inférence). Les poids sauvegardés ont un préfixe `predictor._orig_mod.` que `_load_as_safetensor()` nettoie.

---

## 7. Adaptation SO-101 vs Droid (Franka)

| Dimension | Droid/Franka (original) | SO-101 (réimpl.) |
|-----------|------------------------|------------------|
| Action dim | 7 (xyz + euler_xyz + gripper) | 6 (joints + gripper) |
| Action space | End-effector cartésien | Espace joint |
| Action convention | Delta EE + composition SO(3) | Delta joints (simple soustraction) |
| State | Pose EE 7D | Position joints 6D |
| Gripper range | [-1, 1] → round_small(0.25) | [0, 1] |
| Dataset | DROID (155k épisodes multi-scène) | azaracla/so101_pickup (single task) |
| Training scale | Multi-GPU, production | Single GPU, overfit test |

---

## 8. Points de risque identifiés

1. **Tubelet size mismatch** (critique) : L'encodeur V-JEPA 2.1 a été pré-entraîné avec `tubelet_size=2`. La réimplémentation encode avec T=1 sans duplication de frame. Cela pourrait dégrader la qualité des features par rapport aux résultats du papier.

2. **State dans le CEM rolling** : L'utilisation de `_s = actions[:, h:h+1]` comme "state" dans les steps 1..H-1 du CEM est un approximation. L'original calcule la vraie pose EE via `compute_new_pose()`. L'impact dépend de l'importance du state pour le predictor sur SO-101.

3. **Absence d'extrinsics** : Sur Droid, les extrinsics caméra sont utilisées comme conditioning supplémentaire (`use_extrinsics=True`). Sur SO-101, elles sont absentes. La réimplémentation désactive ce conditioning, ce qui est correct mais peut expliquer une partie de l'écart de performance.

4. **normalize_reps désactivé** : Si les poids pré-entraînés ont été obtenus avec `normalize_reps=True`, les activer à l'inférence pourrait améliorer les résultats.

5. **action_dim=6 vs 7** : Le `state_encoder` dans le predictor est `nn.Linear(action_embed_dim, ...)`. La réimplémentation passe `action_dim=6` à la fois pour les actions et les states, ce qui est correct pour SO-101 mais incompatible avec des poids pré-entraînés Droid (7D).

---

## 9. Résumé

| Composant | Fidélité | Commentaire |
|-----------|----------|-------------|
| Architecture predictor | ✅ Identique | Copie directe |
| Loss d'entraînement | ✅ Identique | jloss + sloss |
| Encodage frames | ⚠️ Différent | T=1 vs T=2 (tubelet) |
| CEM algorithme | ✅ Fidèle | Même structure |
| CEM espace action | 🔄 Adapté | 6D joints vs 4D EE Franka |
| CEM momentum gripper | 🔄 Simplifié | Uniforme vs séparé |
| State tracking | 🔄 Simplifié | Joints vs EE cartésien |
| normalize_reps | ⚠️ Désactivé | Potentiellement impactant |
| Extrinsics caméra | ❌ Absent | Non disponible SO-101 |
| Infrastructure train | 🔄 Adapté | LeRobot vs custom SLURM |

La réimplémentation est **structurellement fidèle** au papier V-JEPA 2-AC. Les adaptations SO-101 (action_dim=6, joints vs EE, pas d'extrinsics) sont justifiées par le changement de plateforme. Les deux risques principaux à investiguer sont l'encodage tubelet_size et l'activation potentielle de `normalize_reps`.
