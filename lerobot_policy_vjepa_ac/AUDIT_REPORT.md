# Rapport d'Audit : V-JEPA 2-AC intégré dans LeRobot

## 1. Vue d'ensemble du projet

**Objectif** : Réimplémenter V-JEPA 2-AC (Meta FAIR) dans LeRobot pour du zero-shot manipulation sur un bras SO-101 (6 DoF). Le predictor AC doit être ré-entraîné car :
- Changement d'encodeur : V-JEPA 2 → V-JEPA 2.1 (features denses améliorées)
- Changement de morphologie : DROID Franka 7D end-effector → SO-101 6D joint space
- Changement de dataset : DROID (62h, téléopération) → community_dataset_v1_aggregated

**État** : Entraînement en cours, loss ~0.930 au step 10K, batch=4 sur RTX 5070 Ti.

---

## 2. Fidélité architecturale vs. l'implémentation originale Meta

### 2.1 VisionTransformerPredictorAC — FIDÈLE

Le predictor (`ac_predictor_utils.py`) est un portage quasi ligne-à-ligne de `vjepa2/src/models/ac_predictor.py` :
- Même structure : `predictor_embed` → interleave [action, state, patches] → 24 ACBlocks avec RoPE → `predictor_norm` → `predictor_proj`
- Même masque causal : `build_action_block_causal_attention_mask(T, H, W, add_tokens=2)`
- Même RoPE 3D : rotation séparée depth/height/width via `rotate_queries_or_keys`
- Même initialisation : `trunc_normal_(std=0.02)` + `_rescale_blocks(sqrt(2*layer_id))`

**Seule différence ajoutée** (bonne) : skip du masque attention quand `T==1` pour activer FlashAttention dans la boucle CEM rolling (`ac_predictor_utils.py:510`). L'original force toujours le masque.

### 2.2 Training Loss — FIDÈLE (avec extensions)

Le forward (`modeling_vjepa_ac.py:276-414`) reproduit exactement la logique du `train.py` original :

| Composant | Original (train.py:426-441) | Ton code (modeling_vjepa_ac.py:374-411) |
|---|---|---|
| Teacher forcing (jloss) | `z_tf = predictor(z[:,:-N], actions, states[:,:-1])` | Identique |
| Auto-regressive (sloss) | Boucle `for n in range(1, auto_steps)` | Identique |
| Loss fn | `mean(abs(z-h)^exp) / exp` | Identique + masque `action_is_pad` |

L'ajout du masque `action_is_pad` est une bonne adaptation LeRobot pour gérer les fins d'épisodes.

### 2.3 Encodeur — ADAPTÉ CORRECTEMENT

- Original : V-JEPA 2 ViT-Giant frozen + `target_encoder` (copie EMA)
- Nous : V-JEPA 2.1 ViT-Giant-384 frozen (un seul encodeur, pas besoin d'EMA car pas de pretraining SSL ici)
- Encodage identique : frame-by-frame, tubelet duplication `repeat(1,1,2,1,1)`, ImageNet norm

---

## 2b. TRACE COMPLÈTE DE LA CHAÎNE PREPROCESSOR

### Training : Dataset → preprocessor → forward()

```
ÉTAPE 0 — Dataset (LeRobotDataset)
  observation.images.image  : [B, T=4, C=3, H=384, W=384]   # 4 frames à ~4fps
  observation.state         : [B, T=4, D=6]                   # positions joints absolues
  action                    : [B, H=15, D=6]                  # 15 positions futures absolues
  action_is_pad             : [B, H=15]                        # padding flag

ÉTAPE 1 — batch_to_transition (défaut pipeline)
  transition["observation"]["observation.state"]  : [B, 4, 6]  (inchangé)
  transition["action"]                            : [B, 15, 6] (inchangé)
  transition["complementary_data"]["action_is_pad"] : [B, 15]

ÉTAPE 2 — RenameObservations (no-op)
ÉTAPE 3 — AddBatchDimension (no-op, batch dim déjà là via DataLoader)
ÉTAPE 4 — DeviceStep (→ cuda)

ÉTAPE 5 — StateToDeltaActionProcessorStep
  ✅ state = obs["observation.state"]         : [B, 4, 6] — BRUT (pas encore normalisé)
  ✅ delta = state[:, 1:] - state[:, :-1]     : [B, 3, 6] — deltas joints bruts
  ✅ transition["action"] = delta              : [B, 3, 6] — REMPLACE les 15 actions futures
  ✅ _current_state = state[:, -1]             : [B, 6]    — caché pour postprocessor
  ⚠️  action_is_pad reste [B, 15]             : mismatch shape (bénin, forward() slice à :3)

ÉTAPE 6 — NormalizerProcessorStep
  VISUAL → IDENTITY : images inchangées [0, 1]
  STATE  → MIN_MAX  : observation.state → 2*(x-min)/(max-min) - 1 → [-1, 1]
  ACTION → IDENTITY : deltas bruts inchangés (forcé par use_delta_actions=True)

ÉTAPE 7 — transition_to_batch → batch dict

RÉSULTAT POUR forward() :
  images  : [B, 4, 3, 384, 384]  — pixels [0,1] (ImageNet norm appliquée dans forward)
  states  : [B, 4, 6]             — MIN_MAX normalisé [-1, 1]
  actions : [B, 3, 6]             — deltas bruts (ex: [-0.03, 0.01, ...])
  → predictor(context_latents, raw_deltas, normalized_states) ✅ CORRECT
```

### Inférence : Robot obs → preprocessor → select_action()

```
ÉTAPE 0 — Robot fournit (1 frame, ou N frames accumulées)
  observation.images.image  : [C=3, H, W]  ou [T, C, H, W]
  observation.state         : [D=6]         ou [T, D]

ÉTAPE 1-4 — batch_to_transition + Device
  images : [1, C, H, W]  ou [1, T, C, H, W]  (avec batch dim)
  state  : [1, D]         ou [1, T, D]

ÉTAPE 5 — StateToDeltaActionProcessorStep
  Pas de clé "action" dans la transition robot → ne modifie rien
  Cache _current_state = state[:, -1]

ÉTAPE 6 — NormalizerProcessorStep
  STATE → MIN_MAX : observation.state normalisé [-1, 1]

RÉSULTAT POUR select_action() :
  images : normalisées (non-ImageNet, fait dans select_action)
  states : [1, T, 6] MIN_MAX normalisé
  ❌ hist_actions = states[:, 1:] - states[:, :-1]  →  deltas de states NORMALISÉS
     (vs. training : deltas BRUTS) → BUG §3.1
  ❌ Si images.ndim==5, assume [B,C,T,H,W] mais LeRobot donne [B,T,C,H,W] → BUG §3.4
```

### Verdict sur l'entraînement actuel

**Le teacher forcing (jloss) est entraîné correctement.** La chaîne preprocessor training est cohérente : deltas bruts calculés avant normalisation, action IDENTITY, states MIN_MAX. Le predictor apprend la bonne correspondance.

**Ce qui manque** : `auto_steps=1` → pas de rollout loss (sloss ≈ sous-ensemble du jloss). Le predictor n'est pas entraîné pour les rollouts autorégressifs.

**Conclusion** : Les poids actuels ne sont **pas perdus** — ils ont appris la prédiction 1-step (teacher forcing). Mais il faut **reprendre l'entraînement avec `auto_steps=2`** pour que le predictor apprenne aussi le mode autorégressif. Avec `mpc_horizon=1`, le modèle actuel est déjà utilisable pour de l'inférence (après fix des bugs 3.1 et 3.4).

Note : SO-101 a 6 joints réguliers (y compris le gripper qui est un servo comme les autres). Pas besoin de traitement spécial pour le gripper, contrairement à DROID qui a un gripper binaire séparé (index 6, dimension 7).

---

## 3. PROBLÈMES CRITIQUES

### 3.1 Mismatch train/inférence sur les actions historiques

**Sévérité : CRITIQUE**

Pendant l'entraînement (`forward()`) :
- Le preprocessor `StateToDeltaAction` calcule `delta = state[:, 1:] - state[:, :-1]` sur les **states bruts** (avant normalisation MIN_MAX)
- L'action normalization est IDENTITY → le predictor voit des **deltas bruts**

Pendant l'inférence (`select_action()`, ligne 215) :
```python
hist_actions = states[:, 1:] - states[:, :-1]  # states DÉJÀ MIN_MAX normalisés!
```
- `states` a traversé le normalizer (MIN_MAX) avant d'arriver dans `select_action`
- Les deltas sont calculés sur des **states normalisés** → espace différent

Concrètement, si MIN_MAX normalise `[min, max] → [-1, 1]` :
- Delta brut : `s_{t+1} - s_t` ≈ [-0.05, 0.05] rad
- Delta normalisé : `(s_{t+1} - s_t) × 2/(max-min)` → amplitude totalement différente

**Le predictor reçoit des hist_actions dans un espace qu'il n'a jamais vu pendant l'entraînement.**

**Fix — `processor_vjepa_ac.py` : cacher les deltas bruts dans le preprocessor**

```python
# Dans StateToDeltaActionProcessorStep.__call__, après le calcul de delta :
# (processor_vjepa_ac.py, dans le bloc state.ndim >= 3)

self._current_state = state[:, -1]
self._raw_deltas = state[:, 1:] - state[:, :-1]  # AJOUTER : cache les deltas bruts
_DELTA_STATE_CACHE["current_state"] = self._current_state
_DELTA_STATE_CACHE["raw_deltas"] = self._raw_deltas  # AJOUTER
```

**Fix — `modeling_vjepa_ac.py` : utiliser les deltas bruts à l'inférence**

```python
# Dans select_action(), remplacer le bloc lignes 213-218 :

# AVANT (BUGUÉ) :
# if states.ndim == 3:
#     hist_states = states
#     hist_actions = states[:, 1:] - states[:, :-1]  # ← deltas de states normalisés !

# APRÈS (FIX) :
from .processor_vjepa_ac import _DELTA_STATE_CACHE

if states.ndim == 3:
    hist_states = states  # normalisé, OK pour le state_encoder
    # Utiliser les deltas bruts cachés par le preprocessor (avant normalisation)
    raw_deltas = _DELTA_STATE_CACHE.get("raw_deltas")
    if raw_deltas is not None and raw_deltas.shape[0] == B:
        hist_actions = raw_deltas.to(device)
    else:
        # Fallback : deltas normalisés (mieux que rien)
        hist_actions = states[:, 1:] - states[:, :-1]
```

### 3.2 CEM : cem_maxnorm jamais appliqué

**Sévérité : CRITIQUE**

Le config définit `cem_maxnorm: 0.05` mais **cette valeur n'est jamais utilisée dans le code CEM** (`modeling_vjepa_ac.py:242-271`). Aucun clipping des actions samplées.

L'original (`mpc_utils.py:94-95`) clippe strictement :
```python
action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
```

Avec `cem_std=0.5` et pas de clipping, les actions samplées ont des magnitudes ~0.5 rad, alors que les deltas joints typiques SO-101 sont ~0.01-0.05 rad/step. **~95% des samples sont hors distribution.**

**Fix — `modeling_vjepa_ac.py` : ajouter le clipping après le sampling**

```python
# Dans select_action(), boucle CEM, juste après le sampling (ligne ~243) :

for _ in range(self.config.cem_num_iters):
    actions = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(N, H, self.config.action_dim, device=device)

    # AJOUTER : clipping des actions samplées (comme l'original mpc_utils.py:94-95)
    maxnorm = self.config.cem_maxnorm
    actions = torch.clamp(actions, -maxnorm, maxnorm)

    # ... reste de la boucle CEM
```

### 3.3 CEM rolling : actions utilisées comme states (bug H>1)

**Sévérité : CRITIQUE** (si mpc_horizon > 1)

Dans la boucle CEM rolling (`modeling_vjepa_ac.py:254-262`), les **actions** (deltas bruts ~0.01-0.05) sont passées au `state_encoder` qui attend des **positions normalisées** ([-1, 1]) :

```python
# BUG — ligne 254
_s = actions[:, :1]          # ← delta brut, PAS une position !
for h in range(1, H):
    _a = actions[:, h : h + 1]
    pred_h = self.predictor(current_z, _a, _s)  # ← _s = delta, pas un state !
    _s = _a                  # ← encore un delta
```

L'original (`world_model_wrapper.py:56-63`) maintient correctement une trajectoire de poses :

```python
# Original Meta
next_rep = self.predictor(reps, actions, poses)          # poses = vraies positions
next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])  # pose + delta → nouvelle pose
```

**Impact** : Le predictor reçoit des valeurs ~0.01-0.05 là où il attend [-1, 1]. Avec `mpc_horizon=1` ce code ne s'exécute pas (la boucle `range(1, 1)` est vide), donc **le fix prioritaire est de mettre `mpc_horizon=1`**. Si horizon > 1 est nécessaire plus tard, il faut maintenir une trajectoire d'états.

**Fix — `modeling_vjepa_ac.py` : maintenir la trajectoire d'état (nécessaire si H > 1)**

```python
# Remplacer le bloc lignes 253-262 :

# Compute initial predicted state from last known state + first action
last_known_state = hist_states[:, -1:]  # [N, 1, D] — MIN_MAX normalized

for h in range(1, H):
    _a = actions[:, h : h + 1]   # delta brut
    # Ici on devrait : 1) dé-normaliser last_state, 2) ajouter delta, 3) re-normaliser
    # Pour SO-101 (MIN_MAX) : new_state_raw = unnorm(state) + delta
    #                         new_state_norm = norm(new_state_raw)
    # Simplifié si on a accès aux stats min/max :
    #   state_raw = (state_norm + 1) / 2 * (max - min) + min
    #   state_raw_new = state_raw + delta
    #   state_norm_new = 2 * (state_raw_new - min) / (max - min) - 1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred_h = self.predictor(current_z, _a, last_known_state)
    current_z = pred_h[:, -tokens_per_frame:].to(torch.float32)
    if normalize_reps:
        current_z = torch.nn.functional.layer_norm(current_z, (D,))
    # TODO: update last_known_state properly if H > 1
```

**Note** : avec `mpc_horizon=1` (recommandé), cette boucle ne s'exécute jamais. Le fix est uniquement nécessaire pour des horizons > 1.

### 3.4 Image dimension ordering dans select_action (multi-frame)

**Sévérité : CRITIQUE** (pour eval multi-frame)

`forward()` sait que LeRobot délivre `[B, T, C, H, W]` et permute :
```python
# forward(), ligne 317-318
B, T, C, H, W = images.shape
images = images.permute(0, 2, 1, 3, 4)  # → [B, C, T, H, W]
```

Mais `select_action()` assume directement `[B, C, T, H, W]` sans permuter :
```python
# select_action(), ligne 172-177
if images.ndim == 5:
    img_seq = images  # ASSUME [B, C, T_obs, H, W] — FAUX si vient de LeRobot !
B_s, C, T_obs, H_img, W_img = img_seq.shape
```

Si l'eval LeRobot envoie `[B, T=4, C=3, H, W]`, le code parse `C=4, T_obs=3` → encodage incorrect.

**Fix — `modeling_vjepa_ac.py` : permuter comme dans forward()**

```python
# select_action(), après "if images.ndim == 5:", ajouter la permutation :
if images.ndim == 5:
    # LeRobot delivers [B, T, C, H, W], convert to [B, C, T, H, W]
    if images.shape[2] == 3:  # C=3 est en position 2 → format LeRobot [B,T,C,H,W]
        images = images.permute(0, 2, 1, 3, 4)
    img_seq = images  # [B, C, T_obs, H, W]
```

### 3.5 Horizon d'entraînement << horizon d'inférence

**Sévérité : CRITIQUE**

**Contexte papier** : Le DROID original entraîne sur des clips de **8 frames à 4fps (= 2 secondes)**, ce qui donne 7 deltas supervisés en teacher forcing + `auto_steps=2` pour le rollout loss. Malgré ça, le papier utilise **`mpc_horizon=1`** à l'inférence (Table 3). V-JEPA 2.1 monte à `horizon=8` (Table 6) mais avec un encodeur plus puissant.

**Notre situation** :
- **Training** : n_obs_steps=4 → 3 deltas supervisés, auto_steps=1 (pas de rollout loss)
- **Inférence** : mpc_horizon=15 → CEM demande **15 pas** de rollout autorégressif

C'est bien `mpc_horizon` (nombre de pas futurs planifiés) qui est le problème, pas `n_obs_steps` (nombre de frames passées en contexte). Ce sont deux choses indépendantes. Le plafond du mpc_horizon dépend de la capacité autorégressif du predictor, qui elle-même dépend de `auto_steps` pendant l'entraînement.

Le predictor n'a jamais été entraîné pour des rollouts longs. L'accumulation d'erreurs sur 15 pas produit des prédictions dégradées, rendant le CEM inefficace.

**Fix — `vjepa_ac.yaml` et `configuration_vjepa_ac.py`**

```yaml
# vjepa_ac.yaml — aligner avec le papier
policy:
  mpc_horizon: 1       # papier V-JEPA 2 Table 3 = 1, V-JEPA 2.1 Table 6 = 8
  auto_steps: 2         # OBLIGATOIRE : entraîner le predictor en mode autorégressif
  cem_num_samples: 800  # papier = 800 (réduire à 200 si OOM sur 16GB)
  cem_num_iters: 10     # papier = 10
```

```python
# configuration_vjepa_ac.py — changer les défauts
mpc_horizon: int = 1     # était 1, OK — mais le YAML l'override à 15, corriger le YAML
auto_steps: int = 2      # était 1, CHANGER à 2 (comme droid-256px-8f.yaml)
```

---

## 4. PROBLÈMES IMPORTANTS

### 4.1 auto_steps=1 (pas de rollout loss)

**Sévérité : HAUTE**

Le config par défaut a `auto_steps: 1`, ce qui signifie que la boucle autoregressive ne s'exécute pas (`range(1, 1)` est vide). Le sloss est alors identique à un sous-ensemble du jloss.

L'original utilise **`auto_steps: 2`** (fichier `droid-256px-8f.yaml:37`). Le rollout loss est essentiel pour que le predictor apprenne à être stable lors de prédictions autorégressives — exactement ce que fait le CEM.

**Impact** : Le predictor n'est jamais entraîné en mode autorégressif, mais l'inférence CEM est 100% autorégressif. C'est un mismatch fondamental.

**Fix — config + YAML (même fix que 3.3)**

```python
# configuration_vjepa_ac.py — changer le défaut
auto_steps: int = 2  # était 1

# vjepa_ac.yaml — ajouter explicitement
policy:
  auto_steps: 2
```

Note : avec n_obs_steps=4 (3 deltas), `auto_steps` ne peut pas dépasser `T_full - 1 = 3`. Le code le clamp déjà (`modeling_vjepa_ac.py:365`) :
```python
auto_steps = min(getattr(self.config, "auto_steps", 1), T_full - 1)
```

### 4.2 CEM dans un espace d'action uniforme vs. structuré

**Sévérité : HAUTE**

L'original optimise un espace d'action **structuré** :
- 3D position (xyz) : optimisé avec momentum 0.25
- 3D rotation (euler) : **fixée à zéro** (jamais optimisée)
- 1D gripper : optimisé avec momentum séparé 0.15

Le CEM actuel traite les 6 joints de manière **uniforme** :
- Tous avec le même std initial (0.5)
- Même momentum pour tous (0.25 / 0.95)

Pour SO-101 en joint space, les joints ont des dynamiques très différentes (base vs. poignet vs. gripper). Un CEM uniforme est sous-optimal — les joints à grande excursion dominent ceux à faible excursion.

**Fix — `modeling_vjepa_ac.py` : initialiser le std par joint depuis les stats du dataset**

```python
# Option A (simple) : ajouter un config per-joint std
# configuration_vjepa_ac.py
cem_std_per_joint: list[float] | None = None  # ex: [0.05, 0.05, 0.03, 0.02, 0.02, 0.1]

# modeling_vjepa_ac.py, dans select_action(), initialisation CEM :
if self.config.cem_std_per_joint is not None:
    std = torch.tensor(self.config.cem_std_per_joint, device=device).unsqueeze(0).expand(H, -1)
else:
    std = torch.full((H, self.config.action_dim), self.config.cem_std, device=device)
```

```python
# Option B (mieux) : calculer le std depuis les stats du dataset
# Calculer l'écart-type des deltas joints dans le dataset :
# std_per_joint = dataset_stats["action"]["std"]  # si normalisé en MEAN_STD
# Ou empiriquement sur quelques épisodes :
#   for ep in dataset: deltas = states[1:] - states[:-1]; collect std
# Utiliser ces valeurs comme cem_std_per_joint dans le config.
```

### 4.3 cem_std=0.5 inadapté

**Sévérité : HAUTE**

L'original initialise std à `maxnorm=0.05` pour la position et `1.0` pour le gripper. Le code actuel utilise `cem_std=0.5` pour tout.

Les deltas joints SO-101 sont typiquement de l'ordre de 0.01-0.05 rad/step. Un std de 0.5 signifie que l'écart-type initial du sampling est **10-50x trop grand**. Le CEM va converger très lentement ou pas du tout avec seulement 200 samples et 3 itérations.

**Fix — `configuration_vjepa_ac.py` et YAML**

```python
# configuration_vjepa_ac.py
cem_std: float = 0.03  # était 0.5 — aligner sur l'ordre de grandeur des deltas joints SO-101
```

```yaml
# vjepa_ac.yaml
policy:
  cem_std: 0.03         # ≈ std empirique des deltas joints dans le dataset
  cem_maxnorm: 0.05     # clipping absolu (déjà dans le config, mais pas utilisé — voir fix 3.2)
```

Pour calibrer précisément, lancer :
```python
# Script de calibration rapide
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("azaracla/community_dataset_v1_aggregated")
states = torch.stack([ds[i]["observation.state"] for i in range(1000)])
deltas = states[1:] - states[:-1]
print(f"Delta stats: mean={deltas.mean(0)}, std={deltas.std(0)}")
# → Utiliser le std comme cem_std, et 3*std comme cem_maxnorm
```

---

## 5. PROBLÈMES MOYENS

### 5.1 embed_dim config vs. réalité

Le config déclare `embed_dim: 1536` mais V-JEPA 2.1 ViT-Giant a en réalité `embed_dim=1408`. Pas un bug car le code utilise `encoder.embed_dim` au runtime (`modeling_vjepa_ac.py:71`), mais confusant si quelqu'un lit le config.

**Fix — `configuration_vjepa_ac.py`**

```python
embed_dim: int = 1408  # était 1536 — ViT-Giant-384 réel = 1408 (fallback only)
```

### 5.2 Masque d'attention pré-alloué trop grand

Avec mpc_horizon=15 : `max_seq_len = 16*2 = 32`, le masque fait `16*(576+2) = 9248` → matrice `[9248, 9248]` = **82 MB** en GPU, dont seule une fraction est utilisée (T=4 → 2312x2312). Gaspillage mémoire inutile.

**Fix — `modeling_vjepa_ac.py` : baser la taille du masque sur le training uniquement**

```python
# modeling_vjepa_ac.py, __init__, ligne ~68
# Le masque n'est utilisé qu'en training (T>1). Le CEM rolling (T=1) skip le masque.
# Donc baser sur n_obs_steps, pas mpc_horizon.

# AVANT :
max_temporal_depth = max(config.n_obs_steps, config.mpc_horizon + 1)

# APRÈS :
max_temporal_depth = config.n_obs_steps  # masque utilisé en training uniquement (CEM T=1 → pas de masque)
max_seq_len = max_temporal_depth * config.tubelet_size
```

Avec n_obs_steps=4 : masque `4*(576+2) = 2312` → `[2312, 2312]` = **5 MB** au lieu de 82 MB.

### 5.3 action_delta_indices charge des données inutiles

`action_delta_indices = list(range(15))` fait charger 15 actions futures du dataset, mais le preprocessor les remplace immédiatement par 3 state deltas. Gaspillage de bandwidth I/O.

**Fix — `configuration_vjepa_ac.py`**

```python
@property
def action_delta_indices(self) -> list:
    # Quand use_delta_actions=True, les actions viennent des state deltas (T-1 steps).
    # Pas besoin de charger mpc_horizon actions futures du dataset.
    if self.use_delta_actions:
        return list(range(self.n_obs_steps - 1))  # ex: n_obs=4 → [0, 1, 2]
    return list(range(self.mpc_horizon))
```

---

## 6. INTÉGRATION LEROBOT : ANALYSE

### 6.1 Ce qui est bien fait

| Aspect | Évaluation |
|---|---|
| Plugin registration via `@PreTrainedConfig.register_subclass("vjepa_ac")` | Correct |
| `get_optim_params()` retourne `self.predictor.parameters()` (encodeur frozen) | Correct |
| WSD scheduler enregistré en tant que plugin (pas dans le core) | Bonne pratique |
| Processor pipeline : pre/post avec delta-absolute conversion | Architecture solide |
| `_load_as_safetensor()` nettoie le prefix `_orig_mod.` de torch.compile | Bien anticipé |
| `action_is_pad` masking dans la loss | Bonne adaptation LeRobot |
| `observation_delta_indices` calcule correctement le sous-échantillonnage fps | Correct |

### 6.2 Écarts avec les best practices LeRobot

| Best Practice | État actuel | Impact |
|---|---|---|
| `forward()` retourne `(loss, dict)` | Implémenté | — |
| `select_action()` devrait utiliser des deques | Pas de deque (MPC replanning à chaque step) | OK pour MPC |
| `predict_action_chunk()` devrait être la méthode principale | Délègue à `select_action()` | OK pour MPC |
| Données model ← preprocessor via le batch | ✅ `RAW_DELTAS_KEY` dans la transition (batch) — `select_action()` lit `batch["observation.raw_deltas"]` | Best practice LeRobot |
| Partage d'état pre/post processor | Ref directe `preprocessor=delta_step` (co-créés dans la même factory) — le postprocessor n'a pas accès au batch d'entrée par design LeRobot, donc la ref entre steps est le pattern le plus propre possible | Acceptable, erreur explicite si ref manquante |
| `reset()` devrait réinitialiser l'état | No-op (pas d'état temporel) | OK |

---

## 7. DIFFÉRENCES DROID vs. SO-101

| Aspect | DROID (original) | SO-101 (nous) | Risque |
|---|---|---|---|
| Espace d'action | End-effector cartésien 7D | Joint space 6D | Le predictor doit apprendre une mapping plus complexe (joint->visual) |
| Deltas de rotation | Composition de matrices de rotation | Soustraction simple (pas de singularités) | Simplifié correctement |
| Gripper | Dimension séparée (index 6) | Joint 6 = servo gripper | Pas de traitement spécial nécessaire |
| Résolution | 256x256 (256 tokens/frame) | 384x384 (576 tokens/frame) | 2.25x plus de tokens, mémoire/compute plus élevés |
| Dataset | 62h, multi-tâche, table-top | Community aggregated, SO-101 | Taille et diversité probablement << DROID |

**Point fondamental** : Le predictor original apprend une correspondance `end-effector delta -> changement visuel` qui est relativement directe (translation cartésienne -> mouvement linéaire dans l'image). En joint space, la correspondance `joint delta -> changement visuel` est **non-linéaire** et dépend de la pose courante (via la cinématique directe). Le predictor a besoin de plus de données et/ou de capacité pour apprendre cette relation.

---

## 8. PARAMÈTRES D'ENTRAÎNEMENT

| Paramètre | FB DROID | Config actuelle | Scaling |
|---|---|---|---|
| GPUs | 32 x H100 | 1 x RTX 5070 Ti | — |
| Batch effectif | 256 | 4 | 64x plus petit |
| LR peak | 4.25e-4 | 1e-5 | ~6.6e-6 par scaling linéaire (1e-5 légèrement au-dessus) |
| Weight decay | 0.04 | 0.04 | Identique |
| Scheduler | WSD (4500/85500/4500) | Cosine + 4500 warmup | Différent |
| auto_steps | 2 | 1 | **Manquant** |
| Frames/clip | 8 | 4 | Moins de contexte temporel |
| Resolution | 256px | 384px | Plus de tokens |
| Total steps | 94,500 | 50,000 -> 100,000 | Comparable |

---

## 9. RECOMMANDATIONS PRIORITAIRES

### P0 — Fix pour le prochain training (reprendre depuis checkpoint actuel)

| # | Problème | Fichier(s) | Fix |
|---|---|---|---|
| 1 | auto_steps=1, pas de rollout loss (§4.1, §3.5) | `configuration_vjepa_ac.py`, `vjepa_ac.yaml` | `auto_steps: 2` |

Le training actuel (jloss teacher-forcing) est **correct**. Les poids ne sont pas perdus. Reprendre depuis le checkpoint actuel avec `auto_steps: 2` ajoutera le sloss autorégressif.

### P1 — Fix avant inférence robot

| # | Problème | Fichier(s) | Fix |
|---|---|---|---|
| 2 | hist_actions train/inférence mismatch (§3.1) | `processor_vjepa_ac.py`, `modeling_vjepa_ac.py` | Cacher `_raw_deltas` dans le preprocessor, les utiliser dans `select_action()` |
| 3 | Image dim ordering dans select_action (§3.4) | `modeling_vjepa_ac.py` | Ajouter permutation [B,T,C,H,W]→[B,C,T,H,W] comme dans forward() |
| 4 | Actions utilisées comme states dans CEM H>1 (§3.3) | `modeling_vjepa_ac.py` | Maintenir trajectoire d'état OU mettre `mpc_horizon: 1` |
| 5 | cem_maxnorm jamais appliqué (§3.2) | `modeling_vjepa_ac.py` | Ajouter `torch.clamp(actions, -maxnorm, maxnorm)` après sampling |
| 6 | cem_std=0.5 trop grand (§4.3) | `configuration_vjepa_ac.py`, YAML | `cem_std: 0.03` (calibrer sur stats dataset) |
| 7 | mpc_horizon=15 >> training (§3.5) | YAML | `mpc_horizon: 1` (comme le papier) — neutralise aussi le bug §3.3 |

### P2 — Optimisations

| # | Problème | Fichier(s) | Fix |
|---|---|---|---|
| 8 | action_delta_indices gaspille I/O (§5.3) | `configuration_vjepa_ac.py` | `return list(range(n_obs_steps - 1))` quand `use_delta_actions` |
| 9 | Masque attention 82MB (§5.2) | `modeling_vjepa_ac.py` | `max_temporal_depth = config.n_obs_steps` |
| 10 | embed_dim=1536 faux (§5.1) | `configuration_vjepa_ac.py` | `embed_dim: int = 1408` |
| 11 | CEM uniforme sur 6 joints (§4.2) | `configuration_vjepa_ac.py`, `modeling_vjepa_ac.py` | Optionnel : `cem_std_per_joint` (moins critique car SO-101 = 6 servos similaires) |

---

## 11. CORRECTIONS APPLIQUÉES (2026-04-09)

### P0 — Appliquées

| # | Problème | Fix appliqué |
|---|---|---|
| 1 | `auto_steps=1` : pas de rollout loss (§4.1/§3.5) | `configuration_vjepa_ac.py` : `auto_steps: int = 2`. `vjepa_ac.yaml` + `vjepa_ac_cloud.yaml` : `auto_steps: 2` |

### P1 — Appliquées (sauf §4.3)

| # | Problème | Fix appliqué |
|---|---|---|
| 2 | hist_actions mismatch train/inférence (§3.1) | **Refactoré (2026-04-09)** : suppression du global `_DELTA_STATE_CACHE`. Deux mécanismes distincts : (1) **raw_deltas → modèle** : `StateToDeltaActionProcessorStep` écrit `observation.raw_deltas` dans la transition/batch, `select_action()` lit `batch["observation.raw_deltas"]` (warning si absent). (2) **current_state → postprocessor** : ref directe `preprocessor=delta_step` entre steps co-créés dans `make_vjepa_ac_pre_post_processors()` — nécessaire car le postprocessor LeRobot n'a pas accès au batch d'entrée par design. `RuntimeError` explicite si ref manquante. |
| 3 | Image dim ordering dans `select_action` (§3.4) | `modeling_vjepa_ac.py` : ajout de `images.permute(0, 2, 1, 3, 4)` pour les tenseurs 5D, aligné avec `forward()` qui faisait déjà la permutation |
| 4 | Actions utilisées comme states dans CEM H>1 (§3.3) | **Contourné** : `mpc_horizon: 1` dans les deux YAMLs — la boucle `for h in range(1, H)` ne s'exécute jamais. Fix complet (dénormalisation/renormalisation des states dans la boucle) à implémenter si horizon > 1 est nécessaire |
| 5 | `cem_maxnorm` jamais appliqué (§3.2) | `modeling_vjepa_ac.py` : ajout de `actions = torch.clamp(actions, -maxnorm, maxnorm)` après le sampling dans la boucle CEM |
| 6 | `cem_std=0.5` trop grand (§4.3) | ❌ **Non fait** — calibration empirique sur le dataset recommandée avant de choisir une valeur. Script de calibration disponible en §4.3. Valeur par défaut ~0.03 suggérée |
| 7 | `mpc_horizon=15` dans les YAMLs (§3.5) | `vjepa_ac.yaml` + `vjepa_ac_cloud.yaml` : `mpc_horizon: 1` (aligne avec Table 3 du papier, neutralise aussi §3.3) |

### P2 — Appliquées (sauf §4.2)

| # | Problème | Fix appliqué |
|---|---|---|
| 8 | `action_delta_indices` charge 15 actions inutiles (§5.3) | `configuration_vjepa_ac.py` : quand `use_delta_actions=True`, retourne `list(range(n_obs_steps - 1))` au lieu de `list(range(mpc_horizon))` |
| 9 | Masque attention pré-alloué trop grand (§5.2) | `modeling_vjepa_ac.py` `__init__` : `max_temporal_depth = config.n_obs_steps` au lieu de `max(n_obs_steps, mpc_horizon + 1)`. Avec n_obs_steps=4 : masque 5MB au lieu de 82MB (ancien mpc_horizon=15) |
| 10 | `embed_dim=1536` incorrect (§5.1) | `configuration_vjepa_ac.py` : `embed_dim: int = 1408` (ViT-Giant-384 réel ; valeur de fallback uniquement, le runtime utilise `encoder.embed_dim`) |
| 11 | CEM uniforme sur 6 joints (§4.2) | ✅ **Fait (2026-04-09)** — voir §13 |

---

## 13. CALIBRATION CEM PAR JOINT (2026-04-09)

### 13.1 Analyse comparative DROID vs SO-101

#### DROID (lerobot/droid_100, 15fps, end-effector cartésien 7D)

Stats mesurées sur 1000 frames :

| Index | Interprétation | State range | State std | Delta std (15fps) | Action Meta |
|-------|---------------|-------------|-----------|-------------------|-------------|
| 0 | End-effector X (m) | [0.31, 0.68] | 0.090 | 0.0063 | Planifié, clamp ±0.05 |
| 1 | End-effector Y (m) | [-0.24, 0.23] | 0.118 | 0.0097 | Planifié, clamp ±0.05 |
| 2 | End-effector Z (m) | [0.06, 0.74] | 0.174 | 0.0163 | Planifié, clamp ±0.05 |
| 3 | Rotation wrist (rad) | **[-π, +π]** | **3.022** | **0.7356** | **Forcé à 0 dans CEM** |
| 4 | Rotation axis 2 | [-0.53, 0.76] | 0.308 | 0.0241 | **Forcé à 0 dans CEM** |
| 5 | Rotation axis 3 | [-1.59, 0.63] | 0.558 | 0.0177 | **Forcé à 0 dans CEM** |
| 6 | Gripper [0=fermé] | [0.0, 0.99] | 0.377 | 0.0241 | Clamp ±0.75, std=1.0 |

**Point clé** : Meta ne planifie que xyz + gripper. Les rotations (indices 3,4,5) sont **forcées à zéro** dans `mpc_utils.py:98-104`. Le joint 3 (std_delta=0.74) représente une rotation complète ±π — jamais planifiée par le CEM.

**Règle implicite Meta** : `maxnorm ≈ 3 × std_delta` pour les dimensions planifiées.
- xyz : std_delta ≈ 0.006-0.016 → maxnorm = 0.05 ≈ 3×0.016 ✅

**Les actions dans lerobot/droid_100 sont des cibles absolues** (pas des deltas), contrairement au CEM Meta qui travaille en deltas. Ce sont deux espaces différents.

#### SO-101 (azaracla/community_dataset_v1_aggregated, 30fps natif → 4fps effectif, joint space 6D)

Stats mesurées sur le dataset complet :

| Index | Interprétation | Delta std naïf¹ | Delta std correct² | 3×std (maxnorm) |
|-------|---------------|----------------|-------------------|-----------------|
| 0 | Shoulder Pan | 0.292 | **0.450** | 1.35 |
| 1 | Shoulder Lift | 0.919 | **1.010** | 3.03 |
| 2 | Elbow Flex | 0.851 | **0.874** | 2.62 |
| 3 | Wrist Flex | 0.273 | **0.342** | 1.03 |
| 4 | Wrist Roll | 0.487 | **0.578** | 1.74 |
| 5 | Gripper (servo) | 0.280 | **0.421** | 1.26 |

¹ Script naïf `states[1:] - states[:-1]` sur 1000 frames, traverse les frontières d'épisodes → sous-estimé ~30%
² Intra-épisode uniquement, 100 épisodes, 41878 frames (`dataset_from_index` → `dataset_to_index`)

**Différences fondamentales avec DROID :**
1. **Joint space vs end-effector** : les deltas SO-101 sont en radians/step, pas en mètres/frame. Les échelles sont incomparables.
2. **4fps effectif vs 15fps** : nos deltas couvrent ~0.267s de mouvement (8 frames à 30fps), soit 4× plus que DROID. Deltas naturellement plus grands.
3. **Gripper SO-101 = servo continu** (std=0.28, similaire aux autres joints) vs gripper DROID = commande binaire (range [-1,1], std~1.0). La logique gripper spéciale de Meta est inapplicable ici.
4. **Joints 1,2 (épaule, coude) sont 3× plus dynamiques** que les autres — le `cem_maxnorm=0.05` uniforme de Meta les clippe à ~17% de leur std, rendant le CEM aveugle sur ces dimensions.

### 13.2 Fix appliqué

**Règle** : `cem_std_per_joint = std(deltas)`, `cem_maxnorm_per_joint = 3 × std(deltas)` (même règle que Meta pour xyz).

```python
# configuration_vjepa_ac.py — nouvelles options
cem_std_per_joint: list[float] | None = None    # override cem_std si set
cem_maxnorm_per_joint: list[float] | None = None  # override cem_maxnorm si set
cem_std: float = 0.05    # corrigé : était 0.5 (bug R1), maintenant = maxnorm comme Meta
cem_maxnorm: float = 0.05

# select_action() — clamp per-joint vectorisé
if self.config.cem_maxnorm_per_joint is not None:
    maxnorm = torch.tensor(self.config.cem_maxnorm_per_joint, device=device)  # [D]
actions = torch.clamp(actions, -maxnorm, maxnorm)  # broadcast sur [N, H, D]
```

```yaml
# vjepa_ac.yaml — valeurs calibrées SO-101 (intra-épisode, 100 eps, 41878 frames)
cem_std_per_joint: [0.45, 1.01, 0.87, 0.34, 0.58, 0.42]
cem_maxnorm_per_joint: [1.35, 3.03, 2.62, 1.03, 1.74, 1.26]
```

### 13.3 Préparation des données : Meta DROID vs notre SO-101

#### Comment Meta charge les données (`vjepa2/app/vjepa_droid/droid.py`)

```python
# droid.py — chargement des states
states = cartesian_position + gripper_position  # concaténation brute, unités SI (mètres)
# Pas de normalisation. Les Franka sont calibrés en usine → ranges universels.

# Calcul des deltas au runtime (comme nous)
actions = self.poses_to_diffs(states)
# Pour xyz : simple soustraction
# Pour rotation : composition de matrices R[t+1] @ R[t].T (correct géométriquement)

# Sous-échantillonnage temporel
states = states[indices][::self.frameskip]  # fps puis tubelet_size skip
```

| Aspect | Meta DROID (`droid.py`) | Notre SO-101 |
|--------|------------------------|--------------|
| Unités states | Mètres (cartésien absolu) | Degrés (joint space, relatif à calibration) |
| Normalisation states | **Aucune** (SI universel) | **MIN_MAX** (ranges asymétriques, cross-robot) |
| Normalisation actions (deltas) | **Aucune** | **IDENTITY** (delta stats ≠ absolute stats) |
| Calcul deltas | Runtime `poses_to_diffs()` | Runtime `StateToDeltaActionProcessorStep` |
| Rotation deltas | `R[t+1] @ R[t].T` (composition matricielle) | Simple soustraction degrés (pas de rotations carte) |
| ImageNet norm | Oui (`mean/std` ImageNet) | Oui (identique) |
| Calibration cross-robot | Pas nécessaire (Franka usine) | Nécessaire (STS3215 calibrés individuellement) |

#### Pourquoi notre MIN_MAX sur les states est correct

Les moteurs STS3215 du SO-101 sont calibrés **individuellement par robot** en position relative (degrés depuis position de repos). Cela signifie :
- Les ranges absolus varient d'un robot à l'autre (ex: shoulder_pan peut aller de -90° à +90° ou de -120° à +70° selon la calibration)
- Certains joints ont des ranges très asymétriques (ex: shoulder_lift [-10°, 110°])
- La normalisation MIN_MAX homogénéise ces disparités → le `state_encoder` voit toujours [-1, 1]

Meta n'a pas ce problème : les Franka industriels ont une calibration usine, les positions cartésiennes en mètres sont directement comparables d'un robot à l'autre.

#### Pourquoi IDENTITY sur les actions (deltas) est correct

Les stats du dataset LeRobot (`stats["action"]`) correspondent aux **actions absolues** stockées dans le dataset (cibles articulaires). La normalisation MIN_MAX apprendrait à normaliser `a_k = s_{k+1} - s_k` avec les min/max de `s_k` — sémantiquement incorrect (les deltas ont une distribution centrée sur 0, pas sur la plage d'opération).

Meta n'utilise pas non plus de normalisation sur ses deltas — `poses_to_diffs()` retourne directement les valeurs brutes en mètres.

---

### 13.4 Pourquoi la normalisation est critique ici

Le predictor reçoit les actions via `action_encoder` (Linear 6→1024). Si les deltas entrent dans des échelles très différentes (0.27 pour wrist vs 0.92 pour shoulder), le gradient qui revient sur les premières dimensions de la matrice de poids est ~3× plus grand. Le réseau apprend à ignorer les joints lents et à sur-fitter les joints rapides.

Sans normalisation des actions (on utilise `IDENTITY` intentionnellement car delta stats ≠ absolute stats), c'est le CEM qui doit explorer dans le bon espace. Si le CEM utilise `cem_std=0.05` pour tous les joints, il explore seulement 17% de la dynamique réelle des joints 1,2 — ces joints ne participent presque pas à l'optimisation de la trajectoire.

Avec `cem_std_per_joint` calibré sur les stats du dataset :
- Le CEM explore proportionnellement à la dynamique de chaque joint
- Les elites sélectionnées sont représentatives de ce que le robot peut faire
- La `mu` finale converge vers une action cohérente avec la distribution des données d'entraînement

---

## 12. AUDIT DE RELECTURE (2026-04-09)

Relecture complète post-corrections, comparaison avec l'original `vjepa2/` et `mpc_utils.py`.

### 12.1 Bugs trouvés

| # | Sévérité | Fichier | Problème | Détail |
|---|----------|---------|----------|--------|
| R1 | **CRITIQUE (inférence)** | `configuration_vjepa_ac.py:46` | `cem_std=0.5` avec `cem_maxnorm=0.05` | Tous les samples CEM sont clampés à ±0.05 — le CEM dégénère en recherche uniforme aléatoire. Avec `cem_momentum_std=0.95` et seulement 3 itérations (YAML), la std ne converge jamais. L'original Meta utilise `std = maxnorm = 0.05` (`mpc_utils.py:77`). **Fix : `cem_std: 0.05`** |
| R2 | Latent | `ac_predictor_utils.py:487` | `_rescale_blocks` crash si SwiGLU activé | `rescale(layer.mlp.fc2.weight.data, ...)` — `SwiGLUFFN` utilise `.fc3` comme couche de sortie, pas `.fc2` (qui est la branche gate). Bug hérité de l'original Meta (`ac_predictor.py:134`), inactif avec GELU (config actuelle). |
| R3 | Cosmétique | `modeling_vjepa_ac.py:309+311` | Double assignation `images = batch[image_key]` | Code dupliqué, inoffensif |
| R4 | Cosmétique | `modeling_vjepa_ac.py:385,399` | Code `extrinsics` mort | `extrinsics = None` toujours, le code de slicing est inatteignable |

### 12.2 Différences vérifiées (pas des bugs)

| Aspect | Notre implémentation | Original Meta | Verdict |
|--------|---------------------|---------------|---------|
| States normalisés | MIN_MAX [-1, 1] via preprocessor | Raw (pas de normalisation) | Choix valide — `state_encoder` apprend sur des states normalisés, cohérent train/infer. Aide pour joint space SO-101 avec ranges non-uniformes |
| CEM momentum | `momentum_mean=0.25, momentum_std=0.95` | Identique | ✅ |
| CEM structure actions | Uniform 6D (joints SO-101) | Structuré xyz(3D) + gripper(1D) séparés, rotation zéro | Adaptation correcte au joint space |
| Formule loss | `torch.abs(z - h) ** loss_exp / loss_exp` | `torch.mean(torch.abs(z - h) ** loss_exp) / loss_exp` | Identique (notre code fait `.mean()` au retour de `loss_fn`) |
| Auto-regressive bounds | `auto_steps=2`, boucle `range(1, 2)` avec T=4 | `auto_steps=2`, boucle `range(1, 2)` avec T=8 | ✅ Même logique, adapté à notre T |

### 12.3 Refactor appliqué : suppression `_DELTA_STATE_CACHE`

Le global `_DELTA_STATE_CACHE` (anti-pattern : état mutable partagé entre pipelines) a été supprimé et remplacé par deux mécanismes :

1. **raw_deltas → modèle** : `StateToDeltaActionProcessorStep` écrit `observation.raw_deltas` dans la transition. `select_action()` lit `batch["observation.raw_deltas"]`. Best practice LeRobot : tout passe par le batch.

2. **current_state → postprocessor** : Ref directe `preprocessor=delta_step` entre steps co-créés dans `make_vjepa_ac_pre_post_processors()`. Nécessaire car le postprocessor LeRobot ne reçoit qu'un `PolicyAction` tensor (pas le batch d'entrée). `RuntimeError` explicite si ref manquante.

---

## 10. RÉSUMÉ

L'intégration architecturale dans LeRobot est **solide** — le predictor est un portage fidèle, le pipeline processor est bien conçu, et les abstractions LeRobot sont correctement utilisées. Les problèmes identifiés sont principalement au niveau de **la cohérence entre entraînement et inférence** (action space mismatch, horizon mismatch, auto_steps) et des **hyperparamètres CEM inadaptés au joint space** SO-101. Ce sont des problèmes corrigeables sans refonte architecturale.

**Post-relecture (2026-04-09)** : Toutes les corrections P0/P1 sont validées. Le seul bug critique restant est `cem_std=0.5` (R1) qui rend le CEM dégénéré à l'inférence — fix trivial (`cem_std: 0.05`). L'entraînement n'est pas affecté par ce paramètre.

---

## 14. REFACTORING PROCESSOR PIPELINE (2026-04-09)

### 14.1 Pourquoi delta actions et non relative actions (LeRobot PR #2970)

LeRobot PR #2970 introduit `RelativeActionsProcessorStep` (terminologie UMI) avec la sémantique :
```
action[k] = target_pos[k] - s_0   # offset depuis l'état courant, même référence pour tout le chunk
```

VJEPA-AC utilise une sémantique **différente** (convention DROID, section 3.1 du papier) :
```
a_k = s_{k+1} - s_k   # diff séquentielle entre frames adjacentes
```

**Pourquoi cette différence est architecturale** : le predictor VJEPA-AC est autorégressif — il reçoit `(a_k, s_k, z_k)` et prédit `z_{k+1}`. Le token d'action doit encoder la **transition locale** entre l'état courant et le suivant. Avec des relative actions, `a_k` encoderait le déplacement cumulatif depuis `s_0` — redondant avec les `s_k` déjà dans le contexte et pas directement utile pour prédire le prochain frame.

Les relative actions conviennent aux policies qui prédisent un chunk entier depuis un état fixe (UMI, π₀). Les delta actions conviennent aux predictors frame-à-frame autorégressifs (VJEPA-AC).

On ne peut donc **pas** utiliser `RelativeActionsProcessorStep` directement — les steps calculent et invertissent avec la mauvaise sémantique.

### 14.2 Limitation architecturale du postprocessor LeRobot

Le postprocessor LeRobot ne reçoit qu'un `PolicyAction` tensor (pas le batch d'entrée complet). Pour convertir delta→absolu en postprocessing, il faut accéder au `current_state` capturé par le preprocessor — d'où le couplage entre steps.

Anti-patterns évalués et rejetés :
- `_DELTA_STATE_CACHE` global → état mutable partagé (section 12.3)
- `state_cache = {}` closure → même anti-pattern sous un autre nom

### 14.3 Solution retenue : pattern LeRobot (_reconnect_vjepa_ac_steps)

Le pattern officiel LeRobot (PR #2970) utilise une ref directe entre steps, re-câblée après désérialisation dans `factory.py`. On suit exactement ce pattern pour les steps VJEPA-AC.

**Pipeline final** :
```
Rename(0) → AddBatch(1) → StateToDeltaActionProcessorStep(2) → DeviceProcessorStep(3) → NormalizerProcessorStep(4)
```

`StateToDeltaActionProcessorStep` s'exécute **avant** `DeviceProcessorStep` (comme upstream) — `_current_state` est caché sur CPU, `DeltaToAbsoluteActionProcessorStep` gère la conversion device/dtype au moment de l'utilisation.

**Fichiers modifiés** :

| Fichier | Changement |
|---|---|
| `processor_vjepa_ac.py` | `VjepaAcDeviceAndDeltaStep` → `StateToDeltaActionProcessorStep` (delta seul, sans device). Ajout `RAW_CURRENT_STATE_KEY`. Fix device mismatch dans `DeltaToAbsoluteActionProcessorStep`. |
| `src/lerobot/policies/factory.py` | Ajout `_reconnect_vjepa_ac_steps()`, appelé après `_reconnect_relative_absolute_steps` dans `make_pre_post_processors()`. |
| `src/lerobot/async_inference/policy_server.py` | Suppression `_load_dataset_stats()` — les stats sont chargées automatiquement via `state_file` dans le JSON du pipeline (comportement upstream). |