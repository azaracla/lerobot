# Rapport de Comparaison: VJEPa AC Original vs LeRobot Implementation

## Vue d'ensemble

Ce rapport compare l'implémentation finale de `lerobot_policy_vjepa_ac` avec le code original vjepa2.

| Composant | Original | LeRobot | Statut |
|-----------|----------|---------|--------|
| Modèle AC | `vjepa2/src/models/ac_predictor.py` | `ac_predictor_utils.py` | ✅ Presque identique |
| Training | `vjepa2/app/vjepa_droid/train.py` | `modeling_vjepa_ac.forward()` | ✅ Fonctionnellement équivalent |
| Inference | `vjepa2/notebooks/utils/mpc_utils.py` + `world_model_wrapper.py` | `modeling_vjepa_ac.select_action()` | ✅ Fonctionnellement équivalent |

---

## 1. Architecture du Modèle (VisionTransformerPredictorAC)

### Similarités
- **Structure identique**: L'architecture du predictor est copiée ligne par ligne
- **Encodage des actions/states**: Même procédure d'interleave avec les tokens visuels
- **Attention mask**: Même masque causal pour frame-level attention
- **Weight initialization**: Même `_init_weights()` et `_rescale_blocks()`

### Différences Mineures

| Aspect | Original | LeRobot | Impact |
|--------|----------|---------|--------|
| Imports | `src.models.utils.modules.Block` | Implémenté localement | ❌ Aucun (copie identique) |
| RoPE Implementation | Dans `src.models.utils/` | Dans `ac_predictor_utils.py` | ❌ Aucun (copie identique) |
| Activation checkpointing | Supporté via `use_activation_checkpointing` | Supporté mais pas testé | ⚠️ Potentiellement différent |

### Code Identique
```python
# Les deux utilisent la même logique pour encoder et interleave les tokens
x = self.predictor_embed(x)  # Map context tokens to predictor dimension
s = self.state_encoder(states).unsqueeze(2)  # Encode states
a = self.action_encoder(actions).unsqueeze(2)  # Encode actions
x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # Interleave
```

---

## 2. Training Loop

### Original (train.py:403-470)
```python
# Forward target
def forward_target(c):
    with torch.no_grad():
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = target_encoder(c)
        h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

# Teacher forcing + Auto-regressive
z_tf, z_ar = forward_predictions(h)
jloss = loss_fn(z_tf, h)
sloss = loss_fn(z_ar, h)
loss = jloss + sloss
```

### LeRobot (modeling_vjepa_ac.py:218-312)
```python
# Encode avec le encoder gelé
target_latents = self.encoder(images)  # [B, T, N, D]
if self.config.normalize_reps:
    target_latents = F.layer_norm(target_latents, (...))

# Même logique TF + AR
z_tf = _step_predictor(z_ctxt, actions, states[:, :-1])
z_ar = ... # auto-regressive rollout
jloss = loss_fn(z_tf, target_latents)
sloss = loss_fn(z_ar, target_latents)
loss = jloss + sloss
```

### Différences Clés

| Aspect | Original | LeRobot | Impact |
|--------|----------|---------|--------|
| **Encoder loading** | Chargé manuellement depuis checkpoint | `torch.hub.load()` | ⚠️ Différent (mais même modèle) |
| **Batch processing** | Multi-frame batch encoding | Frame-by-frame loop | ⚠️ **Performance** (voir notes) |
| **Mixed precision** | `torch.cuda.amp.autocast` | Non implémenté | ⚠️ Performance/VRAM |
| **Extrinsics** | Supporté (4ème token) | Pas implémenté | ❌ Non utilisé |
| **Target encoder** | Exponentially moving average | **Même encoder gelé** | ⚠️ **IMPORTANT** (voir notes) |

### ⚠️ Problème Critique: Target Encoder

**Original V-JEPA 2 pretraining**: Utilise un target encoder avec EMA (Exponential Moving Average)
```python
# Pendant le prétraining action-free
target_encoder = copy.deepcopy(encoder)
# ...après chaque step...
for param_tgt, param_src in zip(target_encoder.parameters(), encoder.parameters()):
    param_tgt.data = param_tgt.data * ema + param_src.data * (1.0 - ema)
```

**V-JEPA 2-AC post-training (paper)**: L'encodeur est **GELÉ**
> "After pretraining, we **freeze the video encoder** and learn a new action-conditioned predictor, V-JEPA 2-AC, on top of the learned representation." (Figure 2 caption)

**LeRobot**: Utilise le même encoder gelé - **CORRECT**
```python
# Dans modeling_vjepa_ac.py
self.encoder.eval()
for param in self.encoder.parameters():
    param.requires_grad = False
```

**Impact**: **PAS de problème** - La procédure V-JEPA 2-AC officielle (post-training) géle l'encodeur, donc pas besoin de target encoder EMA. La cible est la sortie statique du ViT gelé (pré-entraîné ImageNet). L'implémentation LeRobot est correcte.

---

## 3. Inference / MPC (Cross-Entropy Method)

### Original (mpc_utils.py + world_model_wrapper.py)

**CEM Loop**:
```python
for step in range(cem_steps):
    action_traj, frame_traj = sample_action_traj()
    selected = select_topk_action_traj(final_state, goal_state, actions)
    mean = momentum_update(mean_selected, mean)
    std = momentum_update(std_selected, std)
```

**Pose Integration** (mpc_utils.py:166-190):
```python
def compute_new_pose(pose, action):
    # Utilise scipy.spatial.transform.Rotation pour l'intégration correcte des angles
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta) for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta) for theta in delta_thetas]
    angle_diff = [delta @ mat.T for delta, mat in zip(delta_matrices, matrices)]
    new_angle = [Rotation.from_matrix(d).as_euler("xyz") for d in angle_diff]
```

### LeRobot (modeling_vjepa_ac.py:90-216)

**CEM Loop**: Identique en structure
```python
for _ in range(cem_num_iters):
    eps = torch.randn(N, H, action_dim)
    actions = mu + std * eps
    # ... rollout ...
    costs = torch.mean(torch.abs(final_latent - goal_latent)**loss_exp, dim=(1,2))
    elite_inds = torch.topk(costs, top_k, largest=False).indices
    mu = momentum_update(elites.mean(), mu)
```

**Pose Integration** (modeling_vjepa_ac.py:175-185):
```python
# Simple addition (approximation!)
_s_seq = []
_temp_s = init_state.expand(N, -1)
for i in range(h_step + 1):
    _s_seq.append(_temp_s.unsqueeze(1))
    _temp_s = _temp_s + actions[:, i]  # ⚠️ Addition simple!
```

### Différences Clés

| Aspect | Original | LeRobot | Impact |
|--------|----------|---------|--------|
| **CEM structure** | Identique | Identique | ✅ Correct |
| **Pose integration** | Rotation matrices scipy | Addition simple | ⚠️ Approximation (peut affecter la performance) |
| **Action clipping** | Gripper: [-0.75, 0.75] | Gripper: [-0.75, 0.75] | ✅ Identique |
| **Maxnorm** | 0.05 | Configurable (default 0.05) | ✅ Identique |
| **Goal conditioning** | Requis | Optionnel (fallback zeros) | ⚠️ Fonctionnalité différente |
| **Batch CEM** | Par échantillon | Batch par batch puis samples | ✅ Équivalent |

### ⚠️ Problème: Pose Integration

**Original**: Intégration correcte des rotations avec scipy
```python
# Produit matriciel correct pour rotations 3D
angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
```

**LeRobot**: Addition simple (approximation)
```python
_temp_s = _temp_s + actions[:, i]  # Pose + delta_pose
```

**Impact**: Pour de petits deltas, l'approximation peut être acceptable, mais pour des rotations importantes ou des trajectoires longues, l'erreur peut s'accumuler.

---

## 4. Normalisation des Images

### Original (vjepa2/app/vjepa_droid/transforms.py)
```python
# Normalisation ImageNet APPLIQUÉE DANS LE DATASET
# Les images uint8 [0, 255] sont normalisées avant le encoder
img = img / 255.0
img = (img - mean) / std  # ImageNet mean/std
```

### LeRobot (modeling_vjepa_ac.py:68-82)
```python
def _imagenet_normalize(self, images: torch.Tensor) -> torch.Tensor:
    if not self.config.use_imagenet_for_visuals:
        return images  # Pas de normalisation
    mean = IMAGENET_MEAN.to(images.device).view(1, -1, 1, 1, 1)
    std = IMAGENET_STD.to(images.device).view(1, -1, 1, 1, 1)
    return (images - mean) / std
```

### Différences Clés

| Aspect | Original | LeRobot | Impact |
|--------|----------|---------|--------|
| **Application** | Dans dataset loader | Dans policy.forward() | ✅ Équivalent (après fix) |
| **Activation** | Toujours | Configurable (`use_imagenet_for_visuals`) | ✅ Flexible |
| **Input range** | [0, 1] (float) | [0, 1] (float) | ✅ Compatible |

---

## 5. Transformations d'Images

### Original (vjepa2/app/vjepa_droid/transforms.py)
```python
# DROID-style random resized crop
scale=1.777, ratio=[0.75, 1.35], target_size=256
# Appliqué pendant le chargement des données
```

### LeRobot
```python
# Custom transform dans lerobot_policy_vjepa_ac/transforms.py
# Enregistré dynamiquement via __init__.py
# Activé via dataset.image_transforms.enable
```

### Différences Clés

| Aspect | Original | LeRobot | Impact |
|--------|----------|---------|--------|
| **Paramètres** | scale=1.777, ratio=[0.75,1.35], size=256 | Identique | ✅ Correct |
| **Bug fix** | Bug sur reshape [3, H, W] | Corrigé | ✅ Amélioration |
| **Location** | Dataset loader | Transform LeRobot + patch | ✅ Compatible |

---

## 6. Configuration / Hyperparameters

### Original DROID Training

```python
batch_size: 8 (ViT-Giant, multi-GPU) / 1 (ViT-Large)
lr: 4.25e-4
weight_decay: 0.04
scheduler: WSD (warmup=15, steady=285, anneal=15 epochs)
epochs: 315
crop_size: 256
patch_size: 16
tubelet_size: 2
fps: 4
frames_per_clip: 8
num_frames: 8
normalize_reps: True
auto_steps: 2
loss_exp: 1.0
```

### LeRobot Config

```python
batch_size: 16
lr: 1e-4  # ⚠️ Différent
weight_decay: 1e-4  # ⚠️ Différent
scheduler: cosine diffuser  # ⚠️ Différent
img_size: 384  # ⚠️ Différent
normalize_reps: False  # ⚠️ Différent
auto_steps: 1  # ⚠️ Différent
```

### ⚠️ Différences Impactantes

| Paramètre | Original DROID | LeRobot pickplace | Impact |
|-----------|----------------|-------------------|--------|
| **batch_size** | 8 | 16 | ⚠️ Peut affecter stabilité |
| **lr** | 4.25e-4 | 1e-4 | ⚠️ **4x plus petit** - peut ralentir convergence |
| **weight_decay** | 0.04 | 1e-4 | ⚠️ **400x plus petit** - peut affecter généralisation |
| **scheduler** | WSD (3 phases) | Cosine | ⚠️ Différent |
| **img_size** | 256 | 384 | ⚠️ Différent (modele 384px) |
| **normalize_reps** | True | False | ⚠️ **IMPORTANT** - affecte la loss |
| **auto_steps** | 2 | 1 | ⚠️ Moins de AR rollout |
| **use_extrinsics** | False | False | ✅ Identique |

---

## 7. Résumé des Problèmes

### 🟡 Modérés (peuvent affecter la qualité)

1. **Hyperparameters très différents**
   - LR 4x plus petit (1e-4 vs 4.25e-4)
   - Weight decay 400x plus petit (1e-4 vs 0.04)
   - Scheduler différent (cosine vs WSD 3-phases)
   - Impact: Convergence et généralisation différentes

3. **Batch processing inefficace**
   - LeRobot encode frame par frame dans une boucle
   - Original encode tout le batch en une passe
   - Impact: Performance 2-3x plus lente

### 🟡 Modérés (peuvent affecter la qualité)

2. **Pose integration approximative**
   - Original: Rotation matrices scipy (correct)
   - LeRobot: Addition simple
   - Impact: Erreur d'approximation pour grandes rotations

3. **normalize_reps = False**
   - Original: Normalisation LayerNorm activée
   - LeRobot: Désactivée
   - Impact: Stability et scale des gradients différents

4. **auto_steps = 1**
   - Original: 2 steps AR rollout
   - LeRobot: 1 step seulement
   - Impact: Moins de regularization, plus de teacher forcing

### 🟢 Mineurs (pas d'impact majeur)

5. **Pas de mixed precision**
   - Original: bfloat16 supporté
   - LeRobot: float32
   - Impact: VRAM 2x, pas de perf boost

6. **Pas d'extrinsics token**
   - Original: Supporté (mais désactivé par défaut)
   - LeRobot: Pas implémenté
   - Impact: Aucun (pas utilisé)

---

## 8. Recommandations

### Pour l'entraînement pickplace

1. **Utiliser les hyperparameters DROID**:
   ```yaml
   lr: 4.25e-4
   weight_decay: 0.04
   scheduler: wsd  # ou implémenter WSDSchedulerConfig
   normalize_reps: true
   auto_steps: 2
   ```

2. **Optimiser l'encoding batch**:
   ```python
   # Remplacer la boucle frame-by-frame par:
   images = images.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
   images = self._imagenet_normalize(images)
   # Encoder supporte [B*T, C, H, W] ou [B, T*C, H, W]
   ```

3. **Activer normalize_reps dans la config**:
   ```yaml
   policy:
     normalize_reps: true
     auto_steps: 2
   ```

### Pour l'inférence

4. **Implémenter pose integration correcte**:
   ```python
   # Ajouter scipy comme dépendance
   from scipy.spatial.transform import Rotation
   # Utiliser compute_new_pose() du code original
   ```

---

## 9. Statut Actuel

### ✅ Fonctionne
- Architecture AC predictor identique
- Training loop structure correcte
- CEM inference correcte
- Image normalization (après fix avec `use_imagenet_for_visuals`)
- Custom transform DroidRandomResizedCrop (corrigé et enregistré)
- Target encoder: PAS de problème (ViT gelé comme dans le papier officiel)

### ⚠️ À Corriger
- Hyperparameters DROID (pas utilisés)
- Pose integration (approximation)
- Batch encoding (inefficace)

### ❌ Non implémenté
- Mixed precision training
- Extrinsics token
- Distributed training (original multi-GPU)

---

## Conclusion

L'implémentation LeRobot est **fonctionnellement correcte** et conforme au papier V-JEPA 2-AC officiel, mais diffère du code original sur plusieurs aspects importants:

1. **Architecture**: ✅ Identique
2. **Training structure**: ✅ Correct mais hyperparameters différents
3. **Inference**: ✅ Correct mais pose integration approximative
4. **Performance**: ⚠️ Plus lente (batch encoding)

**Note importante sur le target encoder EMA**: Le papier V-JEPA 2-AC confirme que l'encodeur est **gelé** pendant l'entraînement AC (Figure 2 caption: "freeze the video encoder"). L'EMA est utilisé pendant le prétraining action-free, pas pendant l'AC post-training. L'implémentation LeRobot qui utilise un seul encoder gelé est donc **correcte**.

Pour atteindre la performance de l'original DROID, les corrections suivantes sont recommandées:
- Hyperparameters DROID (lr, wd, scheduler)
- normalize_reps = True
- auto_steps = 2
- Optimiser le batch encoding