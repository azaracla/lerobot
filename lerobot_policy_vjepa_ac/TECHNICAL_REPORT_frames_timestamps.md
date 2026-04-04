# Rapport technique : Configuration des frames et timestamps

> Analyse des divergences de configuration entre `vjepa2` original et `lerobot_policy_vjepa_ac`.
> Date : 2026-04-04
> Statut : **En attente de validation**

---

## 1. Problème identifié : Tubelet size mismatch

### Symptôme potentiel

L'encodeur VJEPA2.1 chargé via `torch.hub` est configuré avec `tubelet_size=2` (64 frames), tandis que LeRobot passe typiquement **1 frame** par observation avec `tubelet_size=1`.

### Configuration dans `vjepa2/src/hub/backbones.py`

```python
# _make_vjepa2_1_model()
num_frames=64,
tubelet_size=2,  # ← tubelet de 2 frames
```

### Configuration dans LeRobot (`vjepa_ac.yaml`)

```yaml
policy:
  num_frames: 1      # ← 1 frame
  tubelet_size: 1    # ← tubelet de 1
```

### Code impacté (`modeling_vjepa_ac.py:86-91`)

```python
def select_action(self, batch, **kwargs):
    images = batch[image_key]  # [B, C, T, H, W]
    
    if images.ndim == 5:
        img_seq = images[:, :, -1:]  # ← Prend SEULEMENT la dernière frame
    else:
        img_seq = images.unsqueeze(2)
    
    current_latent = self.encoder(img_seq)  # ← Passe 1 frame à un modèle configuré pour 64
```

### Impact dans le VisionTransformer (`vision_transformer.py:175-177`)

```python
elif x.ndim == 5:
    _, _, T, H, W = x.shape
    T = T // self.tubelet_size  # ← Si T=1 et tubelet_size=2 → T=0 ou T=0.5
```

### Impact dans l'attention RoPE (`modules.py:343-345`)

```python
if T is None or H_patches is None or W_patches is None:
    mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
else:
    mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
# Si T=0 → masque vide ou erreur de dimension
```

---

## 2. Problème identifié : Intégration des poses

### VJEPA original (`mpc_utils.py:166-190`)

```python
def compute_new_pose(pose, action):
    # Utilise des matrices de rotation pour les angles Euler
    matrices = [Rotation.from_euler("xyz", theta) for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", delta) for delta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz") for mat in angle_diff]
```

### LeRobot (`modeling_vjepa_ac.py:153-158`)

```python
for i in range(h_step + 1):
    _s_seq.append(_temp_s.unsqueeze(1))
    _temp_s = _temp_s + actions[:, i]  # ← Addition simple, perte de précision
```

**Impact** : Les rotations sont intégrées comme vecteurs simples au lieu d'utiliser l'algèbre des groupes SO(3). Cela peut causer une drift sur les orientations sur de longues trajectoires.

---

## 3. Dimension des actions

| Aspect | VJEPA original | LeRobot (yaml) | LeRobot (smoke_test) |
|--------|---------------|-----------------|----------------------|
| `action_dim` | 7 | **6** | **7** |

Le YAML indique `action_dim: 6` (sans gripper), mais `smoke_test.py` utilise `action_dim=7`.

---

## 4. Résumé des divergences

| Aspect | VJEPA original | LeRobot | Risque |
|--------|-----------------|---------|--------|
| `tubelet_size` | 2 (64 frames) | 1 | **Élevé** — incompatibilité de dimensions |
| `num_frames` encoder | 64 | 1 | **Élevé** |
| Intégration pose | Matrices de rotation | Addition simple | **Moyen** — drift angulaire |
| `action_dim` | 7 | 6 ou 7 | **Faible** — dépend du dataset |
| Normalisation | Aucune | MIN_MAX | **Faible** — configuration dataset |

---

## 5. Recommandations

### Option A : Adapter LeRobot au modèle chargé

Charger le modèle avec les bons paramètres via `torch.hub.load_state_dict_from_url` personnalisé, ou utiliser `torch.hub.load(..., **{'num_frames': 1, 'tubelet_size': 1})`.

### Option B : Adapter l'inférence au modèle

Répéter la frame d'observation 64 fois (ou 16x avec tubelet_size=1) pour matcher la configuration de l'encodeur :

```python
# Dans select_action()
img_seq = images[:, :, -1:]  # [B, C, 1, H, W]
img_seq = img_seq.repeat(1, 1, 64, 1, 1)  # → [B, C, 64, H, W]
current_latent = self.encoder(img_seq)
```

### Option C : Réentraîner avec la configuration cible

Entraîner le predictor AC avec `tubelet_size=1` et `num_frames=1` pour matcher les observations LeRobot.

---

## 6. Actions à vérifier

- [ ] Tester `select_action()` avec des observations réelles
- [ ] Vérifier les dimensions des tenseurs dans le forward pass
- [ ] Confirmer que `torch.hub.load` accepte les kwargs de configuration
- [ ] Valider que le modèle fonctionne malgré les dimensions potentiellement incorrectes

---

## 7. Différences fondamentales Droid → SO-101

Cette section documente les différences de domaine entre le dataset d'entraînement original (Droid) et les datasets LeRobot (ex: SO-101).

### Comparaison des espaces

| Aspect | Droid (papier VJEPA2) | SO-101 (LeRobot) |
|--------|------------------------|------------------|
| **Robot** | Franka Panda | SO-101 |
| **DoF** | 7 | 6 |
| **Action** | Delta end-effector (7D) | Position joints (6D) |
| **State** | Position + orientation + gripper (7D) | Angles joints (6D) |
| **Format** | Droid HDF5 custom | HuggingFace/LeRobot |
| **Caméra** | Exocentrique gauche fixe | Variable selon dataset |

### Détail des espaces

**Droid (VJEPA2 original)**
```
Action (7D): [dx, dy, dz, dθx, dθy, dθz, dgripper]
State  (7D): [x, y, z, θx, θy, θz, gripper]
```
- Contrôle en coordonnées cartésiennes de l'effecteur
- Actions = delta à appliquer à la pose courante

**SO-101 (LeRobot)**
```
Action (6D): [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
State  (6D): [angle_1, angle_2, angle_3, angle_4, angle_5, angle_6]
```
- Contrôle en positions articulaires
- Actions = positions absolues des joints

### Implications pour l'adaptation

1. **Dimension mismatch** : Le predictor AC a été entraîné avec `action_dim=7` (Droid), mais LeRobot utilise `action_dim=6` (SO-101).

2. **Sémantique différente** : 
   - Droid prédit des **deltas** à appliquer à l'effecteur
   - SO-101 prédit des **positions absolues** de joints

3. **Intégration d'état** : Le code actuel (`_temp_s + actions`) est correct pour des deltas, mais pour des positions absolues, il faudrait prédire `state_next = action` directement.

4. **Normalisation** : Les stats de normalisation doivent être recalculées pour le nouveau dataset.

### Impact sur CEM

Le coût CEM (`mean(abs(final_latent - goal_latent))`) ne dépend pas directement de l'espace d'action, mais :
- Le **predictor** a appris des dynamiques Droid (cartésiennes)
- Les **rollouts** seront moins précis sur SO-101
- L'**intégration d'état** actuelle suppose des deltas

---

## 8. Actions recommandées pour adapter à SO-101

1. **Réentraîner le predictor** avec `action_dim=6` sur données SO-101
2. **Changer le mode d'intégration** : `action = target_state` au lieu de `state += action`
3. **Recalculer les stats de normalisation** pour le nouveau dataset
4. **Vérifier la caméra** : les observations doivent correspondre au domaine d'entraînement

---

## 9. Fichiers à modifier

| Fichier | Modification |
|---------|--------------|
| `modeling_vjepa_ac.py` | Ajouter répétition de frames dans `select_action()` |
| `modeling_vjepa_ac.py` | Adapter intégration d'état (deltas vs positions absolues) |
| `configs/policy/vjepa_ac.yaml` | Clarifier `action_dim` (6 ou 7) selon dataset cible |
| `smoke_test.py` | Utiliser la même config que le YAML |
