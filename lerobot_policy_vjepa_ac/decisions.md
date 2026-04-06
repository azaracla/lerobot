# Décisions

## 2026-04-06 : Passage aux positions absolues pour l'inférence CEM

### Problème
Le code CEM d'inférence traitait les actions comme des deltas (clipping `maxnorm=0.05`, intégration `state += action`), mais le training utilisait des positions absolues de joints (6D) du dataset SO-101. Incohérence documentée dans `TECHNICAL_REPORT_frames_timestamps.md`.

### Décision
- **CEM sample des positions absolues** : `mu = init_state.clone().expand(H, -1)` au lieu de `mu = zeros(H, action_dim)`
- **Suppression du clipping maxnorm** sur xyz (pertinent uniquement pour des deltas)
- **Clipping gripper** : `[0.0, 1.0]` au lieu de `[-0.75, 0.75]`
- **State integration corrigée** : `_s_seq = actions` (les actions SONT les states) au lieu de `_temp_s = _temp_s + actions`

### Raison
Le SO-100/101 attend des positions absolues de joints. Le dataset d'entraînement `azaracla/so101_pickup` fournit des positions absolues. Il est donc cohérent que le CEM produise directement des positions absolues.

## 2026-04-06 : Goal image pré-encodée au __init__

### Décision
- Ajout de `goal_image_path: str | None = None` dans `VjepaAcConfig`
- Le goal image est chargé et encodé une seule fois au `__init__` de la policy
- Le `goal_latent` est stocké et réutilisé à chaque appel de `select_action`
- Fallback à zéros si aucun goal_image_path n'est fourni

### Raison
Plus simple que de passer le goal image via le batch lerobot. Le goal image est statique pour une tâche donnée, donc l'encoder une fois est efficace.

## 2026-04-06 : Intégration inférence asynchrone gRPC

### Décision
- Ajout de `vjepa_ac` à `SUPPORTED_POLICIES` dans `async_inference/constants.py`
- Ajout de `VjepaAcConfig` aux imports dans `async_inference/helpers.py`
- Ajout de `so_follower` à `SUPPORTED_ROBOTS`
- Aucune modification du code server/client nécessaire : `predict_action_chunk` délègue déjà à `select_action`
- Le `goal_image_path` est chargé automatiquement via `from_pretrained()` depuis la config sauvegardée

### Raison
Le module `async_inference/` de lerobot existe déjà et est complet (gRPC, action chunking, queue management, must-go mechanism). VJEPA-AC s'intègre naturellement car il implémente déjà `predict_action_chunk` correctement.

## 2026-04-06 : Normalisation du state dans select_action

### Problème
Le CEM utilisait le state brut (positions absolues du robot) pour initialiser `mu`, alors que l'entraînement utilisait des states normalisés. Le predictor a été entraîné avec:
- States normalisés (MIN_MAX)
- Actions normalisées (MIN_MAX)

Mais l'inférence passait des states absolus → mismatch de distribution.

### Décision
- Stocker `dataset_stats` dans la policy (`self.dataset_stats = dataset_stats`)
- Ajouter méthode `_normalize_state()` qui applique MIN_MAX normalization
- Appeler `_normalize_state(init_state)` avant d'initialiser `mu` dans la boucle CEM
- Policy outputs normalized actions → postprocessor denormalizes them

### Implémentation
1. `__init__`: `self.dataset_stats = dataset_stats`
2. `_normalize_state()`: convertit [min,max] → [-1,1] selon les stats
3. `select_action`: `init_state = self._normalize_state(init_state)`
4. `policy_server.py`: `_load_dataset_stats()` charge les stats depuis le fichier safetensors

### Raison
Le predictor a appris à prédire les latents à partir de states/actions normalisés. Pour que les prédictions soient valides, les entrées doivent être dans la même distribution (normalisée).
