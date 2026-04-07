# Rapport : Qualité d'intégration de VJEPA-AC avec LeRobot

## Contexte

LeRobot (v0.4.0+) définit un contrat d'intégration clair pour les policies via :
- Une classe de base `PreTrainedPolicy` (`src/lerobot/policies/pretrained.py`)
- Un système de processors composables (`src/lerobot/processor/`)
- Un factory de policies (`src/lerobot/policies/factory.py`)

Ce rapport évalue comment `VjepaAcPolicy` respecte ce contrat, et identifie les écarts et risques.

---

## 1. Contrat d'intégration LeRobot

### Interface `PreTrainedPolicy` (méthodes abstraites requises)

| Méthode | Description |
|---|---|
| `get_optim_params()` | Paramètres à optimiser |
| `reset()` | Remet à zéro les caches internes |
| `forward(batch)` | Calcul de la loss (training) |
| `select_action(batch)` | Inférence : retourne une action |
| `predict_action_chunk(batch)` | Inférence : retourne une séquence d'actions |

### Pipeline Processor standard

```
Préprocessing  : Rename → AddBatchDim → [Policy-specific] → Device → Normalize
Postprocessing : Unnormalize → Device(cpu)
```

### Enregistrement factory

Les policies internes sont hardcodées dans `get_policy_class()` et `make_policy_config()`. Les policies externes passent par `_get_policy_cls_from_policy_name()` (fallback dynamique).

---

## 2. État de l'intégration VJEPA-AC

### ✅ Points conformes

**Interface de base implémentée correctement**
- `get_optim_params()` : retourne `self.predictor.parameters()` — seul le predictor est entraînable, l'encoder est frozen. C'est correct.
- `reset()` : implémenté (no-op, acceptable car pas d'action chunking).
- `select_action()` : implémenté, fait le CEM complet.
- `predict_action_chunk()` : délègue à `select_action()` — fonctionnel mais voir §3.

**Processor pipeline conforme au standard**
`make_vjepa_ac_pre_post_processors()` suit exactement le pattern des autres policies :
```python
Rename → AddBatchDim → Device → Normalize   # pré
Unnormalize → Device(cpu)                    # post
```
`VjepaAcLoggingProcessorStep` est correctement enregistré via `@ProcessorStepRegistry.register`.

**Package externe correctement structuré**
- Séparation `modeling_` / `configuration_` / `processor_` respectée.
- `config_class = VjepaAcConfig` et `name = "vjepa_ac"` définis.

**Gestion du torch.compile**
`_load_as_safetensor` override correctement pour nettoyer les clés `_orig_mod.` générées par `torch.compile`. Bonne pratique.

---

### ⚠️ Problèmes et écarts

#### 1. `forward()` non implémenté
**Sévérité : critique pour le training**

`PreTrainedPolicy.forward()` est abstrait et requis. Si `VjepaAcPolicy` ne l'implémente pas, le training via `lerobot_train.py` est impossible. À vérifier : est-ce que la policy est uniquement utilisée en inférence (MPC/CEM), ou un training loop custom existe ?

#### 2. Double normalisation des images (risque d'incohérence)
**Sévérité : haute**

La normalisation ImageNet des images se fait à **deux endroits** :
- Dans le `NormalizerProcessorStep` (piloté par `config.normalization_mapping`)
- Dans `_imagenet_normalize()` appelé dans `select_action()` et `_encode_goal_image()`

Si `normalization_mapping` pour les visuels est `MEAN_STD` avec les stats ImageNet, les images seront normalisées deux fois. Si c'est `IDENTITY`, la normalisation dans le modèle est redondante avec la responsabilité du processor. Il faut clarifier qui normalise quoi.

#### 3. `_normalize_state()` dans le modèle
**Sévérité : moyenne**

La normalisation du state (`MIN_MAX`) est faite manuellement dans `select_action()` via `_normalize_state()`, qui accède à `self.dataset_stats` stocké dans le modèle. Cela viole la séparation model/processor : la normalisation devrait être exclusivement dans le `NormalizerProcessorStep`. Résultat : double dépendance aux stats, état interne au modèle difficile à maintenir.

#### 4. `predict_action_chunk` = `select_action` (horizon MPC ignoré)
**Sévérité : faible / design**

`predict_action_chunk` est censé retourner un chunk d'actions (`[B, chunk_size, action_dim]`). Ici il retourne simplement `select_action()` qui fait un pas CEM. Cela signifie que le mécanisme d'action chunking de LeRobot (execution of `n_action_steps` out of `chunk_size`) est bypassed. Ce n'est pas forcément un bug (MPC recalcule à chaque pas), mais c'est une divergence architecturale explicite avec les policies ACT/Diffusion.

#### 5. Non enregistré dans la factory principale
**Sévérité : faible (design externe)**

`vjepa_ac` n'est pas dans `get_policy_class()` de LeRobot. Il passe par le fallback `_get_policy_cls_from_policy_name()`. C'est le bon mécanisme pour une policy externe, mais cela signifie que `make_policy_config("vjepa_ac")` échouera également (ce factory ne supporte pas les policies externes).

#### 6. Dépendance `torch.hub.load` à l'init
**Sévérité : moyenne**

L'encoder est chargé via `torch.hub.load(config.encoder_repo_id, ...)` dans `__init__`. Cela :
- Nécessite un accès réseau ou un cache local à l'initialisation
- Rend le `from_pretrained()` de HuggingFace Hub potentiellement instable si le Hub VJEPA2 change
- Complique les tests unitaires

#### 7. Taille d'image hardcodée avec fallback interpolation
**Sévérité : faible**

`select_action()` interpole les images si `H != config.img_size`. Cela masque des configurations incorrectes silencieusement et peut introduire un coût latence en inférence.

---

## 3. Récapitulatif

| Critère | Statut | Notes |
|---|---|---|
| Interface `PreTrainedPolicy` | ⚠️ Partiel | `forward()` à vérifier |
| Processor pipeline | ✅ Conforme | Pattern standard respecté |
| Séparation model/processor | ⚠️ Violation | `_normalize_state` et `_imagenet_normalize` dans le modèle |
| Factory registration | ✅ Acceptable | Via fallback externe |
| `torch.compile` support | ✅ Bien géré | Override `_load_as_safetensor` propre |
| Action chunking | ⚠️ Bypassed | MPC ≠ chunking, explicite mais divergent |
| Testabilité | ⚠️ Faible | `torch.hub` à l'init, `dataset_stats` dans le modèle |

---

## 4. Recommandations prioritaires

1. **Implémenter `forward()`** ou documenter explicitement que la policy est inference-only et comment lancer le training.
2. **Centraliser toute la normalisation dans le processor** : retirer `_normalize_state()` du modèle et `_imagenet_normalize()` (ou le garder interne mais bloquer la double application via `normalization_mapping=IDENTITY` pour les visuels).
3. **Clarifier le contrat `predict_action_chunk`** : soit retourner un vrai chunk (en répétant l'action CEM), soit documenter explicitement le mode MPC-only et pourquoi chunking ne s'applique pas.

Ecris par Claude