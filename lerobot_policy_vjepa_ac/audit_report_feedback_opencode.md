# Feedback sur l'Audit Report V-JEPA 2-AC

## Points forts de l'audit

L'audit présente une analyse **solide et bien structurée** de l'intégration V-JEPA 2-AC dans LeRobot. Les points suivants sont particulièrement bien traité :

1. **Trace complète de la chaîne preprocessor** (§2b) — La progression Dataset → batch_to_transition → processors → forward() est documentée avec précision. Cela permet de comprendre exactement où chaque transformation opère.

2. **Identification du mismatch train/inférence fondamental** (§3.1) — Le problème des deltas calculés sur des states normalisés vs bruts est le bug le plus critique. L'analyse est correcte et le fix proposé est appropriée.

3. **Distinction clair entre horizon d'entraînement et horizon d'inférence** — La clarification entre `n_obs_steps` (contexte d'observation) et `mpc_horizon` (planning futur) est importante et souvent confuse dans les implémentations MPC.

4. **Tableau de recommandations priorisées** — La structure P0/P1/P2 permet de prioriser le travail. C'est pragmatique.

---

## Questions et points à clarifier

### 1. Raw deltas caching — Fragilité du global state

**Localisation** : §3.1, fix proposé pour `processor_vjepa_ac.py` et `modeling_vjepa_ac.py`

**Problème** : Le fix utilise un dictionnaire global `_DELTA_STATE_CACHE` :

```python
_DELTA_STATE_CACHE["raw_deltas"] = self._raw_deltas
```

Cette approche pose plusieurs questions :

1. **Garantie de synchronisation** : Comment `select_action()` peut-il garantir que le cache correspond à l'épisode courant ? Si le robot execute plusieurs épisodes sans recharger le modèle, le cache pourrait contenir des données périmées.

2. **Cas multi-robot** : Si plusieurs instances du processor tournent en parallèle (par exemple plusieurs robots physiques), le global state serait partagé → race condition potentielle.

3. **Sérialisation** : Le cache ne survive pas à un `from_pretrained()` / `save_pretrained()`. Le processor perd son état entre deux sessions.

**Alternative suggérée** :

Plutôt qu'un global cache, deux options plus robustes :

**Option A** — Passer les raw deltas via le processor post-traitement :

Le post-processor `DeltaToAbsoluteActionProcessorStep` a déjà accès à `_current_state`. On pourrait étendre son rôle :

```python
# Dans processor_vjepa_ac.py — postprocessor
class DeltaToAbsoluteActionProcessorStep:
    def __call__(self, action, observation, current_state=None):
        if current_state is not None and action.ndim == 2:
            # action = delta, current_state = dernière position connue
            return current_state + action
        return action
```

Mais le problème est que le preprocessor ne peut pas parler au post-processor directement.

**Option B** — Ajouter un paramètre optionnel à `select_action()` :

```python
# Dans modeling_vjepa_ac.py, select_action()
def select_action(
    self,
    observation: dict,
    # ... autres params
    raw_deltas: torch.Tensor | None = None,  # Optionnel, priority over computed
):
    if raw_deltas is not None:
        hist_actions = raw_deltas
    else:
        # Fallback : computed deltas (existing logic)
        hist_actions = states[:, 1:] - states[:, :-1]
```

Cette approche est explicite et ne dépend pas d'état global. Elle nécessite cependant une modification de l'interface LeRobot.

**Option C** — Stocker les raw deltas dans le processor, exposed via une méthode :

```python
class StateToDeltaActionProcessorStep:
    def get_raw_deltas(self) -> torch.Tensor | None:
        return getattr(self, '_raw_deltas', None)
```

Le modèle peut alors appeler `processor.get_raw_deltas()` avant `select_action()`. Requiert que le modèle ait une référence au processor.

**Recommandation** : L'Option B est la plus propre car elle ne зависи pas d'état caché. Cependant, si on veut rester simple, l'Option A/C fonctionne pour un déploiement mono-robot. Pour l'instant, le global cache est une solution pragmatique à documenter comme temporaire.

---

### 2. CEM rolling bug — Possibilité d'horizon > 1 avec V-JEPA 2.1

**Localisation** : §3.3

Le rapport note que `mpc_horizon=1` évite le bug des actions utilisées comme states. Cependant :

**Contexte** :
- V-JEPA 2 original (Table 3) : `mpc_horizon=1`
- V-JEPA 2.1 (Table 6) : `mpc_horizon=8` avec meilleur encodeur

Notre encodeur est V-JEPA 2.1, donc potentiellement capable d'un horizon plus grand.

**Question** : Est-ce que le fix §3.3 (maintenir trajectoire d'état) a été testé ? Avec un horizon de 8, le CEM pourrait être plus efficace pour éviter les minima locaux.

**Réflexion** : Le problème identifié est que les actions (deltas bruts ~0.01-0.05) sont passés au state_encoder qui attend des positions normalisées [-1, 1]. La solution de maintenir une trajectoire d'état est correcte :

```
last_known_state = hist_states[:, -1:]  # [N, 1, D] — MIN_MAX normalized
for h in range(1, H):
    _a = actions[:, h:h+1]  # delta brut
    # 1) Dé-normaliser last_known_state
    # 2) Ajouter le delta
    # 3) Re-normaliser pour le prochain pas
```

Cependant, cela nécessite d'avoir accès aux statistiques de normalisation (min/max). Le normalizer est un processor séparé, donc le modèle devrait avoir une référence au normalizer pour faire cette dénormalisation.

**Suggestion** : Si `mpc_horizon=1` est utilisés, le fix §3.3 n'est pas nécessaire. Mais si on veut مستقبلment passer à horizon > 1, il faudrait :
1. Stocker les stats de normalisation dans le config
2. Implémenter la dénormalisation dans la boucle CEM

---

### 3. auto_steps — Vérification du clamp

**Localisation** : §4.1

Le rapport indique que le code clamp `auto_steps` à `T_full - 1 = 3` :

```python
auto_steps = min(getattr(self.config, "auto_steps", 1), T_full - 1)
```

avec `T_full = n_obs_steps + n_action_steps`.

**Vérification nécessaire** : Dans le code actuel, `n_action_steps` est calculé comment ?

- Si `n_action_steps = mpc_horizon = 15`, alors `T_full = 4 + 15 = 19`, et `auto_steps` pourrait aller jusqu'à 18.
- Si `n_action_steps = n_obs_steps - 1` (les deltas disponibles), alors `T_full = 4 + 3 = 7`, et `auto_steps` clampé à 6.

Le rapport suppose `T_full = n_obs_steps = 4` → 3 deltas disponibles.

**Question** : Quelle est la valeur de `n_action_steps` dans le config actuel ? Si c'est `mpc_horizon`, alors il n'y a pas de clamp et `auto_steps=2` peut être utilisé sans problème. Si c'est `n_obs_steps - 1`, alors le clamp est déjà correct.

**À vérifier dans** : `configuration_vjepa_ac.py` et comment `n_action_steps` est défini.

---

### 4. Masque d'attention — Correction du calcul

**Localisation** : §5.2

Le fix proposé :

```python
max_temporal_depth = config.n_obs_steps  # masque utilisé en training uniquement
```

Ce fix est **incomplet** si `auto_steps > 1`. En effet :

- En training avec teacher forcing : on a `n_obs_steps` frames d'observation
- En training avec auto-regressive (`auto_steps > 1`) : le forward doit prédire `auto_steps` pas supplémentaires
- La séquence totale est donc `n_obs_steps + auto_steps`

Si `auto_steps=2` est implémenté, le masque doit couvrir `4 + 2 = 6` pas temporels.

**Correction suggérée** :

```python
# training : masque couvre observation + prediction autoregressive
# CEM (T=1) : masque pas utilisé (skip flash attention)
max_temporal_depth = config.n_obs_steps + getattr(config, 'auto_steps', 1)
max_seq_len = max_temporal_depth * config.tubelet_size
```

Cela donne `4 + 2 = 6` → `6 * (576 + 2) = 3468` → matrice `3468 x 3468` = **~12 MB** au lieu de 82 MB, tout en couvrant le cas auto_steps > 1.

---

### 5. ImageNet norm dans select_action()

**Localisation** : §2b, trace d'inférence, ligne 106-107

L'audit note correctement que `select_action()` n'applique pas ImageNet norm :

```
RÉSULTAT POUR select_action() :
   images : normalisées (non-ImageNet, fait dans select_action)
```

C'est en effet le cas dans `modeling_vjepa_ac.py:180-187` :

```python
# ImageNet normalization inside encoder
img_seq = self.encoder(images)
```

L'encodeur V-JEPA applique sa propre normalisation interne. Donc ce point n'est pas un bug — c'est le comportement attendu. L'encodeur V-JEPA (comme tous les encodeurs V-JEPA) applique la normalisation ImageNet de manière interne.

**Note** : En revanche, le forward training fait aussi `image_transform` dans `ac_predictor_utils.py`. Donc double normalisation ? Non — le forward training utilise aussi l'encodeur qui appliquant ImageNet norm, et le `image_transform` préalable est une normalisation basique [0,1].

Vérifier que les deux utilisent bien le même pattern pour éviter un mismatch.

---

### 6. Numerical precision — bfloat16 dans CEM

**Non mentionné dans l'audit**

Le forward CEM utilise :

```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
    pred_h = self.predictor(current_z, _a, _s)
```

Questions :

1. **Precision loss** : Le predictor est entraîné en float32, mais l'inférence utilise bfloat16. La précision peut être suffisante, mais il faudrait vérifier empiriquement.

2. **Accumulation d'erreur** : Dans la boucle CEM rolling avec horizon > 1, l'accumulation de bfloat16 sur plusieurs pas pourrait dégrader les prédictions.

3. **Comparaison avec l'original** : L'original utilise quelle précision ? (Probablement float32 ou fp16.)

**Suggestion** : Tester les deux. Si les résultats sont satisfaisants avec bfloat16, le garder pour la performance. Sinon, revenir à float32.

---

### 7. Points manquants dans l'audit

Les aspects suivants ne sont pas couverts mais seraient utiles :

#### a) FlashAttention availability
Le rapport mentionne le skip du masque pour T=1 (§2.1) pour activer FlashAttention, mais :
- FlashAttention est-il bien installé dans l'environnement ?
- Y a-t-il un fallback vers SDPA si FlashAttention n'est pas disponible ?
- Le dtype bfloat16 est requis pour FlashAttention — est-ce bien configuré ?

#### b) Gradient checkpointing
Avec un encodeur ViT-Giant-384 frozen et un predictor de ~1B paramètres :
- Le gradient checkpointing est-il activé pour le predictor ?
- Sans checkpointing, le training avec batch=4 pourrait facilement OOM sur une RTX 5070 Ti.

#### c) Batch size et scaling
Le tableau §8 montre un scaling de 64x entre le training Meta (batch=256) et le notre (batch=4). Questions :
- Le learning rate est-il ajusté ? (1e-5 est suggéré comme équivalent)
- Est-ce que le training converge avec ce petit batch ?
- Avez-vous envisagé le gradient accumulation pour simuler un batch plus grand ?

#### d) Evaluation benchmark
- Comment l'évaluation zero-shot est-elle prévue ? (Pas de code d'eval dans l'audit)
- Y a-t-il un protocole d'évaluation défini pour SO-101 ?

#### e) Checkpointing et resume
- Comment les checkpoints sont-ils sauvegardés ?
- Le resume fonctionne-t-il correctement après un crash ?

---

## Recommandations总结

### Priorité haute (à faire avant le prochain training)

1. **Clarifier le fix §3.1** : Documenter les limitations du global cache ou passer à une approche plus robuste
2. **Vérifier n_action_steps** : Confirmer comment `n_action_steps` est défini pour savoir si le clamp auto_steps est nécessaire
3. **Compléter le fix §5.2** : Utiliser `n_obs_steps + auto_steps` au lieu de `n_obs_steps` seul
4. **Ajouter auto_steps=2** : Comme recommandé, mais vérifier que le clamp fonctionne

### Priorité moyenne (à faire avant l'inférence robot)

5. **Vérifier ImageNet norm** : S'assurer que forward et select_action utilisent le même pattern
6. **Tester bfloat16 vs float32** : Valider que la précision est suffisante pour le CEM

### Priorité basse (optimisations futures)

7. **Gradient checkpointing** : Si OOM, activer pour le predictor
8. **Horizon > 1** : Si les résultats avec horizon=1 ne sont pas satisfaisants, implémenter le fix §3.3
9. **cem_std_per_joint** : Optionnel, moins critique pour SO-101

---

## Conclusion

L'audit est **bien fait et pertinent**. Les 5 problèmes critiques identifiés sont corrects. Les remarques ci-dessus visent à :

1. **Affiner certains fixes** (masque attention, raw deltas cache)
2. **Ajouter du contexte** (ImageNet norm, bfloat16)
3. **Identifier les manquant** (gradient checkpointing, evaluation)

Les points prioritaires restent les fixes §3.1, §3.2, §3.4 et le changement de `auto_steps=2`. Une fois ces corrections appliquées, le modèle devrait être fonctionnel pour de l'inférence robot.