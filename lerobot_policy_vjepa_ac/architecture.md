# Architecture VJEPA-AC

## Vue d'ensemble

Policy goal-conditionée utilisant un world model VJEPA2 pour la planification d'actions via CEM (Cross-Entropy Method).

## Composants

### 1. Encoder ViT (gelé)
- **Source** : `facebookresearch/vjepa2` via torch hub
- **Modèle** : `vjepa2_1_vit_giant_384` (ViT-Giant, RoPE, 40 layers, embed_dim=1408)
- **Input** : `[B, C, 1, H, W]` image normalisée ImageNet
- **Output** : `[B, N_patches, D]` latents par patch
- **Usage** : Encode l'image courante et le goal image en latents

### 2. AC Predictor (entraînable)
- **Source** : `ac_predictor_utils.py` (copie de `vjepa2/src/models/ac_predictor.py`)
- **Input** : `(context_latents, actions, states)` 
- **Token interleaving** : `[action_token, state_token, frame_tokens...]` par timestep
- **Output** : `[B, T * H*W, D]` latents futurs prédits
- **Attention** : Block-causal, chaque timestep voit lui-même et les précédents

### 3. CEM (Cross-Entropy Method)
- **Principe** : Sample N trajectoires d'actions, rollout via le predictor, sélectionne les elites, met à jour la distribution
- **Output** : Positions absolues de joints (6D) — compatible SO-100/101
- **Paramètres** : `cem_num_samples`, `cem_num_iters`, `mpc_horizon`, `cem_elite_ratio`, `cem_std`

## Flux d'inférence

```
1. select_action(batch)
   ├── Encode image courante → current_latent
   ├── Utilise goal_latent (pré-encodé au __init__)
   └── CEM loop:
       ├── Sample N trajectoires de positions absolues autour du state courant
       ├── Rollout autoregressif via predictor
       ├── Cost = L1(final_latent, goal_latent)
       ├── Sélection elites + update distribution
       └── Retourne mu[0] (première action de la meilleure trajectoire)
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `modeling_vjepa_ac.py` | Policy principale, CEM, training forward |
| `configuration_vjepa_ac.py` | Config avec hyperparamètres CEM et goal_image_path |
| `ac_predictor_utils.py` | VisionTransformerPredictorAC |
| `processor_vjepa_ac.py` | Pre/post processors (normalisation, device) |
| `transforms.py` | Augmentations d'images |

## Commande lerobot-record

```bash
lerobot-record \
  --robot.type=so_follower \
  --robot.port=/dev/tty.usbmodem58760431541 \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --robot.id=black \
  --dataset.repo_id=azaracla/eval_vjepa_ac_test \
  --dataset.single_task="Pick up the object" \
  --dataset.num_episodes=5 \
  --policy.path=outputs/vjepa_ac/run_20260406_overfit_4/checkpoints/last \
  --policy.goal_image_path=/chemin/vers/goal_image.png
```
