# Multi-Dataset Training avec VJEPA-AC

Ce projet permet d'entraîner VJEPA-AC sur les 119 sous-datasets du community dataset v1.

## Structure

```
lerobot_policy_vjepa_ac/
├── scripts/
│   ├── train_multi_dataset.sh   # Script principal de training
│   └── list_subdatasets.py      # Liste les sous-datasets disponibles
├── configs/policy/
│   └── vjepa_ac_community.yaml # Config de base (à ajuster)
```

## Utilisation

### 1. Vérifier que le dataset est prêt

```bash
python scripts/list_subdatasets.py --root /mnt/nas/datasets/community_dataset_v1
```

### 2. Lancer le training

```bash
bash scripts/train_multi_dataset.sh
```

Le script va:
1. Scanner `/mnt/nas/datasets/community_dataset_v1` pour trouver tous les sous-datasets v3.0
2. Générer la liste des 119 sous-datasets
3. Lancer `lerobot-train` avec `--dataset.repo_id` contenant tous les sous-datasets

### 3. Options de personnalisation

Modifie les variables en haut de `scripts/train_multi_dataset.sh`:
- `DATASET_ROOT`: Chemin vers les données
- `OUTPUT_DIR`: Où stocker les checkpoints
- Batch size, learning rate, etc. via les arguments `--training.batch_size`, etc.

### Exemple avec overrides CLI

```bash
lerobot-train \
    --config_path=configs/policy/vjepa_ac_community.yaml \
    --dataset.repo_id='["AndrejOrsula/lerobot_double_ball_stacking_random", "aimihat/so100_tape"]' \
    --dataset.root=/mnt/nas/datasets/community_dataset_v1 \
    --batch_size=4
```

## Notes

- Avec 16GB RAM, réduit `batch_size` si OOM
- Le multi-dataset concatène tous les sous-datasets via `MultiLeRobotDataset`
- Seuls les clés communes à tous les datasets sont conservées
