#!/usr/bin/env bash
# vastai_remote_setup.sh — Tourne SUR le remote vast.ai
# Appelé par vastai_launch.sh via SSH dans un screen
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/workspace/lerobot}"
CONFIG_DIR="${CONFIG_DIR:-lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/configs/policy}"
CONFIG_NAME="${CONFIG_NAME:-vjepa_ac_cloud}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-outputs/vjepa_ac/run_cloud}"

echo "════ [1/5] Dépendances système ════"
apt-get update -qq && apt-get install -y -qq \
    ffmpeg \
    libsm6 libxext6 libxrender-dev \
    git \
    > /dev/null 2>&1
echo "✓ ffmpeg installé"

echo "════ [2/5] Installation Python ════"
cd "$REMOTE_DIR"

# lerobot core
pip install -e ".[video_benchmark]" --quiet

# plugin vjepa_ac
pip install -e "lerobot_policy_vjepa_ac/" --quiet

# torchcodec (non inclus dans l'image pytorch de base)
pip install torchcodec --quiet || echo "⚠ torchcodec non installé, fallback sur av"

echo "✓ Packages installés"

echo "════ [3/5] Authentification HF ════"
python -c "
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('✓ HF login OK')
"

echo "════ [4/5] Pré-téléchargement des poids VJEPA2 ════"
python -c "
from huggingface_hub import snapshot_download
import os
print('Téléchargement facebookresearch/vjepa2...')
snapshot_download(repo_id='facebookresearch/vjepa2', cache_dir=os.environ.get('HF_DATASETS_CACHE', '/workspace/.cache/hf'))
print('✓ Poids VJEPA2 téléchargés')
"

echo "════ [5/5] Lancement de l'entraînement ════"
cd "$REMOTE_DIR"

wandb login "$WANDB_API_KEY"

python -m lerobot.scripts.lerobot_train \
    --config-dir "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    output_dir="$REMOTE_OUTPUT"

echo "════ Training terminé ════"
echo "Checkpoints dans : $REMOTE_DIR/$REMOTE_OUTPUT"
echo "Pour récupérer les résultats, lance depuis local :"
echo "  ./scripts/vastai_sync_results.sh"
