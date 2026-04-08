#!/usr/bin/env bash
# vastai_launch.sh — Lance un training vjepa_ac sur Vast.ai
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export WANDB_API_KEY=xxx
#   ./scripts/vastai_launch.sh
#
# Prérequis: vastai CLI installé (pip install vastai)
set -euo pipefail

DOCKER_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=200
REPO="azaracla/lerobot"
BRANCH="main"
CONFIG_DIR="lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/configs/policy"
CONFIG_NAME="vjepa_ac_cloud"
REMOTE_OUTPUT="outputs/vjepa_ac/run_cloud_$(date +%Y%m%d_%H%M)"

# Vérifications
[[ -z "${HF_TOKEN:-}" ]]       && echo "❌ HF_TOKEN manquant"       && exit 1
[[ -z "${WANDB_API_KEY:-}" ]]  && echo "❌ WANDB_API_KEY manquant"  && exit 1

# Sélection de l'offre la moins chère
echo "→ Recherche offre L40S…"
OFFER_ID=$(
    vastai search offers \
        'reliability>0.97 num_gpus=1 gpu_name=L40S rentable=True inet_up_cost<0.05 inet_down_cost<0.05' \
        --order 'dph_total+' --raw \
    | python3 -c "
import sys, json
offers = [o for o in json.load(sys.stdin) if o.get('disk_space', 0) >= 100]
if not offers: raise SystemExit('Aucune offre disponible')
best = offers[0]
print(best['ask_contract_id'])
print(f\"  {best['gpu_name']} {best['gpu_ram']//1024}GB  \${best['dph_total']:.4f}/hr  rel={best['reliability2']:.3f}\", file=__import__('sys').stderr)
"
)
echo "→ Offre : $OFFER_ID"

# Script qui tournera automatiquement au démarrage de l'instance
ONSTART=$(cat <<SCRIPT
#!/bin/bash
set -e
exec > /workspace/train.log 2>&1

echo "=== [1/4] Dépendances système ==="
apt-get update -qq && apt-get install -y -qq ffmpeg git > /dev/null

echo "=== [2/4] Clone + install ==="
git clone --depth 1 --branch ${BRANCH} https://github.com/${REPO}.git /workspace/lerobot
cd /workspace/lerobot
pip install -e ".[video_benchmark]" -q
pip install -e "lerobot_policy_vjepa_ac/" -q
pip install torchcodec -q || true

echo "=== [3/4] Auth ==="
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
wandb login "${WANDB_API_KEY}"

echo "=== [4/4] Training ==="
cd /workspace/lerobot
python -m lerobot.scripts.lerobot_train \
    --config-dir "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    output_dir="${REMOTE_OUTPUT}"

echo "=== Done ==="
SCRIPT
)

# Création de l'instance avec onstart-cmd
echo "→ Création de l'instance…"
INSTANCE_ID=$(
    vastai create instance "$OFFER_ID" \
        --image "$DOCKER_IMAGE" \
        --disk "$DISK_GB" \
        --ssh --direct \
        --env "HF_TOKEN=${HF_TOKEN} WANDB_API_KEY=${WANDB_API_KEY}" \
        --onstart-cmd "$ONSTART" \
    | grep -oP '(?<=new_contract )\d+'
)

echo ""
echo "════════════════════════════════════════════"
echo "  Instance  : $INSTANCE_ID"
echo "  Logs  : ssh \$(vastai ssh-url $INSTANCE_ID) 'tail -f /workspace/train.log'"
echo "  Suivi : vastai show instance $INSTANCE_ID"
echo "  Arrêt : vastai destroy instance $INSTANCE_ID"
echo "════════════════════════════════════════════"
