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

# Charger les secrets depuis .env si présent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a

DOCKER_IMAGE="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"
DISK_GB=200
REPO="azaracla/lerobot"
BRANCH="main"
CONFIG_DIR="lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/configs/policy"
CONFIG_NAME="vjepa_ac_cloud"
REMOTE_OUTPUT="outputs/vjepa_ac/run_cloud_$(date +%Y%m%d_%H%M)"

# Vérifications
[[ -z "${HF_TOKEN:-}" ]]       && echo "❌ HF_TOKEN manquant"       && exit 1
[[ -z "${WANDB_API_KEY:-}" ]]  && echo "❌ WANDB_API_KEY manquant"  && exit 1
[[ -z "${VAST_API_KEY:-}" ]]   && echo "❌ VAST_API_KEY manquant"   && exit 1

# Sélection de l'offre la moins chère
echo "→ Recherche offre L40S…"
OFFER_ID=$(
    uvx vastai search offers \
        'reliability>0.97 num_gpus=1 gpu_name=L40S rentable=True inet_up_cost<0.05 inet_down_cost<0.05 inet_down>=500' \
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

# Auto-destroy on exit (success or error)
cleanup() {
    echo "=== Destroying instance ==="
    pip install vastai -q --break-system-packages 2>/dev/null || true
    SELF_ID=\$(vastai show instances --api-key "\${VAST_API_KEY}" --raw 2>/dev/null | python3 -c "import sys,json; instances=json.load(sys.stdin); [print(i['id']) for i in instances]" | head -1)
    vastai destroy instance "\${SELF_ID}" --api-key "\${VAST_API_KEY}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== [1/4] Dépendances système ==="
apt-get update -qq && apt-get install -y -qq ffmpeg git > /dev/null

echo "=== [2/4] Clone + install ==="
git clone --depth 1 --branch ${BRANCH} https://github.com/${REPO}.git /workspace/lerobot
cd /workspace/lerobot
pip install --break-system-packages -e ".[video_benchmark]" -q
pip install --break-system-packages -e "lerobot_policy_vjepa_ac/" -q
pip install --break-system-packages torchcodec -q || true

echo "=== [2b/4] Téléchargement encoder VJEPA2 ==="
mkdir -p /root/.cache/torch/hub/checkpoints
wget --progress=bar:force -O /root/.cache/torch/hub/checkpoints/vjepa2_1_vitg_384.pt \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt

echo "=== [3/4] Auth ==="
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
wandb login "${WANDB_API_KEY}"

echo "=== [4/4] Training ==="
cd /workspace/lerobot
lerobot-train --config_path="${CONFIG_DIR}/${CONFIG_NAME}.yaml" --output_dir="${REMOTE_OUTPUT}"

echo "=== Done ==="
SCRIPT
)

# Création de l'instance avec onstart-cmd
echo "→ Création de l'instance…"
INSTANCE_ID=$(
    uvx vastai create instance "$OFFER_ID" \
        --image "$DOCKER_IMAGE" \
        --disk "$DISK_GB" \
        --ssh --direct \
        --env "HF_TOKEN=${HF_TOKEN} WANDB_API_KEY=${WANDB_API_KEY} VAST_API_KEY=${VAST_API_KEY}" \
        --onstart-cmd "$ONSTART" \
    | grep -oP "(?<='new_contract': )\d+"
)

echo ""
echo "════════════════════════════════════════════"
echo "  Instance  : $INSTANCE_ID"
echo "  Logs  : ssh \$(uvx vastai ssh-url $INSTANCE_ID) 'tail -f /workspace/train.log'"
echo "  Suivi : uvx vastai show instance $INSTANCE_ID"
echo "  Arrêt : uvx vastai destroy instance $INSTANCE_ID"
echo "════════════════════════════════════════════"
