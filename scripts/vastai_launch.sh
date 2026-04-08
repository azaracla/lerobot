#!/usr/bin/env bash
# vastai_launch.sh — Lance un training vjepa_ac sur Vast.ai
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export WANDB_API_KEY=xxx
#   ./scripts/vastai_launch.sh                        # nouveau run
#   ./scripts/vastai_launch.sh --instance 12345678    # reprend une instance existante
#
# Prérequis: vastai CLI dans l'env conda lerobot
set -euo pipefail

CONDA_PYTHON="conda run -n lerobot python"
VASTAI="conda run -n lerobot vastai"

# ── Paramètres ────────────────────────────────────────────────────────────────
DOCKER_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=150                     # dataset + weights + checkpoints
LOCAL_CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/workspace/lerobot"
CONFIG_NAME="vjepa_ac_cloud"
CONFIG_DIR="lerobot_policy_vjepa_ac/src/lerobot_policy_vjepa_ac/configs/policy"

# Résultat du run à récupérer en local
RUN_DATE=$(date +%Y%m%d_%H%M)
REMOTE_OUTPUT="outputs/vjepa_ac/run_cloud_${RUN_DATE}"

# ── Arguments ─────────────────────────────────────────────────────────────────
EXISTING_INSTANCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance) EXISTING_INSTANCE="$2"; shift 2 ;;
        *) echo "Usage: $0 [--instance INSTANCE_ID]"; exit 1 ;;
    esac
done

# ── Vérifications ─────────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "❌ HF_TOKEN non défini. Exporte-le avant de lancer ce script."
    exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "❌ WANDB_API_KEY non défini. Exporte-le avant de lancer ce script."
    exit 1
fi

# ── Sélection / création de l'instance ────────────────────────────────────────
if [[ -n "$EXISTING_INSTANCE" ]]; then
    INSTANCE_ID="$EXISTING_INSTANCE"
    echo "→ Utilisation de l'instance existante $INSTANCE_ID"
else
    echo "→ Recherche de la meilleure offre L40S (≥46GB VRAM, fiabilité >97%, réseau pas cher)…"
    OFFER_ID=$(
        $VASTAI search offers \
            'reliability>0.97 num_gpus=1 gpu_name=L40S rentable=True inet_up_cost<0.05 inet_down_cost<0.05' \
            --order 'dph_total+' --raw \
        | $CONDA_PYTHON -c "
import sys, json
offers = json.load(sys.stdin)
# Filtre : disk >= 100 GB
ok = [o for o in offers if o.get('disk_space', 0) >= 100]
if not ok:
    raise RuntimeError('Aucune offre disponible')
best = ok[0]
print(best['ask_contract_id'])
import sys; print(f\"# {best['gpu_name']} {best['gpu_ram']//1024}GB  \${best['dph_total']:.4f}/hr  rel={best['reliability2']:.3f}  disk={best['disk_space']:.0f}GB  inet_down=\${best['inet_down_cost']*1000:.4f}/GB\", file=sys.stderr)
"
    )
    echo "→ Offre sélectionnée : $OFFER_ID"

    echo "→ Création de l'instance…"
    INSTANCE_ID=$(
        $VASTAI create instance "$OFFER_ID" \
            --image "$DOCKER_IMAGE" \
            --disk "$DISK_GB" \
            --ssh \
            --direct \
            --env "HF_TOKEN=${HF_TOKEN} WANDB_API_KEY=${WANDB_API_KEY} HF_DATASETS_CACHE=/workspace/.cache/hf" \
        | grep -oP '(?<=new_contract )\d+'
    )
    echo "→ Instance créée : $INSTANCE_ID"
fi

# ── Attente que l'instance soit prête ─────────────────────────────────────────
echo "→ En attente du démarrage de l'instance (peut prendre 1-3 min)…"
SSH_HOST=""
SSH_PORT=""
for i in $(seq 1 40); do
    sleep 15
    INFO=$(
        $VASTAI show instances --raw \
        | $CONDA_PYTHON -c "
import sys, json
instances = json.load(sys.stdin)
for inst in instances:
    if str(inst.get('id')) == '${INSTANCE_ID}':
        print(inst.get('ssh_host',''), inst.get('ssh_port',''), inst.get('actual_status',''))
        break
"
    )
    STATUS=$(echo "$INFO" | awk '{print $3}')
    SSH_HOST=$(echo "$INFO" | awk '{print $1}')
    SSH_PORT=$(echo "$INFO" | awk '{print $2}')

    echo "  [${i}] status=$STATUS"
    if [[ "$STATUS" == "running" && -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
        break
    fi
done

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo "❌ Impossible de récupérer les infos SSH. Vérifie avec: vastai show instances"
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT"
echo "→ Instance prête : ssh root@${SSH_HOST} -p ${SSH_PORT}"

# ── Transfert du code ──────────────────────────────────────────────────────────
echo "→ Transfert du code vers le remote…"
ssh $SSH_OPTS root@"$SSH_HOST" "mkdir -p $REMOTE_DIR"
rsync -az --progress \
    -e "ssh $SSH_OPTS" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='outputs/' \
    --exclude='.env' \
    --exclude='wandb/' \
    --exclude='*.egg-info' \
    "$LOCAL_CODE_DIR/" \
    "root@${SSH_HOST}:${REMOTE_DIR}/"

# ── Transfert du script remote ─────────────────────────────────────────────────
scp $SSH_OPTS \
    "$LOCAL_CODE_DIR/scripts/vastai_remote_setup.sh" \
    "root@${SSH_HOST}:/workspace/remote_setup.sh"

# ── Lancement du setup et de l'entraînement ───────────────────────────────────
echo "→ Lancement du setup et de l'entraînement (screen -d -m pour fond)…"
ssh $SSH_OPTS root@"$SSH_HOST" "
    chmod +x /workspace/remote_setup.sh
    export HF_TOKEN='${HF_TOKEN}'
    export WANDB_API_KEY='${WANDB_API_KEY}'
    export REMOTE_DIR='${REMOTE_DIR}'
    export CONFIG_DIR='${CONFIG_DIR}'
    export CONFIG_NAME='${CONFIG_NAME}'
    export REMOTE_OUTPUT='${REMOTE_OUTPUT}'
    screen -dmS train bash -c '/workspace/remote_setup.sh 2>&1 | tee /workspace/train.log'
    echo 'Training lancé en background (screen -r train pour suivre)'
"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Instance     : $INSTANCE_ID"
echo "  SSH          : ssh root@${SSH_HOST} -p ${SSH_PORT}"
echo "  Logs         : ssh ... 'screen -r train'  ou  tail -f /workspace/train.log"
echo "  Arrêt + sync : ./scripts/vastai_sync_results.sh $INSTANCE_ID $SSH_HOST $SSH_PORT"
echo "════════════════════════════════════════════════════════════"

# Sauvegarde des infos de connexion pour le script de sync
cat > "$LOCAL_CODE_DIR/outputs/.vastai_session" <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
REMOTE_OUTPUT=$REMOTE_OUTPUT
EOF
echo "→ Infos session sauvegardées dans outputs/.vastai_session"
