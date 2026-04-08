#!/usr/bin/env bash
# vastai_sync_results.sh — Récupère les checkpoints et détruit l'instance
#
# Usage:
#   ./scripts/vastai_sync_results.sh                           # lit outputs/.vastai_session
#   ./scripts/vastai_sync_results.sh INSTANCE_ID HOST PORT     # override manuel
#
# Options:
#   --no-destroy   : ne détruit pas l'instance après sync
set -euo pipefail

VASTAI="conda run -n lerobot vastai"
LOCAL_CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_FILE="$LOCAL_CODE_DIR/outputs/.vastai_session"
DESTROY=true

# Lecture des args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-destroy) DESTROY=false; shift ;;
        *) break ;;
    esac
done

if [[ $# -ge 3 ]]; then
    INSTANCE_ID="$1"; SSH_HOST="$2"; SSH_PORT="$3"
    REMOTE_OUTPUT="outputs/vjepa_ac/run_cloud"
elif [[ -f "$SESSION_FILE" ]]; then
    source "$SESSION_FILE"
else
    echo "❌ Pas de session sauvegardée. Passe INSTANCE_ID HOST PORT en argument."
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT"

echo "→ Sync des checkpoints depuis $SSH_HOST…"
mkdir -p "$LOCAL_CODE_DIR/outputs/vjepa_ac/"
rsync -az --progress \
    -e "ssh $SSH_OPTS" \
    "root@${SSH_HOST}:/workspace/lerobot/${REMOTE_OUTPUT}/" \
    "$LOCAL_CODE_DIR/${REMOTE_OUTPUT}/"

echo "→ Récupération du log complet…"
scp $SSH_OPTS \
    "root@${SSH_HOST}:/workspace/train.log" \
    "$LOCAL_CODE_DIR/${REMOTE_OUTPUT}/train.log" || true

echo "✓ Résultats dans : $LOCAL_CODE_DIR/${REMOTE_OUTPUT}/"

if $DESTROY; then
    echo "→ Destruction de l'instance $INSTANCE_ID…"
    $VASTAI destroy instance "$INSTANCE_ID"
    rm -f "$SESSION_FILE"
    echo "✓ Instance détruite"
else
    echo "→ Instance $INSTANCE_ID conservée (--no-destroy)"
fi
