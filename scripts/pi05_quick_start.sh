#!/bin/bash
# ============================================================================
# π₀.₅ Quick Start Script - Fastest way to begin training
# ============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# Quick Start Setup
# ============================================================================

print_header "π₀.₅ Quick Start Setup"

echo "This script will help you configure and start training π₀.₅ in 3 steps."
echo ""

# Step 1: API Tokens
print_header "Step 1: Configure API Tokens"

echo "You need two API tokens:"
echo "  1. HuggingFace token: https://huggingface.co/settings/tokens"
echo "  2. Wandb token: https://wandb.ai/authorize"
echo ""

read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN_INPUT
if [ -n "$HF_TOKEN_INPUT" ]; then
    export HF_TOKEN="$HF_TOKEN_INPUT"
    print_success "HuggingFace token set"
else
    echo "⚠️  Warning: Training will fail if dataset is private"
fi

read -p "Enter your Wandb API key (or press Enter to skip): " WANDB_INPUT
if [ -n "$WANDB_INPUT" ]; then
    export WANDB_API_KEY="$WANDB_INPUT"
    print_success "Wandb API key set"
else
    print_success "Wandb logging disabled (local logs only)"
fi

# Step 2: HuggingFace Repo
print_header "Step 2: Configure Output Repository"

echo "Your trained model will be pushed to HuggingFace Hub."
echo "Create a new repo at: https://huggingface.co/new"
echo ""

read -p "Enter your target repo (format: username/model-name) [default: skip]: " HF_REPO_INPUT
if [ -n "$HF_REPO_INPUT" ]; then
    export HF_REPO_ID="$HF_REPO_INPUT"
    print_success "Target repo: $HF_REPO_ID"
else
    export PUSH_TO_HUB=false
    print_success "Model will be saved locally only"
fi

# Step 3: Training Parameters
print_header "Step 3: Training Configuration"

echo "Choose training duration:"
echo "  1) Fast (1000 steps, ~30-60 min, \$1-2)"
echo "  2) Standard (3000 steps, ~1-3 hours, \$2-4)"
echo "  3) Thorough (10000 steps, ~3-8 hours, \$5-10)"
echo ""

read -p "Choose [1-3] [default: 2]: " STEPS_CHOICE
case $STEPS_CHOICE in
    1) export STEPS=1000 ;;
    3) export STEPS=10000 ;;
    *) export STEPS=3000 ;;
esac

print_success "Training steps: $STEPS"

# Optional: Batch size
read -p "Enter batch size [default: 32]: " BATCH_INPUT
if [ -n "$BATCH_INPUT" ]; then
    export BATCH_SIZE="$BATCH_INPUT"
else
    export BATCH_SIZE=32
fi
print_success "Batch size: $BATCH_SIZE"

# ============================================================================
# Summary and Confirmation
# ============================================================================

print_header "Summary"

echo "Configuration:"
echo "  Dataset: azaracla/smolvla_3dprint_plate"
echo "  Base Model: lerobot/pi05_base"
echo "  Training Steps: $STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output Repo: ${HF_REPO_ID:-'(local only)'}"
echo "  Wandb: ${WANDB_API_KEY:+'enabled':'disabled'}"
echo ""

read -p "Ready to start training? [y/N]: " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Training cancelled."
    exit 0
fi

# ============================================================================
# Pre-flight Check
# ============================================================================

print_header "Pre-flight Check"

if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.10+"
    exit 1
fi
print_success "Python: $(python --version)"

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA GPU not detected - training will be very slow"
else
    print_success "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

if ! python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    print_error "PyTorch not installed"
    exit 1
fi
print_success "PyTorch installed"

if ! python -c "import lerobot" 2>/dev/null; then
    print_header "Installing LeRobot"
    pip install -e ".[pi]" > /dev/null 2>&1 || pip install -e "." > /dev/null 2>&1
    print_success "LeRobot installed"
else
    print_success "LeRobot already installed"
fi

# ============================================================================
# Start Training
# ============================================================================

print_header "Starting Training"

echo "Training will start now. This may take 1-8 hours depending on:"
echo "  - GPU type (H100 is fastest)"
echo "  - Batch size (larger = faster but more memory)"
echo "  - Number of steps"
echo ""

echo "You can monitor training at:"
if [ -n "$WANDB_API_KEY" ]; then
    echo "  - Wandb: https://wandb.ai"
fi
echo "  - Logs: tail -f outputs/pi05_speedrun/training.log"
echo ""

read -p "Press Enter to start training or Ctrl+C to cancel..."

# Export all configuration for training script
export ENABLE_WANDB="${WANDB_API_KEY:+true}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

# Run training (use absolute path to script in same directory)
bash "$SCRIPT_DIR/train_pi05_speedrun.sh"

# Success message
print_header "Training Complete!"

echo "Your trained model is ready at:"
if [ "$PUSH_TO_HUB" = "true" ] && [ -n "$HF_REPO_ID" ]; then
    echo "  🤗 HuggingFace Hub: https://huggingface.co/$HF_REPO_ID"
fi
echo "  💾 Local: outputs/pi05_speedrun"
echo ""
echo "Next steps:"
echo "  1. Check results: cat outputs/pi05_speedrun/training_summary.txt"
echo "  2. View model: ls outputs/pi05_speedrun/checkpoints"
echo "  3. Use model: see PI05_SPEEDRUN_GUIDE.md"
echo ""
