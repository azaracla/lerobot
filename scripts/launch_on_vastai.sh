#!/bin/bash
# ============================================================================
# Vast.ai Training Launcher
# Run this script on a Vast.ai GPU instance to train π₀.₅
# ============================================================================

set -e

# ============================================================================
# EDIT THESE BEFORE LAUNCHING ON VAST.AI
# ============================================================================

# Your HuggingFace API token (https://huggingface.co/settings/tokens)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Your Weights & Biases API key (https://wandb.ai/authorize)
export WANDB_API_KEY="your_wandb_key_here"

# Your target HuggingFace repository (format: username/model-name)
# Create a repo at: https://huggingface.co/new
export HF_REPO_ID="your_username/pi05_smolvla_3dprint"

# Training configuration (optional, adjust as needed)
export STEPS=3000                    # Number of training steps
export BATCH_SIZE=32                 # Batch size per GPU
export LEARNING_RATE=1e-4            # Learning rate

# ============================================================================
# Installation (no need to edit below)
# ============================================================================

echo "Starting π₀.₅ training on Vast.ai..."
echo "Time: $(date)"
echo ""

# Create output directory
mkdir -p training_outputs
cd training_outputs

# Clone lerobot if not already present
if [ ! -d "lerobot" ]; then
    echo "Downloading LeRobot..."
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
else
    cd lerobot
fi

# Update to latest
git pull origin main

# Install LeRobot with Pi0.5 support
echo "Installing LeRobot with Pi0.5 support..."
pip install -e ".[pi]" --upgrade

# Install additional dependencies
pip install wandb torch torchvision --upgrade

# ============================================================================
# Configuration
# ============================================================================

echo ""
echo "Configuration Summary:"
echo "  Dataset: azaracla/smolvla_3dprint_plate"
echo "  Base Model: lerobot/pi05_base"
echo "  Steps: $STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output Repo: $HF_REPO_ID"
echo ""

# Validate tokens
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_YOUR_TOKEN_HERE" ]; then
    echo "⚠️  WARNING: HF_TOKEN not set. Dataset download may fail."
    echo "Edit this script and set HF_TOKEN before running."
fi

if [ -z "$WANDB_API_KEY" ] || [ "$WANDB_API_KEY" = "your_wandb_key_here" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not set. Training metrics won't be logged."
    export ENABLE_WANDB=false
else
    export ENABLE_WANDB=true
fi

# ============================================================================
# Run Training
# ============================================================================

echo ""
echo "Starting training at $(date)..."
echo ""

# Set environment variables
export DATASET_REPO="azaracla/smolvla_3dprint_plate"
export OUTPUT_DIR="./outputs/pi05_speedrun"
export PUSH_TO_HUB=true
export TOKENIZERS_PARALLELISM=false

# Make sure training script exists (now in scripts/ directory)
if [ ! -f "scripts/train_pi05_speedrun.sh" ]; then
    echo "Error: Training script not found at scripts/train_pi05_speedrun.sh"
    echo "Make sure you have the latest version of the lerobot repository."
    exit 1
fi

# Execute training
bash scripts/train_pi05_speedrun.sh

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "Training completed at $(date)!"
echo ""
echo "Results saved to: $(pwd)/outputs/pi05_speedrun"
echo "Check the training summary:"
cat outputs/pi05_speedrun/training_summary.txt 2>/dev/null || echo "Training summary not found"

echo ""
echo "Your model has been pushed to: https://huggingface.co/$HF_REPO_ID"
echo ""
