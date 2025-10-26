#!/bin/bash
# ============================================================================
# π₀.₅ Training Speedrun Script for Vast.ai
# Automated training pipeline for LeRobot Pi0.5 on so101_3dprint_plate dataset
# Inspired by: https://github.com/karpathy/nanochat/blob/master/speedrun.sh
#
# Multi-GPU Support:
# - Automatically detects and uses all available GPUs by default
# - Set ENABLE_MULTI_GPU=false to disable multi-GPU training
# - Set NUM_GPUS=N to use a specific number of GPUs
# - Learning rate and steps are automatically scaled for multi-GPU
# - Uses Hugging Face Accelerate for distributed training
#
# Docs: https://huggingface.co/docs/lerobot/multi_gpu_training
# ============================================================================

set -e  # Exit on first error

# Change to project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_REPO="azaracla/so101_3dprint_plate"
BASE_MODEL="lerobot/pi05_base"
OUTPUT_DIR="${OUTPUT_DIR:=./outputs/pi05_speedrun}"
JOB_NAME="${JOB_NAME:=pi05_speedrun_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Training hyperparameters
STEPS="${STEPS:=3000}"
BATCH_SIZE="${BATCH_SIZE:=32}"
NUM_WORKERS="${NUM_WORKERS:=4}"
LEARNING_RATE="${LEARNING_RATE:=1e-4}"
# Note: WARMUP_STEPS removed - scheduler uses policy defaults
LOG_FREQ="${LOG_FREQ:=100}"
SAVE_FREQ="${SAVE_FREQ:=500}"

# Model repo for pushing (should be: username/repo-name)
HF_REPO_ID="${HF_REPO_ID:=}"
PUSH_TO_HUB="${PUSH_TO_HUB:=true}"

# API tokens (from environment variables)
HF_TOKEN="${HF_TOKEN:=}"
WANDB_API_KEY="${WANDB_API_KEY:=}"

# Flags
ENABLE_WANDB="${ENABLE_WANDB:=true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:=false}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:=}"

# Multi-GPU configuration
ENABLE_MULTI_GPU="${ENABLE_MULTI_GPU:=true}"
NUM_GPUS="${NUM_GPUS:=auto}"  # auto = use all available GPUs, or specify number
MIXED_PRECISION="${MIXED_PRECISION:=bf16}"  # bf16, fp16, or no

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# Pre-flight checks
# ============================================================================

print_header "Pre-flight Checks"

# Check Python version
if ! command_exists python; then
    print_error "Python not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"

# Check CUDA and detect GPUs
if ! command_exists nvidia-smi; then
    print_warning "NVIDIA GPU not detected. Training will be very slow on CPU."
    DETECTED_GPUS=0
    ENABLE_MULTI_GPU="false"
else
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_success "CUDA GPU(s) detected: $DETECTED_GPUS"
    nvidia-smi --query-gpu=name --format=csv,noheader | nl | while read num name; do
        echo "  GPU $num: $name"
    done

    # Set NUM_GPUS if auto
    if [ "$NUM_GPUS" = "auto" ]; then
        NUM_GPUS=$DETECTED_GPUS
    fi

    # Validate NUM_GPUS
    if [ "$NUM_GPUS" -gt "$DETECTED_GPUS" ]; then
        print_warning "Requested $NUM_GPUS GPUs but only $DETECTED_GPUS available. Using $DETECTED_GPUS."
        NUM_GPUS=$DETECTED_GPUS
    fi

    # Disable multi-GPU if only 1 GPU
    if [ "$NUM_GPUS" -le 1 ]; then
        ENABLE_MULTI_GPU="false"
        print_warning "Only 1 GPU available or requested. Multi-GPU disabled."
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# ============================================================================
# Environment Setup
# ============================================================================

print_header "Environment Setup"

# Check if lerobot is installed
if ! python -c "import lerobot" 2>/dev/null; then
    print_warning "LeRobot not installed. Installing with Pi0.5 dependencies..."
    pip install -e ".[pi]"
    print_success "LeRobot installed"
else
    print_success "LeRobot already installed"
fi

# Check if accelerate is installed (required for multi-GPU)
if [ "$ENABLE_MULTI_GPU" = "true" ]; then
    if ! python -c "import accelerate" 2>/dev/null; then
        print_warning "Accelerate not installed. Installing for multi-GPU support..."
        pip install accelerate
        print_success "Accelerate installed"
    else
        print_success "Accelerate already installed"
    fi
fi

# Validate API tokens
if [ -z "$HF_TOKEN" ]; then
    print_warning "HF_TOKEN not set. Dataset download may fail for private repos."
    print_warning "Set with: export HF_TOKEN=your_hf_token"
fi

if [ "$ENABLE_WANDB" = "true" ] && [ -z "$WANDB_API_KEY" ]; then
    print_warning "WANDB_API_KEY not set. Logging will be disabled."
    print_warning "Set with: export WANDB_API_KEY=your_wandb_key"
    ENABLE_WANDB="false"
fi

# Create virtual env if needed
if [ ! -d "venv" ]; then
    print_warning "No virtual environment found. Consider creating one:"
    print_warning "  python -m venv venv && source venv/bin/activate"
fi

print_success "Environment check complete"

# ============================================================================
# Dataset Preparation
# ============================================================================

print_header "Dataset Preparation"

log_step "Validating dataset: $DATASET_REPO"

# Export variables for Python script
export DATASET_REPO
export HF_TOKEN

# Python script to validate and prepare dataset
python << 'EOF'
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import list_repo_files
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_repo = os.environ.get("DATASET_REPO")
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Checking dataset: {dataset_repo}")

    # Try to load dataset metadata
    try:
        dataset = LeRobotDataset(dataset_repo, episodes=list(range(1)))
        print(f"✓ Dataset found with {dataset.num_episodes} episodes")
        print(f"  Total frames: {dataset.num_frames}")
        print(f"  FPS: {dataset.meta.fps}")

        # Check if quantile stats exist
        if hasattr(dataset.meta, 'stats') and dataset.meta.stats:
            print("✓ Quantile normalization stats already present")
        else:
            print("⚠ No quantile stats found. Will use MEAN_STD normalization")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        sys.exit(1)

except ImportError as e:
    print(f"⚠ Warning: {e}")
    print("  Dataset validation skipped, will proceed with training")

EOF

log_step "Dataset preparation complete"
print_success "Dataset ready"

# ============================================================================
# Training Preparation
# ============================================================================

print_header "Training Configuration"

# Adjust hyperparameters for multi-GPU
# LeRobot does NOT automatically scale these, so we must do it manually
EFFECTIVE_BATCH_SIZE=$BATCH_SIZE
EFFECTIVE_LR=$LEARNING_RATE
EFFECTIVE_STEPS=$STEPS

if [ "$ENABLE_MULTI_GPU" = "true" ]; then
    # Effective batch size = batch_size × num_gpus
    EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

    # Scale learning rate linearly with number of GPUs
    EFFECTIVE_LR=$(python -c "print($LEARNING_RATE * $NUM_GPUS)")

    # Reduce steps proportionally since effective batch size increases
    EFFECTIVE_STEPS=$(python -c "import math; print(math.ceil($STEPS / $NUM_GPUS))")

    print_success "Multi-GPU scaling applied:"
    echo "  GPUs: $NUM_GPUS"
    echo "  Batch size per GPU: $BATCH_SIZE → Effective batch: $EFFECTIVE_BATCH_SIZE"
    echo "  Learning rate scaled: $LEARNING_RATE → $EFFECTIVE_LR"
    echo "  Steps adjusted: $STEPS → $EFFECTIVE_STEPS"
    echo ""
fi

# Build training command
if [ "$ENABLE_MULTI_GPU" = "true" ]; then
    TRAIN_CMD="accelerate launch --multi_gpu --num_processes=$NUM_GPUS"
    if [ "$MIXED_PRECISION" != "no" ]; then
        TRAIN_CMD="$TRAIN_CMD --mixed_precision=$MIXED_PRECISION"
    fi
    TRAIN_CMD="$TRAIN_CMD src/lerobot/scripts/lerobot_train.py"
else
    TRAIN_CMD="python src/lerobot/scripts/lerobot_train.py"
fi

# Core arguments
TRAIN_CMD="$TRAIN_CMD --dataset.repo_id=$DATASET_REPO"
TRAIN_CMD="$TRAIN_CMD --policy.type=pi05"
TRAIN_CMD="$TRAIN_CMD --policy.pretrained_path=$BASE_MODEL"
TRAIN_CMD="$TRAIN_CMD --output_dir=$OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --job_name=$JOB_NAME"
TRAIN_CMD="$TRAIN_CMD --steps=$EFFECTIVE_STEPS"
TRAIN_CMD="$TRAIN_CMD --batch_size=$BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --num_workers=$NUM_WORKERS"

# Optimizer and learning rate
TRAIN_CMD="$TRAIN_CMD --optimizer.lr=$EFFECTIVE_LR"
# Note: scheduler parameters use policy defaults (warmup is policy-specific)

# Model optimization for faster training
TRAIN_CMD="$TRAIN_CMD --policy.compile_model=true"
TRAIN_CMD="$TRAIN_CMD --policy.gradient_checkpointing=true"

# Note: Mixed precision is handled by accelerate (--mixed_precision) for multi-GPU
# For single GPU, we still use policy.dtype
if [ "$ENABLE_MULTI_GPU" != "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --policy.dtype=bfloat16"
fi

# Checkpointing and logging
TRAIN_CMD="$TRAIN_CMD --log_freq=$LOG_FREQ"
TRAIN_CMD="$TRAIN_CMD --save_freq=$SAVE_FREQ"
TRAIN_CMD="$TRAIN_CMD --save_checkpoint=true"

# Logging configuration
if [ "$ENABLE_WANDB" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb.enable=true"
    if [ -n "$HF_REPO_ID" ]; then
        TRAIN_CMD="$TRAIN_CMD --wandb.project=lerobot-pi05"
        TRAIN_CMD="$TRAIN_CMD --wandb.entity=$(echo $HF_REPO_ID | cut -d'/' -f1)"
    fi
else
    TRAIN_CMD="$TRAIN_CMD --wandb.enable=false"
fi

# Hub pushing
if [ "$PUSH_TO_HUB" = "true" ] && [ -n "$HF_REPO_ID" ]; then
    TRAIN_CMD="$TRAIN_CMD --policy.push_to_hub=true"
    TRAIN_CMD="$TRAIN_CMD --policy.repo_id=$HF_REPO_ID"
else
    TRAIN_CMD="$TRAIN_CMD --policy.push_to_hub=false"
fi

# Resume from checkpoint if specified
if [ "$RESUME_FROM_CHECKPOINT" = "true" ] && [ -n "$CHECKPOINT_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume=true"
    TRAIN_CMD="$TRAIN_CMD --checkpoint_path=$CHECKPOINT_PATH"
fi

# Device (only for single GPU, accelerate handles multi-GPU automatically)
if [ "$ENABLE_MULTI_GPU" != "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --policy.device=cuda"
fi

# Normalization mapping (optional, for datasets without quantile stats)
TRAIN_CMD="$TRAIN_CMD --policy.normalization_mapping='{\"ACTION\": \"MEAN_STD\", \"STATE\": \"MEAN_STD\", \"VISUAL\": \"IDENTITY\"}'"

# Display configuration
echo "Training Configuration:"
echo "  Dataset: $DATASET_REPO"
echo "  Base Model: $BASE_MODEL"
if [ "$ENABLE_MULTI_GPU" = "true" ]; then
    echo "  Multi-GPU: Enabled ($NUM_GPUS GPUs)"
    echo "  Mixed Precision: $MIXED_PRECISION"
    echo "  Batch Size per GPU: $BATCH_SIZE"
    echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
    echo "  Learning Rate (scaled): $EFFECTIVE_LR"
    echo "  Steps (adjusted): $EFFECTIVE_STEPS"
else
    echo "  Multi-GPU: Disabled"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Steps: $STEPS"
fi
echo "  Model Compilation: true"
echo "  Gradient Checkpointing: true"
echo "  Wandb: $ENABLE_WANDB"
echo "  Push to Hub: $PUSH_TO_HUB"
if [ -n "$HF_REPO_ID" ]; then
    echo "  Target Repo: $HF_REPO_ID"
fi
echo ""

log_step "Training command built successfully"

# ============================================================================
# Training Execution
# ============================================================================

print_header "Starting Training"

log_step "Starting Pi0.5 training on $DATASET_REPO"
echo "Training will be logged to: $LOG_FILE"
echo ""

# Set environment variables
export HF_TOKEN
export WANDB_API_KEY
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Record start time
START_TIME=$(date +%s)
log_step "Training started at $(date '+%Y-%m-%d %H:%M:%S')"

# Execute training
if eval "$TRAIN_CMD" | tee -a "$LOG_FILE"; then
    TRAIN_SUCCESS=true
else
    TRAIN_SUCCESS=false
fi

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""

if [ "$TRAIN_SUCCESS" = true ]; then
    print_success "Training completed successfully!"
    log_step "Training completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
else
    print_error "Training failed! Check $LOG_FILE for details"
    log_step "Training failed after ${HOURS}h ${MINUTES}m ${SECONDS}s"
    exit 1
fi

# ============================================================================
# Post-training
# ============================================================================

print_header "Post-Training Steps"

# Create summary
SUMMARY_FILE="$OUTPUT_DIR/training_summary.txt"

# Build multi-GPU section if applicable
MULTI_GPU_INFO=""
if [ "$ENABLE_MULTI_GPU" = "true" ]; then
    MULTI_GPU_INFO="
Multi-GPU Configuration:
  Number of GPUs: $NUM_GPUS
  Mixed Precision: $MIXED_PRECISION
  Batch Size per GPU: $BATCH_SIZE
  Effective Batch Size: $EFFECTIVE_BATCH_SIZE
  Learning Rate (scaled): $EFFECTIVE_LR
  Steps (adjusted): $EFFECTIVE_STEPS
"
else
    MULTI_GPU_INFO="
Training Configuration:
  Batch Size: $BATCH_SIZE
  Learning Rate: $LEARNING_RATE
  Steps: $STEPS
"
fi

cat > "$SUMMARY_FILE" << SUMMARY_END
╔════════════════════════════════════════════════════════════════╗
║           Pi0.5 Training Summary                              ║
╚════════════════════════════════════════════════════════════════╝

Training Details:
  Dataset: $DATASET_REPO
  Base Model: $BASE_MODEL
  Target Repo: ${HF_REPO_ID:-"(not pushed)"}
$MULTI_GPU_INFO
Results:
  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s
  Output Directory: $OUTPUT_DIR
  Checkpoints: $OUTPUT_DIR/checkpoints
  Logs: $LOG_FILE

Final Model Location: $OUTPUT_DIR/final

Wandb Tracking: ${ENABLE_WANDB}
Pushed to Hub: ${PUSH_TO_HUB}

Next Steps:
  - Check metrics: tail -f $LOG_FILE
  - View model: ls $OUTPUT_DIR
  - Resume training: export RESUME_FROM_CHECKPOINT=true CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints/step_*

SUMMARY_END

cat "$SUMMARY_FILE"
log_step "Summary written to $SUMMARY_FILE"
print_success "Training summary created"

# List output files
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR" | grep -v "^total" | awk '{print "  "$9" ("$5")"}'

# ============================================================================
# Success message
# ============================================================================

print_header "Training Complete"

if [ "$PUSH_TO_HUB" = "true" ] && [ -n "$HF_REPO_ID" ]; then
    print_success "Model pushed to: https://huggingface.co/$HF_REPO_ID"
fi

print_success "Training logs: $LOG_FILE"
print_success "Checkpoints: $OUTPUT_DIR/checkpoints"

echo ""
echo "To resume training if interrupted:"
echo "  export RESUME_FROM_CHECKPOINT=true"
echo "  export CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints/step_*"
echo "  bash train_pi05_speedrun.sh"
echo ""
