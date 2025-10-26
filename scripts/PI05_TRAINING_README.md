# 🚀 π₀.₅ Automated Training Suite for Vast.ai

Complete turnkey solution for training π₀.₅ on `azaracla/smolvla_3dprint_plate` dataset.

Inspired by [Karpathy's speedrun.sh](https://github.com/karpathy/nanochat/blob/master/speedrun.sh) - just a single script to run!

---

## 📦 What You Get

| File | Purpose | When to Use |
|------|---------|-----------|
| **`train_pi05_speedrun.sh`** | Main training script (bash) | Core training execution |
| **`pi05_quick_start.sh`** | Interactive setup wizard | First time setup |
| **`launch_on_vastai.sh`** | Vast.ai launcher template | Running on cloud |
| **`train_pi05_utils.py`** | Dataset validation & config | Advanced customization |
| **`train_pi05_resume.py`** | Checkpoint management | Resume/cleanup checkpoints |
| **`pi05_speedrun_config.yaml`** | Configuration reference | Understanding all parameters |
| **`PI05_SPEEDRUN_GUIDE.md`** | Complete documentation | Detailed reference |
| **`PI05_TRAINING_README.md`** | This file | Quick overview |

---

## ⚡ Quick Start (3 Steps)

### 1️⃣ **Set Your API Tokens**

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"           # https://huggingface.co/settings/tokens
export WANDB_API_KEY="YOUR_WANDB_KEY_HERE"     # https://wandb.ai/authorize
export HF_REPO_ID="your_username/pi05_model"   # https://huggingface.co/new
```

### 2️⃣ **Run Interactive Setup** (Recommended for first time)

```bash
bash pi05_quick_start.sh
```

This will ask you for tokens and training preferences interactively.

### 3️⃣ **Or Run Training Directly**

```bash
bash train_pi05_speedrun.sh
```

That's it! Training will:
- ✓ Validate your dataset and GPU
- ✓ Download and fine-tune π₀.₅ base model
- ✓ Log metrics to Wandb in real-time
- ✓ Save checkpoints every 500 steps
- ✓ Push final model to HuggingFace Hub
- ✓ Generate training summary

---

## 🎯 Three Ways to Train

### **Option 1: Local Machine (Simplest)**

```bash
# Just run one command
export HF_TOKEN="..." WANDB_API_KEY="..." HF_REPO_ID="..."
bash train_pi05_speedrun.sh
```

### **Option 2: Interactive Setup**

```bash
# Let the wizard guide you
bash pi05_quick_start.sh
```

### **Option 3: Vast.ai Cloud (Recommended)**

1. Edit `launch_on_vastai.sh` with your tokens
2. Copy the script to Vast.ai instance
3. Run: `bash launch_on_vastai.sh`

---

## 🔧 Configuration

All configuration via **environment variables**:

```bash
# Dataset
export DATASET_REPO="azaracla/smolvla_3dprint_plate"
export BASE_MODEL="lerobot/pi05_base"

# Training
export STEPS=3000                    # How many steps
export BATCH_SIZE=32                 # Batch size per GPU
export LEARNING_RATE=1e-4            # Learning rate
export WARMUP_STEPS=100              # Warmup steps

# Output
export OUTPUT_DIR="./outputs/pi05_speedrun"
export PUSH_TO_HUB=true
export HF_REPO_ID="username/model"

# Monitoring
export ENABLE_WANDB=true
export WANDB_API_KEY="..."

# Resume
export RESUME_FROM_CHECKPOINT=false
export CHECKPOINT_PATH=""
```

---

## 📊 Monitor Training

### Real-time Metrics (Wandb Dashboard)
```
https://wandb.ai/your_username/lerobot-pi05
```

### Local Logs
```bash
tail -f outputs/pi05_speedrun/training.log
```

### Checkpoint Status
```bash
python train_pi05_resume.py list outputs/pi05_speedrun
```

---

## ✅ Checkpoint Management

### List All Checkpoints
```bash
python train_pi05_resume.py list ./outputs/pi05_speedrun
```

### Resume from Latest
```bash
python train_pi05_resume.py resume ./outputs/pi05_speedrun
bash train_pi05_speedrun.sh
```

### Backup Important Checkpoint
```bash
python train_pi05_resume.py backup ./outputs/pi05_speedrun/checkpoints/step_003000 --name final_model
```

### Cleanup Old Checkpoints (Save Space)
```bash
python train_pi05_resume.py cleanup ./outputs/pi05_speedrun --keep 3
```

---

## 🛠️ Advanced Usage

### Validate Dataset Before Training
```bash
python train_pi05_utils.py validate azaracla/smolvla_3dprint_plate --hf-token $HF_TOKEN
```

### Generate Training Config
```bash
python train_pi05_utils.py config \
    --dataset azaracla/smolvla_3dprint_plate \
    --steps 5000 \
    --batch-size 16 \
    --output my_config.json
```

### Check System Info
```bash
python train_pi05_utils.py sysinfo
```

### Generate Vast.ai Script
```bash
python train_pi05_utils.py vastai --output my_vastai_launcher.sh
```

---

## 📈 Performance Expectations

| GPU | Batch | Time | Cost |
|-----|-------|------|------|
| H100 (80GB) | 64 | 1-2 hrs | $1-2 |
| A100 (80GB) | 32 | 2-3 hrs | $2-3 |
| RTX 4090 | 16 | 4-6 hrs | $3-4 |

**Costs on Vast.ai** - full 3000 step training

---

## 🚨 Troubleshooting

### GPU Out of Memory?
```bash
export BATCH_SIZE=16  # Reduce batch size
bash train_pi05_speedrun.sh
```

### Can't Access Dataset?
```bash
# Verify token and dataset access
python train_pi05_utils.py validate azaracla/smolvla_3dprint_plate --hf-token $HF_TOKEN
```

### Training Interrupted?
```bash
# Find latest checkpoint and resume
python train_pi05_resume.py latest ./outputs/pi05_speedrun
export RESUME_FROM_CHECKPOINT=true
bash train_pi05_speedrun.sh
```

### Permission Issues?
```bash
chmod +x train_pi05_speedrun.sh pi05_quick_start.sh launch_on_vastai.sh
```

See **`PI05_SPEEDRUN_GUIDE.md`** for complete troubleshooting.

---

## 📁 Output Structure

```
outputs/pi05_speedrun/
├── checkpoints/
│   ├── step_000500/        # Checkpoint
│   ├── step_001000/        # Checkpoint
│   └── step_003000/        # Final
├── training.log            # Raw logs
├── training_summary.txt    # Report
└── final/                  # Symlink to final checkpoint
```

---

## 📚 Full Documentation

See **`PI05_SPEEDRUN_GUIDE.md`** for:
- Complete setup instructions
- Advanced configuration
- System requirements
- Performance tuning
- Detailed troubleshooting
- Architecture notes
- Citation information

---

## 🎓 Example: End-to-End Training

```bash
# 1. Set tokens
export HF_TOKEN="hf_abc123..."
export WANDB_API_KEY="abc123..."
export HF_REPO_ID="myusername/pi05_smolvla"

# 2. Start training (automatic setup)
bash train_pi05_speedrun.sh

# Monitor in another terminal
tail -f outputs/pi05_speedrun/training.log

# Or watch on Wandb: https://wandb.ai/myusername/lerobot-pi05

# 3. After training completes
# Model automatically pushed to: https://huggingface.co/myusername/pi05_smolvla

# 4. Use your trained model
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
model = PI05Policy.from_pretrained("myusername/pi05_smolvla")
```

---

## 🔗 Key Resources

- **LeRobot Docs**: https://huggingface.co/docs/lerobot/
- **π₀.₅ Docs**: https://huggingface.co/docs/lerobot/pi05
- **Dataset**: https://huggingface.co/datasets/azaracla/smolvla_3dprint_plate
- **Vast.ai**: https://www.vast.ai/

---

## 💡 Tips & Tricks

- **First time?** Use `pi05_quick_start.sh` for interactive setup
- **Vast.ai?** Edit `launch_on_vastai.sh` and copy to cloud instance
- **Monitoring?** Use Wandb dashboard for real-time metrics
- **Interrupted?** Training is resumable from checkpoints
- **Save space?** Run `python train_pi05_resume.py cleanup` to keep last 3 checkpoints
- **Validate first?** Run `python train_pi05_utils.py validate` before training

---

## 📝 License

Apache 2.0 - Same as LeRobot

---

## ❓ Questions?

1. Check `PI05_SPEEDRUN_GUIDE.md` for comprehensive documentation
2. Review `outputs/pi05_speedrun/training.log` for error details
3. Open an issue on [LeRobot GitHub](https://github.com/huggingface/lerobot/issues)

---

**Ready to train? Start with:**
```bash
bash pi05_quick_start.sh
```

🚀 Happy training!

