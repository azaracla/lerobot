#!/usr/bin/env python
"""Script pour pousser le checkpoint pi05 vers le Hub"""

from pathlib import Path
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.train import TrainPipelineConfig

# Chemin vers votre checkpoint
checkpoint_path = Path("./outputs/pi05_speedrun_20251026_115842/checkpoints/003000/pretrained_model")

# Vérifier que le checkpoint existe
if not checkpoint_path.exists():
    print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
    print("Recherche de checkpoints disponibles...")
    import os
    for root, dirs, files in os.walk("./outputs"):
        if "model.safetensors" in files:
            print(f"  Trouvé: {root}")
    exit(1)

# Charger le policy depuis le checkpoint avec la classe concrète
print(f"Chargement du checkpoint depuis {checkpoint_path}")
policy = PI05Policy.from_pretrained(checkpoint_path)

# Charger les processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=checkpoint_path
)

# Configurer le repo Hub
policy.config.repo_id = "azaracla/pi05_3dprint_plate_3k"  # À adapter selon votre username
policy.config.private = False  # True si vous voulez un repo privé

# Charger la config d'entraînement
train_config_path = checkpoint_path / "train_config.json"
if train_config_path.exists():
    print("Chargement de train_config.json")
    cfg = TrainPipelineConfig.from_pretrained(checkpoint_path)
else:
    print("⚠️  train_config.json non trouvé, création d'une config minimale")
    from lerobot.configs.dataset import DatasetConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="azaracla/smolvla_3dprint_plate"),
        policy=policy.config
    )

# Push vers le Hub
print(f"🚀 Push vers le Hub: {policy.config.repo_id}")
print("   - Policy weights...")
policy.push_model_to_hub(cfg)

print("   - Preprocessor...")
preprocessor.push_to_hub(policy.config.repo_id)

print("   - Postprocessor...")
postprocessor.push_to_hub(policy.config.repo_id)

print("✅ Checkpoint poussé avec succès !")
print(f"   https://huggingface.co/{policy.config.repo_id}")
