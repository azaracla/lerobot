# Plan d'intégration V-JEPA 2.1 dans LeRobot

> **Objectif**: Utiliser V-JEPA 2.1 (frozen) comme encodeur visuel + réentraîner le predictor AC sur SO-101/SO-100.

---

## 1. Ressources et Sources

### LeRobot

| Ressource | URL/Path |
|-----------|----------|
| Documentation "Bring Your Own Policy" | https://huggingface.co/docs/lerobot/bring_your_own_policies |
| Base Policy Class | `/lerobot/src/lerobot/policies/pretrained.py` |
| Base Config Class | `/lerobot/src/lerobot/configs/policies.py` |
| Policy Factory | `/lerobot/src/lerobot/policies/factory.py` |
| ACT Policy Example | `/lerobot/src/lerobot/policies/act/` |
| SMOLVLA (VLM frozen) | `/lerobot/src/lerobot/policies/smolvla/` |
| SAC (freeze func) | `/lerobot/src/lerobot/policies/sac/` |
| X-VLA Training | `/lerobot/docs/source/xvla.mdx` |

### V-JEPA 2.1

| Ressource | URL/Path |
|-----------|----------|
| GitHub | https://github.com/facebookresearch/vjepa2 |
| README (models) | `/vjepa2/README.md` (lines 150-264) |
| Encoder (ViT) | `/vjepa2/src/models/vision_transformer.py` |
| AC Predictor | `/vjepa2/src/models/ac_predictor.py` |
| Hub Loading | `/vjepa2/src/hub/backbones.py` |
| Hubconf | `/vjepa2/hubconf.py` |

### Datasets SO-101/SO-100

| Dataset | Repo ID |
|---------|---------|
| SVLA SO-101 | `lerobot/svla_so101_pickplace` |
| SO-100 Handover | `<USER>/bimanual-so100-handover-cube` |
| SVLA SO-100 | `<USER>/svla_so100_task1_v3` |
| Docs SO-101 | `/lerobot/docs/source/so101.mdx` |
| Docs SO-100 | `/lerobot/docs/source/so100.mdx` |

---

## 2. Modèles V-JEPA 2.1 Disponibles

```python
# Via torch.hub
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384')      # 80M, 384
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')     # 300M, 384
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_giant_384')     # 1B, 384
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_gigantic_384')  # 2B, 384

# AC model (pour robotique)
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')  # 1B, 256, action-conditioned
```

---

## 3. Structure Plugin LeRobot

```
lerobot_policy_vjepa21/
├── pyproject.toml
└── src/
    └── lerobot_policy_vjepa21/
        ├── __init__.py
        ├── configuration_vjepa21.py
        ├── modeling_vjepa21.py
        └── processor_vjepa21.py
```

### Conventions de nommage (CRITIQUES)

- Config: `VJEPAPolicy21Config`
- Policy: `VJEPAPolicy21`
- Factory: `make_vjepa21_pre_post_processors()`
- Enregistrement: `@PreTrainedConfig.register_subclass("vjepa21")`

---

## 4. Implémentation

### 4.1 Configuration Class

**Pattern à suivre**: `/lerobot/src/lerobot/policies/act/configuration_act.py`

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("vjepa21")
@dataclass
class VJEPAPolicy21Config(PreTrainedConfig):
    # Model
    pretrained_path: str | None = None
    hub_name: str = "vjepa2_1_vit_giant_384"  # ou "vjepa2_ac_vit_giant"
    freeze_backbone: bool = True
    img_size: int = 384
    patch_size: int = 16
    
    # Action head
    action_dim: int = 7  # SO-101/SO-100: 7 DOF
    head_hidden_dim: int = 1024
    head_num_layers: int = 3
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    def validate_features(self):
        pass  # À implémenter
    
    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.learning_rate, weight_decay=self.weight_decay)
```

### 4.2 Policy Class

**Pattern à suivre**: `/lerobot/src/lerobot/policies/sac/modeling_sac.py` (freeze pattern)

```python
class VJEPAPolicy21(PreTrainedPolicy):
    config_class = VJEPAPolicy21Config
    name = "vjepa21"
    
    def __init__(self, config, dataset_stats=None):
        super().__init__(config, dataset_stats)
        
        # 1. Load V-JEPA 2.1 backbone via torch.hub
        self.encoder = torch.hub.load(
            'facebookresearch/vjepa2', 
            config.hub_name,
            map_location='cpu'
        )
        
        # 2. Optionnel: projection layer si needed
        self.projection = nn.Linear(encoder.embed_dim, config.feature_dim)
        
        # 3. Freeze backbone si config.freeze_backbone
        if config.freeze_backbone:
            self._freeze_encoder()
        
        # 4. Action head (MLP ou Transformer)
        self.action_head = MLPActionHead(
            input_dim=config.feature_dim,
            action_dim=config.action_dim,
            hidden_dim=config.head_hidden_dim,
            num_layers=config.head_num_layers,
        )
    
    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def get_optim_params(self) -> dict:
        """Retourne seulement les params trainables (action head)."""
        return [
            {
                "params": [p for n, p in self.named_parameters() 
                          if "encoder" not in n and p.requires_grad],
                "lr": self.config.learning_rate,
            }
        ]
    
    def forward(self, batch):
        # Training forward
        features = self._extract_features(batch)
        actions = self.action_head(features)
        loss = F.mse_loss(actions, batch["action"])
        return {"loss": loss}, {"action": actions}
    
    def predict_action_chunk(self, batch):
        # Inference
        features = self._extract_features(batch)
        return self.action_head(features)
    
    def select_action(self, batch):
        actions = self.predict_action_chunk(batch)
        return actions[:, 0]  # Première action du chunk
    
    def reset(self):
        pass
```

### 4.3 Processor Function

```python
def make_vjepa21_pre_post_processors(config, dataset_stats):
    """Process images for V-JEPA 2.1 input format."""
    preprocessor = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    def preprocess(observation):
        # observation.images.xxx -> [C, H, W] tensor
        # Ajouter dimension temporelle: [1, C, 1, H, W] (T=1 pour images)
        ...
    
    return preprocess, None  # postprocessor=None pour actions directes
```

---

## 5. Points d'attention

### 5.1 Input Format V-JEPA 2.1

V-JEPA 2.1 attend `[B, C, T, H, W]` (5D):
- Pour images: `T=1`
- Pour vidéos: `T=16` (comme dans le papier)

```python
# Shape transformation needed
image = tensor.unsqueeze(0).unsqueeze(2)  # [C, H, W] -> [1, C, 1, H, W]
```

### 5.2 Feature Extraction

L'encodeur retourne `[B, N, D]` où:
- `N = H*W / patch_size^2` (nombre de patches)
- `D = embed_dim` (1024 pour ViT-L, 1408 pour ViT-G)

Options pour l'action head:
1. **Global pooling**: Moyenne sur les patches → `[B, D]`
2. **Concat all patches**: Flatten → `[B, N*D]`
3. **Attention pooling**: Learnable attention weights

### 5.3 Robot DOF

| Robot | DOF | Action dim |
|-------|-----|------------|
| SO-100 | 6 + gripper | 7 |
| SO-101 | 6 + gripper | 7 |

### 5.4 Freeze Pattern Example

**Pattern SAC** (`/lerobot/src/lerobot/policies/sac/modeling_sac.py`):
```python
def freeze_image_encoder(image_encoder):
    for param in image_encoder.parameters():
        param.requires_grad = False
```

---

## 6. Training Command

```bash
# SO-101
lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=vjepa21 \
  --policy.hub_name=vjepa2_1_vit_large_384 \
  --policy.freeze_backbone=true \
  --policy.learning_rate=1e-4 \
  --output_dir=./outputs/vjepa21_so101 \
  --steps=50000

# SO-100 (bimanual)
lerobot-train \
  --dataset.repo_id=<USER>/bimanual-so100-handover-cube \
  --policy.type=vjepa21 \
  --policy.hub_name=vjepa2_1_vit_giant_384 \
  --policy.freeze_backbone=true \
  --policy.action_dim=14 \
  --output_dir=./outputs/vjepa21_so100 \
  --steps=50000
```

---

## 7. Alternative: Utiliser V-JEPA 2-AC Pré-entraîné

Le modèle `vjepa2_ac_vit_giant` est déjà pré-entraîné pour robotique:
- Encodeur: ViT-Giant (1B)
- Predictor: Action-conditioned (24 layers)
- Entraîné sur DROID

```python
encoder, predictor = torch.hub.load(
    'facebookresearch/vjepa2', 
    'vjepa2_ac_vit_giant',
    pretrained=True
)
```

Option: Fine-tune seulement le predictor AC sur SO-101/SO-100.

---

## 8. Vérifications avant implémentation

- [ ] Vérifier shape des images dans datasets SO-101/SO-100
- [ ] Vérifier `action` feature shape (doit être 7D)
- [ ] Vérifier que `torch.hub.load` fonctionne avec le bon hub_name
- [ ] Confirmer format input V-JEPA 2.1 (5D tensor)
- [ ] Tester feature extraction avec une image

---

## 9. Références Complètes

### LeRobot Base Classes
```python
# /lerobot/src/lerobot/policies/pretrained.py
class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    config_class: None
    name: None
    
    @abc.abstractmethod
    def get_optim_params(self) -> dict: ...
    @abc.abstractmethod
    def reset(self): ...
    @abc.abstractmethod
    def forward(self, batch) -> tuple[Tensor, dict]: ...
    @abc.abstractmethod
    def predict_action_chunk(self, batch) -> Tensor: ...
    @abc.abstractmethod
    def select_action(self, batch) -> Tensor: ...
```

### V-JEPA 2.1 Encoder Output
```python
# From /vjepa2/src/models/vision_transformer.py
class VisionTransformer(nn.Module):
    def forward(self, x):
        # x: [B, C, T, H, W] ou [B, C, H, W]
        # Return: [B, N, D] où N = spatial patches
```

### AC Predictor (pour robotique)
```python
# From /vjepa2/src/models/ac_predictor.py
class VisionTransformerPredictorAC(nn.Module):
    def forward(self, x, actions, states, extrinsics=None):
        # x: context tokens [B, N, D]
        # actions: [B, T, 7] 
        # states: [B, T, 7]
        # Return: [B, N_pred, D]
```
