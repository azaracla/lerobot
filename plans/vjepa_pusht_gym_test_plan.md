# V-JEPA + gym-pusht Integration Plan

> **TL;DR**: Pour tester V-JEPA sur Pusht (tâche simple), un petit action-conditioned head (~1-10M params) suffit, PAS les 300M du full V-JEPA 2-AC.

---

## Contexte

### Pourquoi Pusht ?

| Caractéristique | DROID (V-JEPA 2-AC full) | Pusht |
|----------------|---------------------------|-------|
| Action dim | 8D (7 joints + gripper) | **2D** (dx, dy, dtheta) |
| Complexity | Haute ( bras 7DoF) | **Basse** (pushing planar) |
| Full Head size | ~300M | **~1-10M** suffisent |

### Architecture V-JEPA Simplifiée pour Pusht

```
┌─────────────────────────────────────────────────────────────┐
│  V-JEPA Backbone (frozen)                                  │
│  Extrait z_t depuis images                                 │
│  ~600M params (ViT-G) ou ~300M (ViT-L)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ z_t
┌─────────────────────────────────────────────────────────────┐
│  Action-Conditioned Head (LIGHT)                          │
│  - Input: (a_t, s_t, z_t)                                 │
│  - Output: z_{t+1} prediction                             │
│  - Size: ~1-10M params (MLP ou petit Transformer)         │
└─────────────────────────────────────────────────────────────┘
                            ↓ CEM Planner
                     Action 2D (dx, dy, dtheta)
```

---

## Ce qu'il faut implémenter

### 1. VJEPAPolicy (LeRobot Policy)

Structure requise:
```
src/lerobot/policies/vjepa/
├── __init__.py
├── configuration_vjepa.py      # Config class
├── modeling_vjepa.py           # V-JEPA encoder + light head
└── processor_vjepa.py          # Pre/post processors
```

### 2. Configuration

```python
@dataclass
class VJEPAPushtConfig(PreTrainedConfig):
    policy_name: str = "vjepa"
    
    # Backbone (from vjepa2)
    pretrained_path: str = "/path/to/vitl.pt"  # ou vitg.pt
    model_name: str = "vit_large"  # ou vit_giant
    
    # Light head pour Pusht
    head_hidden_dim: int = 256
    head_num_layers: int = 3
    action_dim: int = 2  # Pusht: dx, dy, dtheta
    
    # CEM Planner
    cem_num_samples: int = 1000
    cem_num_iterations: int = 5
    cem_elite_fraction: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
```

### 3. Modèle

```python
class VJEPAPushtPolicy(PreTrainedPolicy):
    """V-JEPA avec light head pour Pusht."""
    
    def __init__(self, config: VJEPAPushtConfig):
        # 1. Charger V-JEPA backbone (frozen)
        self.backbone = load_vjepa_backbone(config.pretrained_path)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Light action-conditioned head
        self.head = ActionConditionedHead(
            action_dim=config.action_dim,
            hidden_dim=config.head_hidden_dim,
            num_layers=config.head_num_layers,
        )
        
        # 3. CEM planner pour inférence
        self.cem = CEMPlanner(
            num_samples=config.cem_num_samples,
            num_iterations=config.cem_num_iterations,
            elite_fraction=config.cem_elite_fraction,
        )
        
    def forward(self, batch):
        # Training: teacher forcing loss
        z_t = self.backbone(batch["observation"])  # frozen features
        z_next_pred = self.head(
            action=batch["action"],
            state=batch.get("observation.state", None),
            z=z_t,
        )
        return z_next_pred
        
    @torch.no_grad()
    def predict_action_chunk(self, batch):
        # Inférence: CEM planning
        z_t = self.backbone(batch["observation"])
        goal_z = batch.get("goal_observation", z_t)  # si goal fourni
        
        # Échantillonner actions candidates
        best_actions = self.cem.plan(
            goal_z, z_t, self.head, action_dim=2
        )
        return best_actions.unsqueeze(0)  # (1, horizon, 2)
```

---

## Étapes de Test dans gym-pusht

### Étape 1: Setup Environment

```bash
# Installer gym-pusht
pip install "gym-pusht>=0.1.5,<0.2.0" "pymunk>=6.6.0,<7.0.0"

# Vérifier que ça marche
python -c "
import gym_pusht
env = gym.make('PushT-v0')
print('Action space:', env.action_space)
print('Observation space:', env.observation_space)
obs, info = env.reset()
print('Obs keys:', obs.keys() if hasattr(obs, 'keys') else obs)
"
```

### Étape 2: Intégrer V-JEPA Policy

Créer `src/lerobot/policies/vjepa/` avec les fichiers ci-dessus.

### Étape 3: Test Évaluation

```bash
# Évaluer sur Pusht
lerobot-eval \
    --policy.path=outputs/vjepa_pusht/checkpoints/latest \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=50 \
    --policy.use_amp=false \
    --policy.device=cuda
```

### Étape 4: Baseline Comparison

Comparer avec les policies existantes:

| Policy | Expected Success Rate |
|--------|----------------------|
| Diffusion (baseline) | ~80-90% |
| V-JEPA + Light Head | À déterminer |

---

## Design de la Light Head

### Option A: MLP Simple (~1-5M params)

```python
class MLPActionHead(nn.Module):
    def __init__(self, action_dim, z_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        # Input: action + state (optional) + z_t features
        input_dim = action_dim + z_dim  # + state_dim if available
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, z_dim))  # Output: next z prediction
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, action, state, z):
        # z shape: (B, T, spatial, feature_dim) -> flatten to (B*T, feature_dim)
        B, T = z.shape[0], z.shape[1]
        z_flat = z.flatten(2).transpose(1, 2)  # (B, T*spatial, feat)
        
        # Concatenate action and z
        action_expanded = action.unsqueeze(1).expand(-1, z_flat.shape[1], -1)
        x = torch.cat([action_expanded, z_flat], dim=-1)
        
        # Flatten time dimension for MLP
        x = x.flatten(0, 1)  # (B*T*spatial, input_dim)
        out = self.mlp(x)
        
        # Reshape back
        out = out.view(B, T, z.shape[2], -1)  # (B, T, spatial, z_dim)
        return out.mean(dim=2)  # (B, T, z_dim)
```

### Option B: Small Transformer (~5-10M params)

```python
class TransformerActionHead(nn.Module):
    def __init__(self, action_dim, z_dim, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.z_embed = nn.Linear(z_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(hidden_dim, z_dim)
        
    def forward(self, action, state, z):
        # z: (B, T, spatial, z_dim) -> (B*T, spatial, z_dim)
        B, T, S, D = z.shape
        z = z.flatten(1, 2)  # (B, T*S, D)
        
        # Action: (B, T, action_dim) -> expand to (B, T*S, action_dim)
        action_expanded = action.unsqueeze(2).expand(-1, -1, S, -1).flatten(1, 2)
        
        # Embed and combine
        z_emb = self.z_embed(z)
        a_emb = self.action_embed(action_expanded)
        x = z_emb + a_emb  # additive conditioning
        
        # Transformer
        x = self.transformer(x)
        
        # Predict next z
        z_next = self.predictor(x)
        return z_next.view(B, T, S, D).mean(dim=2)
```

---

## Données pour Pusht

Dataset: `lerobot/pusht` (sur HuggingFace)

```bash
# Vérifier le dataset
python -c "
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset('lerobot/pusht')
print('Action dim:', ds.meta.features['action']['shape'])
print('Observation keys:', [k for k in ds.meta.features.keys() if 'image' in k])
"
```

---

## Résumé des Étapes

| Étape | Description | Complexité |
|-------|-------------|------------|
| 1 | Setup gym-pusht et vérifier installation | Basse |
| 2 | Créer structure `lerobot_policy_vjepa/` | Moyenne |
| 3 | Implémenter VJEPAPushtPolicy avec backbone gelé | Haute |
| 4 | Implémenter light head (MLP ou Transformer) | Moyenne |
| 5 | Implémenter CEM planner | Moyenne |
| 6 | Test training sur lerobot/pusht | Haute |
| 7 | Évaluation et comparaison avec baseline | Basse |

---

## Questions Ouvertes

1. **Goal image**: Comment fournir la goal image au CEM planner dans LeRobot ?
2. **Temporal sequence**: Faut-il des clips de plusieurs frames (16 frames @ 4FPS) ?
3. **Fine-tuning backbone**: Faut-il fine-tuner le backbone ou garder gelé ?

---

## Resources

- [gym-pusht GitHub](https://github.com/hrabric/gym-pusht)
- [V-JEPA 2 Paper](https://arxiv.org/abs/2506.09985)
- [LeRobot Policy Interface](./bring_your_own_policies.mdx)
