"""
Consolidate into LeRobot PretrainedPolicy class:
- VJepa AC model: vjepa2/src/models/ac_predictor.py
- VJepa AC training code: vjepa2/app/vjepa_droid/train.py
- VJepa AC inference: vjepa2/notebooks/utils/mpc_utils.py & world_model_wrapper.py
"""

import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_vjepa_ac import VjepaAcConfig
from .ac_predictor_utils import VisionTransformerPredictorAC


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class VjepaAcPolicy(PreTrainedPolicy):
    name = "vjepa_ac"
    config_class = VjepaAcConfig

    def __init__(self, config: VjepaAcConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        self.config = config

        # Load the frozen video encoder from PyTorch Hub
        device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        with torch.device(device):
            encoder_output = torch.hub.load(config.encoder_repo_id, config.model_name)

        if isinstance(encoder_output, tuple):
            self.encoder = encoder_output[0]
        else:
            self.encoder = encoder_output

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Initialize the Action-Conditioned Predictor
        max_seq_len = max(config.num_frames, config.mpc_horizon + 1)

        encoder_embed_dim = self.encoder.embed_dim if hasattr(self.encoder, "embed_dim") else config.embed_dim
        self.predictor = VisionTransformerPredictorAC(
            img_size=(config.img_size, config.img_size),
            patch_size=config.patch_size,
            num_frames=max_seq_len,
            tubelet_size=config.tubelet_size,
            embed_dim=encoder_embed_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            action_embed_dim=config.action_dim,
            depth=config.pred_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        ).to(device)

    # --- Required abstract method implementations ---

    def get_optim_params(self) -> dict:
        """Only the AC predictor is trained; the encoder is frozen."""
        return self.predictor.parameters()

    def reset(self):
        """No stateful cache needed for this policy (no action chunking)."""
        pass

    def _imagenet_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to images.

        The vjepa2 encoder was pretrained with ImageNet normalization:
        - mean = (0.485, 0.456, 0.406)
        - std = (0.229, 0.224, 0.225)

        This assumes input images are in [0, 1] range.
        """
        if not getattr(self.config, "use_imagenet_for_visuals", True):
            return images

        mean = IMAGENET_MEAN.to(images.device).view(1, -1, 1, 1, 1)
        std = IMAGENET_STD.to(images.device).view(1, -1, 1, 1, 1)
        return (images - mean) / std

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Run one step of CEM to produce the best action for the current observation."""
        return self.select_action(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Model Predictive Control using Cross-Entropy Method (CEM).
        Encodes the current image and searches for the best action sequence.
        """
        # Find the observation image key dynamically
        image_key = next(
            (k for k in batch if k.startswith("observation.image") and batch[k].ndim >= 4),
            None,
        )
        if image_key is None:
            raise KeyError("No observation.image* key found in batch")

        images = batch[image_key]  # [B, C, T, H, W] or [B, C, H, W]
        states = batch["observation.state"]  # [B, T, D] or [B, D]

        B = images.size(0)
        device = images.device

        with torch.no_grad():
            if images.ndim == 5:
                img_seq = images[:, :, -1:]
            else:
                img_seq = images.unsqueeze(2)
            img_seq = self._imagenet_normalize(img_seq)
            current_latent = self.encoder(img_seq)  # [B, N_patches, D]

        # Extract goal image if present (fallback to dummy zeros if not goal-conditioned)
        goal_key = next((k for k in batch if "goal" in k and "image" in k), None)
        with torch.no_grad():
            if goal_key is not None:
                goal_img = batch[goal_key].unsqueeze(2) if batch[goal_key].ndim == 4 else batch[goal_key]
                goal_img = self._imagenet_normalize(goal_img)
                goal_latent = self.encoder(goal_img)  # [B, N, D]
            else:
                goal_latent = torch.zeros(B, current_latent.size(1), current_latent.size(-1), device=device)

            if getattr(self.config, "normalize_reps", False):
                goal_latent = torch.nn.functional.layer_norm(goal_latent, (goal_latent.size(-1),))

        # CEM parameters from config
        N = self.config.cem_num_samples
        H = self.config.mpc_horizon
        top_k = max(2, int(N * self.config.cem_elite_ratio))  # ensure at least 2 for std

        momentum_mean = getattr(self.config, "cem_momentum_mean", 0.25)
        momentum_std = getattr(self.config, "cem_momentum_std", 0.95)
        maxnorm = getattr(self.config, "cem_maxnorm", 0.05)

        best_first_actions = []

        for b in range(B):
            # mu and std for action trajectory [H, action_dim]
            mu = torch.zeros(H, self.config.action_dim, device=device)
            std = torch.full((H, self.config.action_dim), self.config.cem_std, device=device)

            c_latent = current_latent[b : b + 1]  # [1, N_patches, D]
            if getattr(self.config, "normalize_reps", False):
                c_latent = torch.nn.functional.layer_norm(c_latent, (c_latent.size(-1),))

            init_state = states[b : b + 1]  # [1, T, D] or [1, D]
            if init_state.ndim == 3:
                init_state = init_state[:, -1]  # [1, D]
            elif init_state.ndim == 1:
                init_state = init_state.unsqueeze(0)

            tokens_per_frame = c_latent.size(1)

            for _ in range(self.config.cem_num_iters):
                # 1. Sample trajectories: [N, H, action_dim]
                # actions are deltas in vjepa
                eps = torch.randn(N, H, self.config.action_dim, device=device)
                actions = mu.unsqueeze(0) + std.unsqueeze(0) * eps

                # 2. Clip actions (vjepa style)
                actions[..., :3] = torch.clamp(actions[..., :3], -maxnorm, maxnorm)
                # Gripper clip (vjepa uses -0.75 to 0.75)
                actions[..., -1:] = torch.clamp(actions[..., -1:], -0.75, 0.75)

                # 3. Batched autoregressive rollout
                z = c_latent.expand(N, -1, -1)
                curr_s = init_state.expand(N, -1)

                for h_step in range(H):
                    _a = actions[:, : h_step + 1]
                    # Iterative state integration: state_next = state_curr + action_curr
                    # Note: Original uses Rotation matrices but simple addition is often used
                    # for small delta eulers in many RL contexts.
                    # To be perfectly compliant with pose integration:
                    _s_seq = []
                    _temp_s = init_state.expand(N, -1)
                    for i in range(h_step + 1):
                        _s_seq.append(_temp_s.unsqueeze(1))
                        _temp_s = _temp_s + actions[:, i]

                    _s_seq = torch.cat(_s_seq, dim=1)  # [N, h_step+1, D]

                    pred = self.predictor(z, _a, _s_seq)
                    if getattr(self.config, "normalize_reps", False):
                        pred = torch.nn.functional.layer_norm(pred, (pred.size(-1),))

                    # Update context with the last prediction for next step
                    if h_step < H - 1:
                        z = torch.cat([z, pred[:, -tokens_per_frame:]], dim=1)
                    else:
                        final_latent = pred[:, -tokens_per_frame:]

                # 4. Evaluation (Cost = distance to goal)
                g_lat = goal_latent[b : b + 1].expand(N, -1, -1)
                costs = torch.mean(
                    torch.abs(final_latent - g_lat) ** getattr(self.config, "loss_exp", 1.0), dim=(1, 2)
                )

                # 5. Selection
                _, elite_inds = torch.topk(costs, top_k, largest=False)
                elites = actions[elite_inds]

                # 6. Momentum-based distribution update
                new_mu = elites.mean(dim=0)
                new_std = elites.std(dim=0) + 1e-5

                mu = new_mu * (1.0 - momentum_mean) + mu * momentum_mean
                std = new_std * (1.0 - momentum_std) + std * momentum_std

            best_first_actions.append(mu[0])

        return torch.stack(best_first_actions)  # [B, action_dim]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for training: predict future latents from context and compute L1 loss
        against the frozen encoder's target latents.
        """
        image_key = next(
            (k for k in batch if k.startswith("observation.image") and batch[k].ndim >= 4),
            None,
        )
        if image_key is None:
            raise KeyError(f"No observation.image* key found in batch. Available keys: {list(batch.keys())}")

        images = batch[image_key]

        images = batch[image_key]  # [B, C, T, H, W]
        actions = batch["action"]  # [B, T-1, D]
        states = batch["observation.state"]  # [B, T, D] or [B, D]

        # Encode all frames with the frozen encoder
        with torch.no_grad():
            if images.ndim not in (4, 5):
                raise ValueError(
                    f"images tensor for key '{image_key}' has shape {images.shape}, expected 4D or 5D"
                )

            if images.ndim == 4:
                images = images.unsqueeze(2)  # [B, C, 1, H, W]
                images = self._imagenet_normalize(images)
   
                z = self.encoder(images)
                target_latents = z.unsqueeze(1)  # [B, 1, N, D]
            else:
                # Multi-frame case: [B, T, C, H, W] from LeRobot, needs to be [B, C, T, H, W]
                B, T, C, H, W = images.shape
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                latents = []
                images = self._imagenet_normalize(images)

                for t in range(T):
                    z = self.encoder(images[:, :, t : t + 1, :, :])
                    latents.append(z)
                target_latents = torch.stack(latents, dim=1)  # [B, T, N_patches, D]

        if target_latents.abs().max() < 1e-6:
            print(
                f"WARNING: target_latents is near zero! shape={target_latents.shape}, max={target_latents.abs().max()}"
            )

        T_full = target_latents.shape[1]
        if T_full < 2:
            dummy_loss = sum(p.sum() for p in self.predictor.parameters()) * 0.0
            return dummy_loss, {"loss": dummy_loss.item()}

        target_latents = target_latents.flatten(1, 2)  # [B, T*N, D]
        if getattr(self.config, "normalize_reps", False):
            target_latents = torch.nn.functional.layer_norm(target_latents, (target_latents.size(-1),))

        tokens_per_frame = target_latents.size(1) // T_full

        def _step_predictor(_z, _a, _s, _e=None):
            if self.config.use_extrinsics and _e is not None:
                _pred = self.predictor(_z, _a, _s, _e)
            else:
                _pred = self.predictor(_z, _a, _s)

            if getattr(self.config, "normalize_reps", False):
                _pred = torch.nn.functional.layer_norm(_pred, (_pred.size(-1),))
            return _pred

        if states.ndim == 2:
            states = states.unsqueeze(1).expand(-1, T_full, -1)

        auto_steps = min(getattr(self.config, "auto_steps", 1), T_full - 1)
        loss_exp = getattr(self.config, "loss_exp", 1.0)

        extrinsics = None  # extrinsics not provided in LeRobot batch currently

        # Use only T_full-1 actions to match states[:, :-1]
        # LeRobot provides actions for full MPC horizon, but states for n_obs_steps
        n_action_steps = min(actions.shape[1], T_full - 1)

        # -- teacher forcing (jloss)
        z_ctxt = target_latents[:, :-tokens_per_frame]
        z_tf = _step_predictor(z_ctxt, actions[:, :n_action_steps], states[:, :-1], extrinsics)

        # -- auto-regressive (sloss)
        _z = torch.cat([target_latents[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
        for n in range(1, auto_steps):
            _a, _s = actions[:, : n + 1], states[:, : n + 1]
            _e = extrinsics[:, : n + 1] if extrinsics is not None else None
            _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        z_ar = _z[:, tokens_per_frame:]

        def loss_fn(z_pred, target):
            _target = target[:, tokens_per_frame : z_pred.size(1) + tokens_per_frame]
            loss_val = torch.mean(torch.abs(z_pred - _target) ** loss_exp) / loss_exp
            return loss_val

        jloss = loss_fn(z_tf, target_latents)
        sloss = loss_fn(z_ar, target_latents)
        loss = jloss + sloss

        return loss, {"loss": loss.item(), "jloss": jloss.item(), "sloss": sloss.item()}
