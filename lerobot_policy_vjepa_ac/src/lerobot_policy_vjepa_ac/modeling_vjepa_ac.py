"""
Consolidate into LeRobot PretrainedPolicy class:
- VJepa AC model: vjepa2/src/models/ac_predictor.py
- VJepa AC training code: vjepa2/app/vjepa_droid/train.py
- VJepa AC inference: vjepa2/notebooks/utils/mpc_utils.py & world_model_wrapper.py
"""

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_vjepa_ac import VjepaAcConfig
from .ac_predictor_utils import VisionTransformerPredictorAC


from safetensors.torch import load_file, load_model
import os

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class VjepaAcPolicy(PreTrainedPolicy):
    name = "vjepa_ac"
    config_class = VjepaAcConfig

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict=True):
        """Override to clean _orig_mod. prefix from torch.compile keys."""
        from lerobot.policies.utils import log_model_loading_keys

        state_dict = load_file(model_file)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("predictor._orig_mod.", "predictor.")
            cleaned_state_dict[new_key] = v

        kwargs = {"strict": False}
        missing, unexpected = model.load_state_dict(cleaned_state_dict, **kwargs)
        log_model_loading_keys(missing, unexpected)
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
        return model

    def __init__(self, config: VjepaAcConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.dataset_stats = dataset_stats

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
        # max_seq_len must be a multiple of tubelet_size so that grid_depth = max_seq_len // tubelet_size
        # covers the longest sequence we'll ever pass (max of n_obs_steps and mpc_horizon+1 temporal positions).
        max_temporal_depth = max(config.n_obs_steps, config.mpc_horizon + 1)
        max_seq_len = max_temporal_depth * config.tubelet_size

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
        ).to(device, dtype=torch.bfloat16)

        # Pre-encode goal image if provided
        self.goal_latent = None
        if config.goal_image_path:
            import os

            goal_path = config.goal_image_path
            if not os.path.exists(goal_path):
                goal_path = os.path.join(os.path.dirname(__file__), config.goal_image_path)
            if os.path.exists(goal_path):
                self.goal_latent = self._encode_goal_image(goal_path)
                print(
                    f"[VJEPA_AC] Loaded goal image from {goal_path}, latent shape: {self.goal_latent.shape}"
                )
            else:
                print(f"[VJEPA_AC] WARNING: Goal image not found at {goal_path} or {config.goal_image_path}")

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

        mean = IMAGENET_MEAN.to(images.device, non_blocking=True).view(1, -1, 1, 1, 1).to(images.dtype)
        std = IMAGENET_STD.to(images.device, non_blocking=True).view(1, -1, 1, 1, 1).to(images.dtype)
        return (images - mean) / std

    def _encode_goal_image(self, path: str) -> torch.Tensor:
        """Load and encode a goal image once at init time."""
        img = PIL.Image.open(path).convert("RGB")
        img = img.resize((self.config.img_size, self.config.img_size), PIL.Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        device = next(self.encoder.parameters()).device
        img = img.to(device, non_blocking=True)
        img = self._imagenet_normalize(img)
        if self.config.tubelet_size == 2:
            img = img.repeat(1, 1, 2, 1, 1)  # [1, C, 1, H, W] → [1, C, 2, H, W]
        with torch.no_grad():
            latent = self.encoder(img)
        if getattr(self.config, "normalize_reps", False):
            latent = torch.nn.functional.layer_norm(latent, (latent.size(-1),))
        return latent

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Run one step of CEM to produce the best action for the current observation."""
        return self.select_action(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Model Predictive Control using Cross-Entropy Method (CEM).
        Encodes the current image and searches for the best action sequence.

        Matches VJEPA2 original CEM from mpc_utils.py and world_model_wrapper.py.
        """
        image_key = next(
            (k for k in batch if k.startswith("observation.image") and batch[k].ndim >= 4),
            None,
        )
        if image_key is None:
            raise KeyError("No observation.image* key found in batch")

        images = batch[image_key]
        states = batch["observation.state"]

        B = images.size(0)
        device = images.device

        with torch.no_grad():
            # images: [B, C, T, H, W] (ndim=5) or [B, C, H, W] (ndim=4)
            if images.ndim == 5:
                img_seq = images  # [B, C, T_obs, H, W]
            else:
                img_seq = images.unsqueeze(2)  # [B, C, 1, H, W]

            B_s, C, T_obs, H_img, W_img = img_seq.shape

            if H_img != self.config.img_size or W_img != self.config.img_size:
                img_seq = torch.nn.functional.interpolate(
                    img_seq.view(B_s * T_obs, C, H_img, W_img),
                    size=(self.config.img_size, self.config.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).view(B_s, C, T_obs, self.config.img_size, self.config.img_size)

            img_seq = self._imagenet_normalize(img_seq)

            # Encode each frame independently: [B*T_obs, C, 1, H, W] → [B*T_obs, tokens, D]
            img_flat = img_seq.permute(0, 2, 1, 3, 4).reshape(
                B_s * T_obs, C, 1, self.config.img_size, self.config.img_size
            )
            if self.config.tubelet_size == 2:
                img_flat = img_flat.repeat(1, 1, 2, 1, 1)  # match encoder pre-training tubelet_size=2
            with torch.amp.autocast("cuda", enabled=img_flat.is_cuda, dtype=torch.bfloat16):
                latents_flat = self.encoder(img_flat)  # [B*T_obs, tokens, D]
            latents_flat = latents_flat.to(torch.float32)
            tokens_per_frame = latents_flat.size(1)
            D = latents_flat.size(-1)

            # Full multi-frame context: [B, T_obs*tokens, D]
            context_latent = latents_flat.view(B_s, T_obs, tokens_per_frame, D).flatten(1, 2)

        goal_latent = self.goal_latent
        if goal_latent is None:
            raise RuntimeError(
                "goal_latent is None: encode a goal image with set_goal() before calling select_action."
            )

        # Historical actions from state differences (Droid convention: a_t = s_{t+1} - s_t)
        # states: [B, T_obs, state_dim] or [B, state_dim]
        # states are already normalized by the processor (MIN_MAX), no need to re-normalize
        if states.ndim == 3:
            hist_states = states                              # [B, T_obs, state_dim]
            hist_actions = states[:, 1:] - states[:, :-1]    # [B, T_obs-1, action_dim]
        else:
            hist_states = states.unsqueeze(1)
            hist_actions = torch.zeros(B, 0, self.config.action_dim, device=device)

        N = self.config.cem_num_samples
        H = self.config.mpc_horizon
        top_k = max(2, int(N * self.config.cem_elite_ratio))
        loss_exp = getattr(self.config, "loss_exp", 1.0)
        momentum_mean = getattr(self.config, "cem_momentum_mean", 0.25)
        momentum_std = getattr(self.config, "cem_momentum_std", 0.95)
        normalize_reps = getattr(self.config, "normalize_reps", False)

        # B=1 in robot inference — process batch dim directly without loop
        c_lat = context_latent  # [B, T_obs*tokens, D]
        if normalize_reps:
            c_lat = torch.nn.functional.layer_norm(c_lat, (D,))

        # Expand to N samples (assumes B=1; for B>1 this takes first element)
        c_lat_N = c_lat[0:1].expand(N, -1, -1)           # [N, T_obs*tokens, D]
        h_act_N = hist_actions[0:1].expand(N, -1, -1)    # [N, T_obs-1, action_dim]
        s_traj0 = hist_states[0:1].expand(N, -1, -1)     # [N, T_obs, state_dim]
        g_lat = goal_latent[0:1].expand(N, -1, -1)       # [N, tokens, D]

        mu = torch.zeros(H, self.config.action_dim, device=device)
        std = torch.full((H, self.config.action_dim), self.config.cem_std, device=device)

        for _ in range(self.config.cem_num_iters):
            actions = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(N, H, self.config.action_dim, device=device)

            # Step 0: full T_obs-frame context (matches training), causal attention active
            a_traj0 = torch.cat([h_act_N, actions[:, :1]], dim=1)  # [N, T_obs, action_dim]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred0 = self.predictor(c_lat_N, a_traj0, s_traj0)
            current_z = pred0[:, -tokens_per_frame:].to(torch.float32)
            if normalize_reps:
                current_z = torch.nn.functional.layer_norm(current_z, (D,))

            # Steps 1..H-1: rolling T=1 context (FlashAttention active)
            _s = actions[:, :1]
            for h in range(1, H):
                _a = actions[:, h : h + 1]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred_h = self.predictor(current_z, _a, _s)
                current_z = pred_h[:, -tokens_per_frame:].to(torch.float32)
                if normalize_reps:
                    current_z = torch.nn.functional.layer_norm(current_z, (D,))
                _s = _a

            costs = torch.mean(torch.abs(current_z - g_lat) ** loss_exp, dim=(1, 2))
            _, elite_inds = torch.topk(costs, top_k, largest=False)
            elites = actions[elite_inds]

            new_mu = elites.mean(dim=0)
            new_std = elites.std(dim=0) + 1e-5
            mu = new_mu * (1.0 - momentum_mean) + mu * momentum_mean
            std = new_std * (1.0 - momentum_std) + std * momentum_std

        # Return [B, H, action_dim] — mu is the best action trajectory
        return mu.unsqueeze(0).expand(B, -1, -1)

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
        # actions are delta values a_k = s_{k+1} - s_k when use_delta_actions=True (preprocessor)
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
                if self.config.tubelet_size == 2:
                    images = images.repeat(1, 1, 2, 1, 1)  # [B, C, 2, H, W]

                with torch.amp.autocast("cuda", enabled=images.is_cuda, dtype=torch.bfloat16):
                    z = self.encoder(images)
                target_latents = z.unsqueeze(1)  # [B, 1, N, D]
            else:
                # Multi-frame case: [B, T, C, H, W] from LeRobot, needs to be [B, C, T, H, W]
                B, T, C, H, W = images.shape
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                images = self._imagenet_normalize(images)

                # Vectorized encoding: each frame independently as [B*T, C, 1, H, W]
                images_flat = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, 1, H, W)
                if self.config.tubelet_size == 2:
                    images_flat = images_flat.repeat(1, 1, 2, 1, 1)  # match encoder pre-training tubelet_size=2
                with torch.amp.autocast("cuda", enabled=images.is_cuda, dtype=torch.bfloat16):
                    latents_flat = self.encoder(images_flat)
                target_latents = latents_flat.reshape(B, T, -1, latents_flat.shape[-1])

        # Ensure target_latents is back to Float32 for the predictor which is Float32
        target_latents = target_latents.to(torch.float32)

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

        _autocast = torch.amp.autocast("cuda", enabled=images.is_cuda, dtype=torch.bfloat16)

        def _step_predictor(_z, _a, _s, _e=None):
            with _autocast:
                if self.config.use_extrinsics and _e is not None:
                    _pred = self.predictor(_z, _a, _s, _e)
                else:
                    _pred = self.predictor(_z, _a, _s)

            _pred = _pred.to(torch.float32)
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

        # Build per-step valid mask from action_is_pad if available: [B, n_steps, tokens, D]
        valid_mask = None
        if "action_is_pad" in batch:
            n_steps = n_action_steps
            pad = batch["action_is_pad"][:, :n_steps]  # [B, n_steps]
            valid = ~pad  # True = valid step
            valid_mask = (
                valid.unsqueeze(-1).unsqueeze(-1)
                .expand(-1, -1, tokens_per_frame, target_latents.size(-1))
                .reshape(B, n_steps * tokens_per_frame, target_latents.size(-1))
                .to(target_latents.dtype)
            )

        def loss_fn(z_pred, target, mask=None):
            _target = target[:, tokens_per_frame : z_pred.size(1) + tokens_per_frame]
            err = torch.abs(z_pred - _target) ** loss_exp / loss_exp
            if mask is not None:
                # Slice mask to match z_pred length (z_ar may be shorter than z_tf)
                mask_slice = mask[:, : err.size(1), :]
                if mask_slice.shape == err.shape:
                    return (err * mask_slice).sum() / mask_slice.sum().clamp(min=1)
            return err.mean()

        jloss = loss_fn(z_tf, target_latents, valid_mask)
        sloss = loss_fn(z_ar, target_latents, valid_mask)
        loss = jloss + sloss

        return loss, {"loss": loss.item(), "jloss": jloss.item(), "sloss": sloss.item()}
