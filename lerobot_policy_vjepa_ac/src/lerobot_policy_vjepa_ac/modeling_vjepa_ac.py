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

        # Disable torch.compile for inference to avoid shape issues with CEM
        # torch.compile is only useful for training speed, not needed for inference

        # Pre-encode goal image if provided
        self.goal_latent = None
        if config.goal_image_path:
            self.goal_latent = self._encode_goal_image(config.goal_image_path)

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
        with torch.no_grad():
            latent = self.encoder(img)
        if getattr(self.config, "normalize_reps", False):
            latent = torch.nn.functional.layer_norm(latent, (latent.size(-1),))
        return latent

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using MIN_MAX normalization to match training."""
        if self.dataset_stats is None:
            return state  # No normalization if no stats available

        obs_state = self.dataset_stats.get("observation.state", {})
        if not obs_state:
            return state

        state_min = torch.as_tensor(obs_state.get("min", None), device=state.device, dtype=state.dtype)
        state_max = torch.as_tensor(obs_state.get("max", None), device=state.device, dtype=state.dtype)

        if state_min is None or state_max is None:
            return state

        # MIN_MAX normalization: map [min, max] to [-1, 1]
        denom = state_max - state_min
        eps = 1e-8
        denom = torch.where(denom == 0, torch.tensor(eps, device=state.device), denom)
        return 2 * (state - state_min) / denom - 1

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Run one step of CEM to produce the best action for the current observation."""
        return self.select_action(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Model Predictive Control using Cross-Entropy Method (CEM).
        Encodes the current image and searches for the best action sequence.

        NOTE: This policy outputs normalized actions (MIN_MAX). The postprocessor
        will denormalize them to absolute joint positions for the robot.
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
            if images.ndim == 5:
                img_seq = images[:, :, -1:]
            else:
                img_seq = images.unsqueeze(2)

            # Resize to match encoder's expected size (img_size x img_size)
            # This fixes mismatch when robot sends different resolution images
            _, C, T, H, W = img_seq.shape
            if H != self.config.img_size or W != self.config.img_size:
                img_seq = torch.nn.functional.interpolate(
                    img_seq.view(B * T, C, H, W),
                    size=(self.config.img_size, self.config.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                img_seq = img_seq.view(B, C, T, self.config.img_size, self.config.img_size)

            img_seq = self._imagenet_normalize(img_seq)
            current_latent = self.encoder(img_seq)

        goal_latent = self.goal_latent
        if goal_latent is None:
            goal_latent = torch.zeros(B, current_latent.size(1), current_latent.size(-1), device=device)

        N = self.config.cem_num_samples
        H = self.config.mpc_horizon
        top_k = max(2, int(N * self.config.cem_elite_ratio))

        momentum_mean = getattr(self.config, "cem_momentum_mean", 0.25)
        momentum_std = getattr(self.config, "cem_momentum_std", 0.95)

        best_trajectories = []

        for b in range(B):
            c_latent = current_latent[b : b + 1]
            if getattr(self.config, "normalize_reps", False):
                c_latent = torch.nn.functional.layer_norm(c_latent, (c_latent.size(-1),))

            init_state = states[b : b + 1]
            if init_state.ndim == 3:
                init_state = init_state[:, -1]
            elif init_state.ndim == 1:
                init_state = init_state.unsqueeze(0)

            # Normalize state for CEM to match training (MIN_MAX normalization)
            # The predictor was trained with normalized states, so inference must match
            init_state = self._normalize_state(init_state)

            mu = init_state.clone().expand(H, -1)
            std = torch.full((H, self.config.action_dim), self.config.cem_std, device=device)

            tokens_per_frame = c_latent.size(1)

            for _ in range(self.config.cem_num_iters):
                eps = torch.randn(N, H, self.config.action_dim, device=device)
                actions = mu.unsqueeze(0) + std.unsqueeze(0) * eps

                actions[..., -1:] = torch.clamp(actions[..., -1:], 0.0, 1.0)

                z = c_latent.expand(N, -1, -1)

                for h_step in range(H):
                    _a = actions[:, : h_step + 1]
                    _s_seq = actions[:, : h_step + 1]

                    pred = self.predictor(z, _a, _s_seq)
                    if getattr(self.config, "normalize_reps", False):
                        pred = torch.nn.functional.layer_norm(pred, (pred.size(-1),))

                    if h_step < H - 1:
                        z = torch.cat([z, pred[:, -tokens_per_frame:]], dim=1)
                    else:
                        final_latent = pred[:, -tokens_per_frame:]

                g_lat = goal_latent[b : b + 1].expand(N, -1, -1)
                costs = torch.mean(
                    torch.abs(final_latent - g_lat) ** getattr(self.config, "loss_exp", 1.0), dim=(1, 2)
                )

                _, elite_inds = torch.topk(costs, top_k, largest=False)
                elites = actions[elite_inds]

                new_mu = elites.mean(dim=0)
                new_std = elites.std(dim=0) + 1e-5

                mu = new_mu * (1.0 - momentum_mean) + mu * momentum_mean
                std = new_std * (1.0 - momentum_std) + std * momentum_std

            best_trajectories.append(mu)

        torch.cuda.empty_cache()

        return torch.stack(best_trajectories)

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

                with torch.amp.autocast("cuda", enabled=images.is_cuda, dtype=torch.bfloat16):
                    z = self.encoder(images)
                target_latents = z.unsqueeze(1)  # [B, 1, N, D]
            else:
                # Multi-frame case: [B, T, C, H, W] from LeRobot, needs to be [B, C, T, H, W]
                B, T, C, H, W = images.shape
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                images = self._imagenet_normalize(images)

                # Vectorized encoding: Treat B*T as a single batch of 1-frame "videos"
                # Correct reshape: images is [B, C, T, H, W], we want [B*T, C, 1, H, W]
                # We need to swap C and T back to [B, T, C, H, W] BEFORE reshaping to [B*T, C, 1, H, W]
                # Actually, images.permute(0, 2, 1, 3, 4) gives [B, T, C, H, W] which is what we need to flatten
                images_flat = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, 1, H, W)
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
