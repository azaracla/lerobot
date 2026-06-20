"""MPC planning solvers for world model policy.

Ported from https://github.com/galilai-group/stable-worldmodel (MIT-adjacent).
Implements CEM (Cross-Entropy Method) and iCEM (improved CEM) solvers.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
import torch.nn as nn


class CostModel(Protocol):
    """Protocol for models that can compute action costs."""

    def get_cost(
        self,
        info: dict[str, torch.Tensor],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost for each action candidate.

        Args:
            info: dict with context embeddings.
            action_candidates: (B, S, H, A) tensor.

        Returns:
            (B, S) cost per candidate (lower is better).
        """
        ...


class CEMSolver:
    """Cross-Entropy Method solver for MPC.

    Samples action candidates from a Gaussian distribution, evaluates
    them via the world model's get_cost(), selects elite samples,
    and iteratively refines the distribution.

    Parameters
    ----------
    model : CostModel
        World model with get_cost() method.
    num_samples : int
        Number of action candidates per iteration.
    n_steps : int
        Number of CEM iterations.
    topk : int
        Number of elite samples to keep.
    var_scale : float
        Initial exploration variance.
    horizon : int
        Planning horizon (number of action steps).
    action_dim : int
        Dimensionality of the action space.
    action_low : Optional[torch.Tensor]
        Lower bounds for actions (for clamping).
    action_high : Optional[torch.Tensor]
        Upper bounds for actions (for clamping).
    init_mean : Optional[float]
        Initial mean value for the action distribution. If None, defaults to 0.
        Set to mid-point of action space for better exploration (e.g., 256 for PushT).
    device : str
        Device for computation.
    """

    def __init__(
        self,
        model: CostModel,
        num_samples: int = 300,
        n_steps: int = 30,
        topk: int = 30,
        var_scale: float = 1.0,
        horizon: int = 5,
        action_dim: int = 2,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
        init_mean: Optional[float] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.var_scale = var_scale
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device
        self._action_low = action_low
        self._action_high = action_high
        self._init_mean = init_mean

        if action_low is not None:
            self._action_low = action_low.to(device)
        if action_high is not None:
            self._action_high = action_high.to(device)

    @property
    def n_envs(self) -> int:
        return 1  # Single env per solver instance

    def solve(
        self,
        info: dict[str, torch.Tensor],
        init_action: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run CEM to find optimal action sequence.

        Args:
            info: context dict with encoded embeddings.
            init_action: optional warm-start action sequence (B, H, A).

        Returns:
            dict with 'actions' tensor of shape (B, H, A).
        """
        B = info["emb"].shape[0]
        A = self.action_dim
        H = self.horizon

        # Initialize distribution
        if init_action is not None:
            mu = init_action.clone()
        elif self._init_mean is not None:
            mu = torch.full((B, H, A), self._init_mean, device=self.device)
        else:
            mu = torch.zeros(B, H, A, device=self.device)
        var = torch.ones(B, H, A, device=self.device) * self.var_scale

        for _ in range(self.n_steps):
            # Sample candidates: (B, H, A) → (B, S, H, A)
            candidates = mu.unsqueeze(1) + torch.randn(
                B, self.num_samples, H, A, device=self.device
            ) * var.unsqueeze(1).sqrt()

            # Force first candidate to be the current mean
            candidates[:, 0] = mu

            # Clamp to action bounds
            if self._action_low is not None:
                candidates = torch.clamp(candidates, min=self._action_low)
            if self._action_high is not None:
                candidates = torch.clamp(candidates, max=self._action_high)

            # Evaluate costs
            costs = self.model.get_cost(info, candidates)  # (B, S)

            # Select elites
            _, elite_indices = costs.topk(self.topk, dim=1, largest=False)  # (B, topk)
            elites = torch.gather(
                candidates,
                1,
                elite_indices.view(B, self.topk, 1, 1).expand(B, self.topk, H, A),
            )  # (B, topk, H, A)

            # Update distribution
            mu = elites.mean(dim=1)  # (B, H, A)
            var = elites.var(dim=1, unbiased=False) + 1e-8  # (B, H, A)

        return {"actions": mu}


class ICEMSolver(CEMSolver):
    """Improved CEM solver with momentum, colored noise, and elite injection.

    Adds:
    - Momentum EMA on mean and variance
    - Colored noise for temporally correlated exploration
    - Elite injection from previous iteration

    Parameters
    ----------
    alpha : float
        Momentum coefficient for mean/variance updates (0 = full update).
    noise_beta : float
        Colored noise exponent (0 = white noise, higher = more correlation).
    n_elite_keep : int
        Number of elites from previous iteration to inject.
    """

    def __init__(
        self,
        model: CostModel,
        num_samples: int = 300,
        n_steps: int = 30,
        topk: int = 30,
        var_scale: float = 1.0,
        horizon: int = 5,
        action_dim: int = 2,
        alpha: float = 0.1,
        noise_beta: float = 2.0,
        n_elite_keep: int = 5,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        super().__init__(
            model=model,
            num_samples=num_samples,
            n_steps=n_steps,
            topk=topk,
            var_scale=var_scale,
            horizon=horizon,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        self.alpha = alpha
        self.noise_beta = noise_beta
        self.n_elite_keep = n_elite_keep

        # Precompute colored noise frequency scaling
        self._noise_scale = self._compute_colored_noise_scale(self.horizon, noise_beta, device)

        self._prev_elites: Optional[torch.Tensor] = None

    @staticmethod
    def _compute_colored_noise_scale(horizon: int, beta: float, device: str) -> torch.Tensor:
        """Precompute FFT-based scaling for colored noise."""
        freqs = torch.arange(horizon, dtype=torch.float32, device=device)
        scale = freqs ** (-beta / 2)
        scale[0] = scale[1] if len(scale) > 1 else 1.0  # Avoid div by zero
        return scale  # (H,)

    def solve(
        self,
        info: dict[str, torch.Tensor],
        init_action: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run iCEM optimization."""
        B = info["emb"].shape[0]
        A = self.action_dim
        H = self.horizon

        # Initialize distribution
        if init_action is not None:
            mu = init_action.clone()
        else:
            mu = torch.zeros(B, H, A, device=self.device)
        var = torch.ones(B, H, A, device=self.device) * self.var_scale
        prev_elites = None

        for i in range(self.n_steps):
            num_random = self.num_samples
            if prev_elites is not None and i > 0:
                num_random = self.num_samples - prev_elites.shape[1]

            # Generate colored noise
            white_noise = torch.randn(B, num_random, H, A, device=self.device)
            if self.noise_beta > 0 and H > 1:
                # Apply FFT-based coloring along the horizon dimension
                noise_fft = torch.fft.rfft(white_noise, dim=2)
                scale = self._noise_scale[: noise_fft.shape[2]].view(1, 1, -1, 1)
                colored_noise = torch.fft.irfft(noise_fft * scale, n=H, dim=2)
            else:
                colored_noise = white_noise

            candidates = mu.unsqueeze(1) + colored_noise * var.unsqueeze(1).sqrt()

            # Inject previous elites
            if prev_elites is not None:
                candidates = torch.cat([prev_elites, candidates], dim=1)

            # Force first candidate to be current mean
            candidates[:, 0] = mu

            # Clamp
            if self._action_low is not None:
                candidates = torch.clamp(candidates, min=self._action_low)
            if self._action_high is not None:
                candidates = torch.clamp(candidates, max=self._action_high)

            # Evaluate
            costs = self.model.get_cost(info, candidates)

            # Select elites
            _, elite_indices = costs.topk(self.topk, dim=1, largest=False)
            elites = torch.gather(
                candidates,
                1,
                elite_indices.view(B, self.topk, 1, 1).expand(B, self.topk, H, A),
            )

            # Momentum update
            elite_mu = elites.mean(dim=1)
            elite_var = elites.var(dim=1, unbiased=False) + 1e-8
            mu = self.alpha * mu + (1 - self.alpha) * elite_mu
            var = self.alpha * var + (1 - self.alpha) * elite_var

            # Keep top elites for next iteration
            prev_elites = elites[:, : self.n_elite_keep]

        return {"actions": mu}
