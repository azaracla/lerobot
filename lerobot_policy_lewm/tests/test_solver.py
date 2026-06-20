"""Unit tests for CEM and iCEM solvers."""

import pytest
import torch

from lerobot_policy_lewm.solver import CEMSolver, ICEMSolver


class DummyCostModel:
    """Toy model with known optimal action at [1.0, 0.5]."""

    def __init__(self, optimal_action=None):
        if optimal_action is None:
            optimal_action = torch.tensor([1.0, 0.5])
        self.optimal = optimal_action
        self.training = False

    def get_cost(self, info, candidates):
        """Cost is squared distance to optimal action (averaged over horizon)."""
        # candidates: (B, S, H, A), optimal: (A,)
        target = self.optimal.to(candidates.device).view(1, 1, 1, -1)
        return ((candidates - target) ** 2).mean(dim=(-2, -1))  # (B, S)


class TestCEMSolver:
    def test_converges_to_optimum(self):
        model = DummyCostModel()
        solver = CEMSolver(
            model=model,
            num_samples=100,
            n_steps=20,
            topk=20,
            var_scale=1.0,
            horizon=3,
            action_dim=2,
            device="cpu",
        )

        info = {"emb": torch.randn(1, 3, 64)}  # dummy context
        result = solver.solve(info)
        actions = result["actions"]  # (1, H, A)

        # Should be close to optimal
        expected = model.optimal.unsqueeze(0).unsqueeze(0).expand(1, 3, 2)
        error = (actions - expected).abs().mean()
        assert error < 0.3, f"Expected error < 0.3, got {error}"

    def test_returns_correct_shape(self):
        model = DummyCostModel()
        solver = CEMSolver(
            model=model,
            num_samples=50,
            n_steps=10,
            topk=10,
            var_scale=1.0,
            horizon=5,
            action_dim=2,
            device="cpu",
        )

        info = {"emb": torch.randn(1, 3, 64)}
        result = solver.solve(info)
        assert result["actions"].shape == (1, 5, 2)

    def test_clamps_to_bounds(self):
        model = DummyCostModel(optimal_action=torch.tensor([100.0, 100.0]))  # outside bounds
        solver = CEMSolver(
            model=model,
            num_samples=100,
            n_steps=20,
            topk=20,
            var_scale=1.0,
            horizon=3,
            action_dim=2,
            action_low=torch.tensor([-1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0]),
            device="cpu",
        )

        info = {"emb": torch.randn(1, 3, 64)}
        result = solver.solve(info)
        actions = result["actions"]

        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()

    def test_warm_start(self):
        model = DummyCostModel()
        solver = CEMSolver(
            model=model,
            num_samples=100,
            n_steps=20,
            topk=20,
            var_scale=0.1,
            horizon=3,
            action_dim=2,
            device="cpu",
        )

        # Warm start with near-optimal action
        init = model.optimal.unsqueeze(0).unsqueeze(0).expand(1, 3, 2) + 0.05

        info = {"emb": torch.randn(1, 3, 64)}
        result = solver.solve(info, init_action=init)
        actions = result["actions"]

        expected = model.optimal.unsqueeze(0).unsqueeze(0).expand(1, 3, 2)
        error = (actions - expected).abs().mean()
        assert error < 0.1  # should converge very close with warm start


class TestICEMSolver:
    def test_converges_to_optimum(self):
        model = DummyCostModel()
        solver = ICEMSolver(
            model=model,
            num_samples=100,
            n_steps=30,
            topk=20,
            var_scale=1.0,
            horizon=3,
            action_dim=2,
            alpha=0.1,
            noise_beta=2.0,
            n_elite_keep=5,
            device="cpu",
        )

        info = {"emb": torch.randn(1, 3, 64)}
        result = solver.solve(info)
        actions = result["actions"]

        expected = model.optimal.unsqueeze(0).unsqueeze(0).expand(1, 3, 2)
        error = (actions - expected).abs().mean()
        assert error < 0.3, f"Expected error < 0.3, got {error}"

    def test_returns_correct_shape(self):
        model = DummyCostModel()
        solver = ICEMSolver(
            model=model,
            num_samples=50,
            n_steps=10,
            topk=10,
            var_scale=1.0,
            horizon=5,
            action_dim=2,
            device="cpu",
        )

        info = {"emb": torch.randn(1, 3, 64)}
        result = solver.solve(info)
        assert result["actions"].shape == (1, 5, 2)
