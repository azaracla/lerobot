"""LeWM JEPA World Model policy plugin for LeRobot.

A lightweight JEPA (Joint Embedding Predictive Architecture) that trains
end-to-end from pixels with only 2 loss terms. Uses CEM for MPC inference.

Reference: https://github.com/lucas-maes/le-wm (MIT License)
"""

from .configuration_lewm import LeWMConfig
from .modeling_lewm import LeWMPolicy
from .processor_lewm import make_lewm_pre_post_processors
from .jepa import JEPA
from .modules import (
    ARPredictor,
    Embedder,
    MLP,
    SIGReg,
    Transformer,
    ViTEncoder,
)
from .solver import CEMSolver, ICEMSolver

__all__ = [
    "LeWMConfig",
    "LeWMPolicy",
    "make_lewm_pre_post_processors",
    "JEPA",
    "ViTEncoder",
    "ARPredictor",
    "Embedder",
    "MLP",
    "SIGReg",
    "Transformer",
    "CEMSolver",
    "ICEMSolver",
]
