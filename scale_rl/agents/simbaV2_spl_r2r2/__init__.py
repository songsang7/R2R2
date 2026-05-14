"""
SimbaV2-SPL Agent Module
"""

from scale_rl.agents.simbaV2_spl_r2r2.agent import SimbaV2SplR2R2Agent
from scale_rl.agents.simbaV2_spl.simbaV2_spl_network import (
    HypersphereEncoder,
    HypersphereLatentDynamicsModel,
    ObsProjector,
    SimbaV2SplActor,
    SimbaV2SplCritic,
    SimbaV2SplDoubleCritic,
    SimbaV2SplTemperature,
)

__all__ = [
    "SimbaV2SplR2R2Agent",
    "HypersphereEncoder",
    "HypersphereLatentDynamicsModel",
    "ObsProjector",
    "SimbaV2SplActor",
    "SimbaV2SplCritic",
    "SimbaV2SplDoubleCritic",
    "SimbaV2SplTemperature",
]
