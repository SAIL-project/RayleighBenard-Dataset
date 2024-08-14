from dataclasses import dataclass, field
from typing import List


@dataclass
class RBCSimConfig:
    solver_steps: int = 1
    episode_length: int = 500
    cook_length: int = 100  # initial time to evolve the system without taking snapshots
    N: List[int] = field(
        default_factory=lambda: [64, 96]
    )  # discretization of the domain, vertical horizontal.
    N_obs: List[int] = field(default_factory=lambda: [8, 48])
    ra: int = 10000
    pr: float = 0.7
    dt: float = 0.025  # time-step of simulation
    bcT: List[float] = field(
        default_factory=lambda: [2, 1]
    )  # unforced boundary condition, bottom and top.
    checkpoint_path: str = "shenfun"
