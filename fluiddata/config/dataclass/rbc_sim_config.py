from dataclasses import dataclass, field
from typing import List


@dataclass
class RBCSimConfig:
    solver_steps: int = 1
    episode_length: int = 500
    cook_length: int = 100
    N: List[int] = field(default_factory=lambda: [64, 96])
    ra: int = 10000
    pr: float = 0.7
    dt: float = 0.025
    bcT: List[float] = field(default_factory=lambda: [2, 1])
    checkpoint_path: str = "shenfun"
