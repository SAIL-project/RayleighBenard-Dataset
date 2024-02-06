import pathlib
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RBCEnvConfig:
    ckpt_path: pathlib.Path = pathlib.Path("/tmp/shenfun")
    solver_steps: int = 1
    episode_length: int = 500
    cook_time: int = 100
    N: List[int] = field(default_factory=lambda: [64, 96])
    Ra: int = 10000
    Pr: float = 0.7
    dt: float = 0.025
    bcT: List[float] = field(default_factory=lambda: [2, 1])
    domain: List[List[float]] = field(default_factory=lambda: [[-1, 1], [0, 2 * np.pi]])
