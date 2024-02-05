import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig


@dataclass
class RBCEnvConfig:
    ckpt_path: pathlib.Path = pathlib.Path("/tmp/shenfun")
    solver_steps: int = 1
    episode_length: int = 500
    cook_time: int = 100
    N: Tuple[int, int] = (64, 96)
    Ra: int = 10000
    Pr: float = 0.7
    dt: float = 0.025
    bcT: Tuple[int, int] = (2, 1)
    domain: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1, 1), (0, 2 * np.pi))
