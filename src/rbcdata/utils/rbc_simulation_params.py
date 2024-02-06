from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RBCSimulationParams:
    N: List[int] = field(default_factory=lambda: [64, 96])
    Nobs: List[int] = field(default_factory=lambda: [8, 32])
    domain: List[List[float]] = field(default_factory=lambda: [[-1, 1], [0, 2 * np.pi]])
    Ra: float = 10000
    Pr: float = 0.7
    nu: float = 0.01
    dt: float = 0.1
    bcT: List[float] = field(default_factory=lambda: [2, 1])
    conv: float = 0
    dpdy: float = 1
    filename: str = "data/shenfun/RB_2D"
    family: str = "C"
    padding: List[float] = field(default_factory=lambda: [1, 1.5])
    modsave: float = 1e8
    checkpoint: int = 1000
