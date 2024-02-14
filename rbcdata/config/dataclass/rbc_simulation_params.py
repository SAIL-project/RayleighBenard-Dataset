from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RBCSimulationParams:
    N: List[int] = field(default_factory=lambda: [64, 96])
    domain: List[List[float]] = field(default_factory=lambda: [[-1, 1], [0, 2 * np.pi]])
    Ra: float = 1000000
    Pr: float = 0.7
    dt: float = 0.1
    bcT: List[float] = field(default_factory=lambda: [2, 1])
