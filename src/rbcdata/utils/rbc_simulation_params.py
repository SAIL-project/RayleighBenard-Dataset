from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from rbcdata.utils import RBCType


@dataclass
class RBCSimulationParams:
    N: Tuple[int, int] = (64, 96)
    Nobs: Tuple[int, int] = (8, 32)
    domain: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1, 1), (0, 2 * np.pi))
    Ra: float = 10000
    Pr: float = 0.7
    nu: float = 0.01
    dt: float = 0.1
    bcT: Tuple[float, float] = (2, 1)
    conv: float = 0
    dpdy: float = 1
    filename: str = "data/shenfun/RB_2D"
    family: str = "C"
    padding: Tuple[float, float] = (1, 1.5)
    modsave: float = 1e8
    checkpoint: int = 1000
