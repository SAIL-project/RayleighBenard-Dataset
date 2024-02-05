from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from rbcdata.utils import RBCType


@dataclass
class RBCDatasetParams:
    sequence_length: int
    type: RBCType
    dt: float
    start_idx: int
    end_idx: int
    step_factor: int
    sim_length: int
    spatial_mesh: npt.NDArray[np.float32]
    domain: Tuple[Tuple[float, float], Tuple[float, float]]
    N: Tuple[int, int]
    seed: int
