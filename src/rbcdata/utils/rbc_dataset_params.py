from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

from rbcdata.utils.rbc_type import RBCType


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
    domain: List[List[float]]
    N: List[int]
    seed: int
