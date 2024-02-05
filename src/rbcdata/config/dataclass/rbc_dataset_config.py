import pathlib
from dataclasses import dataclass

from rbcdata.utils import RBCType


@dataclass
class RBCDatasetConfig:
    path: pathlib.Path
    sequence_length: int = 1
    type: RBCType = RBCType.NORMAL
    dt: float = 0.25
    start_idx: int = 0
    end_idx: int = 499
