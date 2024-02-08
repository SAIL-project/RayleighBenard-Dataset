from dataclasses import dataclass

from rbcdata.utils.rbc_type import RBCType


@dataclass
class RBCDatasetConfig:
    context_window_len: int = 1
    type: RBCType = RBCType.NORMAL
    dt: float = 0.25
    start_idx: int = 0
    end_idx: int = 499
