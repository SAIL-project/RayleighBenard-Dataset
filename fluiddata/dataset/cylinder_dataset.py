from enum import Enum, IntEnum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fluiddata.dataset.h5_dataset import H5SequenceDataset


class CylinderField(IntEnum):
    UX = 0
    UY = 1
    P = 2
    VORT = 3
    MAGN = 4


class CylinderType(Enum):
    VORTICITY = "vorticity"
    SIM = "sim"
    FULL = "full"


class CylinderDataset(H5SequenceDataset):
    def __init__(
        self,
        path: Path,
        sequence_length: int,
        include_control: bool = False,
        type: CylinderType = CylinderType.FULL,
        transform: torch.nn.Module | None = None,
    ):
        super().__init__(path, sequence_length, include_control)
        self.type = type
        self.transform = transform

    def __len__(self) -> int:
        return int(self.parameters["steps"])

    def get_dataset_control(self, idx: int) -> Tensor:
        return torch.tensor(np.array(self.dataset["control"][idx]), dtype=torch.float32)

    def get_dataset_state(self, idx: int) -> Tensor:
        state = torch.tensor(np.array(self.dataset["state"][idx]), dtype=torch.float32)

        if self.type == CylinderType.VORTICITY:
            state = state[CylinderField.VORT]
        elif self.type == CylinderType.SIM:
            state = state[:3]

        # Apply transform
        if self.transform:
            state = self.transform(state)

        return state
