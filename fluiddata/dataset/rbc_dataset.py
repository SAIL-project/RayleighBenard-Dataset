import math
from enum import Enum, IntEnum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fluiddata.dataset.h5_dataset import H5SequenceDataset


class RBCField(IntEnum):
    UY = 0
    UX = 1
    T = 2
    JCONV = 3


class RBCType(Enum):
    NORMAL = "normal"
    CONVECTION = "convection"
    FULL = "full"


class RBCDataset(H5SequenceDataset):
    def __init__(
        self,
        path: Path,
        sequence_length: int = 1,
        dt: float = 0.25,
        start_idx: int = 0,
        end_idx: int = 1999,
        include_control: bool = False,
        type: RBCType = RBCType.NORMAL,
        transform: torch.nn.Module | None = None,
    ):
        super().__init__(path, sequence_length, include_control)
        # dataset parameters and transform
        self.dt = dt
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.type = type
        self.transform = transform
        # Legacy
        if "dt" in self.parameters.keys():
            self.step_factor = math.floor(dt / self.parameters["dt"])
        else:
            self.step_factor = math.floor(dt / self.parameters["action_duration"])

    def __len__(self) -> int:
        return int(self.end_idx - self.start_idx - self.sequence_length + 2)

    def get_dataset_control(self, idx: int) -> Tensor:
        return torch.tensor(
            np.array(self.dataset["action"][idx * self.step_factor]), dtype=torch.float32
        )

    def get_dataset_state(self, idx: int) -> Tensor:
        # Load state from dataset; multiply by step factor for correct dt
        state = torch.tensor(
            np.array(self.dataset["data"][idx * self.step_factor]), dtype=torch.float32
        )

        # only convection field
        if self.type == RBCType.CONVECTION:
            state = (
                state[RBCField.UY] * (state[RBCField.T] - torch.mean(state[RBCField.T]))
            ).unsqueeze(0)
        # return normal + convection fields
        elif self.type == RBCType.FULL:
            jconv = (
                state[RBCField.UY] * (state[RBCField.T] - torch.mean(state[RBCField.T]))
            ).unsqueeze(0)
            state = torch.cat((state, jconv), dim=0)

        # Apply transform
        if self.transform:
            state = self.transform(state)

        return state
