import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rbcdata.config.dataclass.rbc_dataset_config import RBCDatasetConfig
from rbcdata.utils.rbc_field import RBCField
from rbcdata.utils.rbc_type import RBCType


class RBCDataset(Dataset[Tensor]):
    def __init__(
        self,
        path: Path,
        cfg: RBCDatasetConfig,
        transform: torch.nn.Module | None = None,
    ):
        # dataset parameters and transform
        self.cfg = cfg
        self.transform = transform
        # Read dataset parameters
        try:
            with h5py.File(path, "r") as simulation:
                self.data = np.array(simulation["data"])
                self.parameters = dict(simulation.attrs.items())
                self.step_factor = math.floor(cfg.dt / simulation.attrs["dt"])
                # assertions
                assert (
                    cfg.dt >= simulation.attrs["dt"]
                ), "dt must be greater equal than the simulation dt"

        except Exception:
            raise ValueError(f"Error reading dataset: {path}")

    def __len__(self) -> int:
        return int(self.cfg.end_idx - self.cfg.start_idx - self.cfg.sequence_length + 2)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.stack(
            [self.get_dataset_state(idx + j) for j in range(0, self.cfg.sequence_length)]
        )

    def get_dataset_state(self, idx: int) -> Tensor:
        # Load state from dataset; multiply by step factor for correct dt
        state = torch.tensor(self.data[idx * self.step_factor], dtype=torch.float32)

        # only convection field
        if self.cfg.type == RBCType.CONVECTION:
            state = (
                state[RBCField.UY] * (state[RBCField.T] - torch.mean(state[RBCField.T]))
            ).unsqueeze(0)
        # return normal + convection fields
        elif self.cfg.type == RBCType.FULL:
            jconv = (
                state[RBCField.UY] * (state[RBCField.T] - torch.mean(state[RBCField.T]))
            ).unsqueeze(0)
            state = torch.cat((state, jconv), dim=0)

        # Apply transform
        if self.transform:
            state = self.transform(state)

        return state
