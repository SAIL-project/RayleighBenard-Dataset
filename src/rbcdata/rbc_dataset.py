import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rbcdata.config.dataclass.rbc_dataset_config import RBCDatasetConfig
from rbcdata.utils.rbc_dataset_params import RBCDatasetParams
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
                self.dataset = np.array(simulation["data"])

                # misc parameters
                step_factor = math.floor(cfg.dt / simulation.attrs["dt"])
                sim_length = math.floor(simulation.attrs["samples"] / step_factor)

                # dataset parameters
                self.parameters = RBCDatasetParams(
                    sequence_length=cfg.sequence_length,
                    type=cfg.type,
                    dt=cfg.dt,
                    start_idx=cfg.start_idx,
                    end_idx=cfg.end_idx,
                    step_factor=step_factor,
                    sim_length=sim_length,
                    spatial_mesh=np.array(simulation["spatial_mesh"]),
                    domain=simulation.attrs["domain"],
                    N=simulation.attrs["N"],
                    seed=simulation.attrs["seed"],
                )

                # assertions
                assert (
                    cfg.dt >= simulation.attrs["dt"]
                ), "dt must be greater equal than the simulation dt"
                assert (
                    cfg.end_idx is None or cfg.end_idx < sim_length
                ), "end_idx must be smaller than the simulation length"
                assert (
                    cfg.sequence_length <= cfg.end_idx - cfg.start_idx
                ), "sequence length too long"

        except Exception:
            raise ValueError(f"Error reading dataset: {path}")

    def __len__(self) -> int:
        return self.cfg.end_idx - self.cfg.start_idx - self.cfg.sequence_length + 1

    def __getitem__(self, idx: int) -> Tensor:
        # check if in range
        assert self.cfg.start_idx + idx <= self.cfg.end_idx, "index out of range"
        # Single data frame
        if self.cfg.sequence_length == 0:
            return self.get_dataset_state(idx)

        # Data Sequence
        else:
            return torch.stack(
                [
                    self.get_dataset_state(idx + j)
                    for j in range(0, self.cfg.sequence_length)
                ]
            )

    def get_dataset_state(self, idx: int) -> Tensor:
        # Load state from dataset; multiply by step factor for correct dt
        state = torch.tensor(
            self.dataset[idx * self.parameters.step_factor], dtype=torch.float32
        )

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
