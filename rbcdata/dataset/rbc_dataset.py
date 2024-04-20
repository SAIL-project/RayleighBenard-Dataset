import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rbcdata.config.dataclass.rbc_dataset_config import RBCDatasetConfig
from rbcdata.utils.enums import RBCField, RBCType


class RBCDataset(Dataset[Tensor]):
    def __init__(
        self,
        path: Path,
        cfg: RBCDatasetConfig,
        transform: torch.nn.Module | None = None,
    ):
        self.dataset = None
        self.path = path
        # dataset parameters and transform
        self.cfg = cfg
        self.transform = transform
        # Read dataset parameters
        try:
            with h5py.File(path, "r") as simulation:
                self.parameters = dict(simulation.attrs.items())

                # Compute step factor
                if "dt" in simulation.attrs:
                    self.step_factor = math.floor(cfg.dt / simulation.attrs["dt"])
                elif "action_duration" in simulation.attrs:
                    self.step_factor = math.floor(cfg.dt / simulation.attrs["action_duration"])
                else:
                    raise ValueError("No dt or action_duration in dataset")

                # assertions
                assert (
                    self.step_factor > 0
                ), "dataset dt must be a multiple of sim dt/agent_duration"

        except Exception:
            raise ValueError(f"Error reading dataset: {path}")

    def __len__(self) -> int:
        return int(self.cfg.end_idx - self.cfg.start_idx - self.cfg.sequence_length + 2)

    def __getitem__(self, idx: int) -> Tensor:
        state_seq = torch.stack(
            [self.get_dataset_state(idx + j) for j in range(0, self.cfg.sequence_length)]
        )

        if self.cfg.include_control:
            control_seq = torch.stack(
                [self.get_dataset_control(idx + j) for j in range(0, self.cfg.sequence_length)]
            )
            return state_seq, control_seq

        return state_seq

    def get_dataset_control(self, idx: int) -> Tensor:
        return torch.tensor(
            np.array(self.dataset["action"][idx * self.step_factor]), dtype=torch.float32
        )

    def get_dataset_state(self, idx: int) -> Tensor:
        # Load singleton
        if self.dataset is None:
            self.dataset = h5py.File(self.path, "r")

        # Load state from dataset; multiply by step factor for correct dt
        state = torch.tensor(
            np.array(self.dataset["data"][idx * self.step_factor]), dtype=torch.float32
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
