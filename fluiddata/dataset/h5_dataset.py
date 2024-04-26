from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class H5SequenceDataset(ABC, Dataset[Tensor]):
    def __init__(self, path: Path, sequence_length: int, include_control: bool = False):
        # Parameters
        self.path = path
        self.sequence_length = sequence_length
        self.include_control = include_control
        self.dataset = None

        # Try to read dataset and its parameters
        try:
            with h5py.File(path, "r") as simulation:
                self.parameters = dict(simulation.attrs.items())
        except Exception:
            raise ValueError(f"Error reading dataset: {path}")

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()

    def __getitem__(self, idx: int) -> Tensor:
        # Load singleton
        if self.dataset is None:
            self.dataset = h5py.File(self.path, "r")

        # Get sequence of states
        state_seq = torch.stack(
            [self.get_dataset_state(idx + j) for j in range(0, self.sequence_length)]
        )

        # Get sequence of controls
        if self.include_control:
            control_seq = torch.stack(
                [self.get_dataset_control(idx + j) for j in range(0, self.sequence_length)]
            )
            return state_seq, control_seq

        return state_seq

    @abstractmethod
    def get_dataset_control(self, idx: int) -> Tensor:
        raise NotImplementedError(
            "Subclasses of H5Dataset should implement get_dataset_control()."
        )

    @abstractmethod
    def get_dataset_state(self, idx: int) -> Tensor:
        raise NotImplementedError("Subclasses of H5Dataset should implement get_dataset_state().")
