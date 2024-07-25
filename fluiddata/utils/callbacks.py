import os
from typing import Callable, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from firedrake import FunctionSpace, VertexOnlyMesh, assemble
    from firedrake.__future__ import interpolate
    from hydrogym.core import CallbackBase, PDEBase
except ImportError:
    print("Hydrogym/Firedrake not found, Cylinder simulation is not available")


class CylinderVisCallback(CallbackBase):

    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)

        # matplotlib stuff
        matplotlib.use("QtAgg")
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 3))
        plt.show(block=False)

    def __call__(self, iter: int, t: float, flow: PDEBase):
        if super().__call__(iter, t, flow):
            self.render(flow)

    def render(self, flow: PDEBase):
        flow.render(axes=self.ax, cmap=sns.color_palette("RdBu", as_cmap=True))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class H5DatasetCallback(CallbackBase):
    CHANNELS = 1

    def __init__(
        self,
        filename: str,
        t_start: float,
        flow: PDEBase,
        fields: Callable,
        steps: int,
        grid_N: Tuple[int, int],
        grid_domain: Tuple[Tuple[float, float], Tuple[float, float]],
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)

        # Get number of fields
        self.get_fields = fields
        self.channels = len(fields(flow))

        # Get points to evaluate at
        y = np.linspace(grid_domain[0][0], grid_domain[0][1], num=grid_N[0])
        x = np.linspace(grid_domain[1][0], grid_domain[1][1], num=grid_N[1])
        xv, yv = np.meshgrid(x, y, indexing="ij")
        self.points = np.array([xv.ravel(), yv.ravel()]).T

        # Create vertex only mesh
        self.grid_mesh = VertexOnlyMesh(flow.mesh, self.points, missing_points_behaviour="warn")
        self.grid = FunctionSpace(self.grid_mesh, "DG", 0)

        # Create file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = h5py.File(filename, "w")

        # Create datasets for state and control
        self.state_idx = 0
        self.dataset_state = self.file.create_dataset(
            "state",
            (steps, self.channels, grid_N[0], grid_N[1]),
            chunks=(10, self.channels, grid_N[0], grid_N[1]),
            compression="gzip",
            dtype=np.float32,
        )
        # TODO control dataset

        # Save simulation parameters
        self.t_start = t_start
        self.N = grid_N
        self.domain = grid_domain

        self.file.attrs["steps"] = steps
        self.file.attrs["N"] = grid_N
        self.file.attrs["domain"] = grid_domain

    def __call__(self, iter: int, t: float, flow: PDEBase):
        if super().__call__(iter, t, flow):
            # Save after start time
            if t < self.t_start:
                return
            # Get functions and coordinates
            functions = []
            coordinates = self.grid_mesh.coordinates.dat.data
            f = self.get_fields(flow)

            # Evaluate fields at coordinates and save to dataset
            for i, field in enumerate(f):
                # Interpolate field to grid
                functions.append(assemble(interpolate(field, self.grid)).dat.data)

            # Build state
            state = np.zeros((self.channels, self.N[0], self.N[1]))
            for idx, (x, y) in enumerate(coordinates):
                # domain to index
                i = self.domain2index(y, self.domain[0], self.N[0])
                j = self.domain2index(x, self.domain[1], self.N[1])
                for c, f in enumerate(functions):
                    state[c, i, j] = f[idx]
            # save to datset
            self.dataset_state[self.state_idx] = state
            self.state_idx += 1

    def domain2index(self, value: float, domain: Tuple[float, float], N) -> int:
        return round((value - domain[0]) * (N - 1) / (domain[1] - domain[0]))

    def __del__(self):
        self.file.close()
