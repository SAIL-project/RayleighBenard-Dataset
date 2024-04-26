from typing import Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from fluiddata.utils.image_visualizer import ImageVisualizer


class VectorFieldVisualizer(ImageVisualizer):
    def __init__(
        self,
        size: Tuple[int],
        skip: int,
        vrange: Tuple[float, float],
        show: float = True,
        cmap: str = "coolwarm",
        title: str = "",
        ax_args=None,
    ) -> None:
        super().__init__(size, vrange, show, cmap, title, ax_args)

        # Velocity Field
        self.skip = skip
        X, Y = np.meshgrid(np.arange(0, size[1]), np.arange(0, size[0]))
        self.vector_field = self.ax.quiver(
            X[:: self.skip, :: self.skip],
            Y[:: self.skip, :: self.skip],
            np.zeros(tuple(size))[:: self.skip, :: self.skip],
            np.zeros(tuple(size))[:: self.skip, :: self.skip],
            pivot="mid",
            scale=0.01,
            color="white",
            width=0.005,
        )

    def draw(
        self,
        data: npt.NDArray[np.float32],
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        t: float | None = None,
    ) -> Figure:
        """
        Show an image or update the image being shown
        """
        # Update data
        self.image.set_array(data)
        self.vector_field.set_UVC(x[:: self.skip, :: self.skip], y[:: self.skip, :: self.skip])
        self.vector_field.scale = max(0.01, np.linalg.norm(y) / 1.5)

        # Set title
        title = self.title
        if t is not None:
            title += f" t={round(t, 3)}"
        self.ax.set_title(title, loc="left")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self.fig


class RBCFieldVisualizer(VectorFieldVisualizer):
    def __init__(
        self,
        show: float = True,
    ) -> None:
        ax_args = {
            "title": "Temperature and Velocity",
            "ylabel": "spatial y",
            "yticks": [0, 32, 63],
            "yticklabels": [-1, 0, 1],
            "xlabel": "spatial x",
            "xticks": [0, 48, 95],
            "xticklabels": [0, r"$\pi$", r"2$\pi$"],
        }
        super().__init__(
            size=(64, 96), skip=4, vrange=(1, 2), cmap="coolwarm", show=show, ax_args=ax_args
        )
