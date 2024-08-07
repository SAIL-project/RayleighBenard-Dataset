from abc import ABC
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure


class RBCConvectionVisualizer(ABC):
    def __init__(
        self,
        size: List[int] = [64, 96],
        vmin: float = -0.1,
        vmax: float = 0.2,
        show: float = True,
    ) -> None:
        # Matplotlib settings
        self.closed = False
        if show:
            matplotlib.use("QtAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        # Create the figure and axes
        plt.rcParams["font.size"] = 15
        fig, (ax, cbar) = plt.subplots(
            1,
            2,
            gridspec_kw={
                "width_ratios": (0.9, 0.02),
                "wspace": 0.05,
            },
            figsize=(10, 6),
        )
        self.image = ax.imshow(
            np.zeros(size),
            cmap="coolwarm",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        # y axis
        ax.set_ylabel(
            "spatial y",
        )
        ax.set_yticks([0, 32, 63])
        ax.set_yticklabels([-1, 0, 1])
        # x axis
        ax.set_xlabel(
            "spatial x",
        )
        ax.set_xticks([0, 48, 95])
        ax.set_xticklabels([0, r"$\pi$", r"2$\pi$"])
        # Colorbar
        fig.colorbar(
            self.image,
            cax=cbar,
            orientation="vertical",
            ticks=[vmin, vmin / 2, 0, vmax / 2, vmax],
        )
        # cbar.set_yticklabels([-0.1, 0, 0.1, 0.2])
        fig.canvas.mpl_connect("close_event", self.close)
        self.ax = ax
        self.fig = fig

        # Show
        if show:
            plt.show(block=False)

    def draw(self, data: npt.NDArray[np.float32], t: float | None = None) -> Figure:
        """
        Show an image or update the image being shown
        """
        if len(data.shape) == 3:
            data = data.squeeze(0)
        self.image.set_array(data)

        if t is not None:
            self.ax.set_title(f"Local Convective Field at t={round(t, 3)}", loc="left")
        else:
            self.ax.set_title("Local Convective Field", loc="left")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self.fig

    def close(self, event: Event) -> None:
        """
        Close the window
        """
        self.closed = True
        plt.close()
        plt.ioff()
