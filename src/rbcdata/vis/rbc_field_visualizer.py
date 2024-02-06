import math
from abc import ABC
from typing import Any, List

import numpy as np
import numpy.typing as npt
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not found, visualization is not available")


class RBCFieldVisualizer(ABC):
    def __init__(
        self,
        N: List[int],
        spatial_mesh: npt.NDArray[np.float32],
        vmin: float = 0,
        vmax: float = 1,
        block: bool = False,
        show_u: bool = True,
        show: bool = True,
    ) -> None:
        # Rendering
        self.last_image_shown = None
        self.show_u = show_u
        self.closed = False

        # Data
        X = spatial_mesh

        # Create the figure and axes
        plt.rcParams["font.size"] = 15

        self.fig, (self.ax, self.cbar) = plt.subplots(
            1,
            2,
            gridspec_kw={
                "width_ratios": (0.9, 0.02),
                "wspace": 0.05,
            },
            figsize=(10, 6),
        )
        extent = [0, 2 * math.pi, -1, 1]
        self.image_T = self.ax.imshow(
            np.zeros(N),
            vmax=vmax,
            vmin=vmin,
            cmap="turbo",
            extent=extent,
            aspect="auto",
        )
        # Y axis
        self.ax.set_ylabel(
            "spatial y",
        )
        self.ax.set_yticks([-1, 0, 1])
        self.ax.set_yticklabels([-1, 0, 1])
        # X axis
        self.ax.set_xlabel(
            "spatial x",
        )
        self.ax.set_xticks([0, math.pi, 2 * math.pi])
        self.ax.set_xticklabels([0, r"$\pi$", r"2$\pi$"])

        self.fig.colorbar(
            self.image_T,
            cax=self.cbar,
            orientation="vertical",
            ticks=[1, 1.5, 2],
        )
        self.cbar.set_yticklabels([1, 1.5, 2])
        self.fig.canvas.mpl_connect("close_event", self.close)
        # Velocity Field
        if show_u:
            self.image_u = self.ax.quiver(
                X[1][::5, ::5],
                X[0][::5, ::5],
                np.zeros(N)[::5, ::5],
                np.zeros(N)[::5, ::5],
                pivot="mid",
                scale=0.01,
                color="white",
                width=0.005,
            )

        # Show
        self.show = show
        if show:
            if not block:
                plt.ion()
            else:
                plt.ioff()
            plt.show(block=block)
        else:
            # Change backend if plots are not displayed
            matplotlib.pyplot.switch_backend("Agg")

    def draw(
        self,
        T: npt.NDArray[np.float32],
        ux: npt.NDArray[np.float32],
        uy: npt.NDArray[np.float32],
        t: float,
        cooking: bool = False,
    ) -> Figure:
        """
        Show an image or update the image being shown
        """
        # Update T image
        self.image_T.set_array(T)
        if cooking:
            self.ax.set_title(
                f"Cooking Time active at t={round(t, 3)}",
                loc="left",
            )
        else:
            self.ax.set_title(
                f"Temperature Field at t={round(t, 3)}",
                loc="left",
            )

        if self.show:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # Update u image
        if self.show_u:
            self.image_u.set_UVC(ux[::5, ::5], uy[::5, ::5])
            scale = np.linalg.norm(uy) / 1.5
            self.image_u.scale = 0.01 if scale == 0 else scale

        return self.fig

    def close(self, event: Event | None = None) -> Any:
        """
        Close the window
        """
        self.closed = True
        plt.close()
        plt.ioff()
