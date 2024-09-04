from abc import ABC
from typing import Any, List

import numpy as np
import numpy.typing as npt

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import Event
    from matplotlib.figure import Figure
except ImportError:
    print("Matplotlib not found, visualization is not available")


class RBCFieldVisualizer(ABC):
    def __init__(
        self,
        size: List[int] = [64, 96],
        vmin: float = 0,
        vmax: float = 1,
        show_u: bool = True,
        show: bool = True,
        skip: int = 4,
    ) -> None:
        # Matplotlib settings
        self.closed = False
        if show:
            matplotlib.use("QtAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        # Rendering
        self.last_image_shown = None
        self.show_u = show_u

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
        # extent = [0, 2 * math.pi, -1, 1]
        self.image_T = self.ax.imshow(
            np.zeros(size),
            cmap="coolwarm",  # "turbo",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        # y axis
        self.ax.set_ylabel(
            "spatial y",
        )
        self.ax.set_yticks([0, 32, 63])
        self.ax.set_yticklabels([-1, 0, 1])
        # X axis
        self.ax.set_xlabel(
            "spatial x",
        )
        self.ax.set_xticks([0, 48, 95])
        self.ax.set_xticklabels([0, r"$\pi$", r"2$\pi$"])

        self.fig.colorbar(
            self.image_T,
            cax=self.cbar,
            orientation="vertical",
            ticks=[vmin, (vmin + vmax) / 2, vmax],
        )
        self.cbar.set_yticklabels([vmin, (vmin + vmax) / 2, vmax])
        self.fig.canvas.mpl_connect("close_event", self.close)
        # Velocity Field
        self.skip = skip
        if show_u:
            X, Y = np.meshgrid(np.arange(0, size[1]), np.arange(0, size[0]))
            self.image_u = self.ax.quiver(
                X[:: self.skip, :: self.skip],
                Y[:: self.skip, :: self.skip],
                np.zeros(tuple(size))[:: self.skip, :: self.skip],
                np.zeros(tuple(size))[:: self.skip, :: self.skip],
                pivot="mid",
                scale=0.01,
                color="white",
                width=0.005,
            )

        # Show
        if show:
            plt.show(block=False)

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

        # TODO maybe a better, way, but show max velocity in title for now
        max_velo = np.max(np.sqrt(ux**2 + uy**2))

        # Update T image
        self.image_T.set_array(T)
        if cooking:
            self.ax.set_title(
                f"Cooking Time active at t={round(t, 3)}. Max velocity: {max_velo:.3f}",
                loc="left",
            )
        else:
            self.ax.set_title(
                f"Temperature Field at t={round(t, 3)}. Max velocity: {max_velo:.3f}",
                loc="left",
            )

        # Update u image
        if self.show_u:
            self.image_u.set_UVC(ux[:: self.skip, :: self.skip], uy[:: self.skip, :: self.skip])
            scale = np.linalg.norm(uy) / 1.5
            self.image_u.scale = 0.01 if scale == 0 else scale

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self.fig

    def close(self, event: Event | None = None) -> Any:
        """
        Close the window
        """
        self.closed = True
        plt.close()
        plt.ioff()
