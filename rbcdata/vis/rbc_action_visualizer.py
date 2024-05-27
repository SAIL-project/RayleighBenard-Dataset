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


class RBCActionVisualizer(ABC):
    def __init__(
        self,
        show: bool = True,
    ) -> None:
        # Matplotlib settings
        self.closed = False
        if show:
            matplotlib.use("TkAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        # Rendering
        self.last_image_shown = None

        # Create the figure and axes
        plt.rcParams["font.size"] = 15

        self.fig, (self.ax, self.cbar) = plt.subplots(
            figsize=(10, 6),
        )
        # extent = [0, 2 * math.pi, -1, 1]
        self.image_T = self.ax.plot(np.linspace(0, 2*np.pi, 10), np.zeros(10))
        # y axis
        self.ax.set_ylabel("Applied temperature")
        # self.ax.set_yticks([0, 32, 63])
        # self.ax.set_yticklabels([-1, 0, 1])
        # X axis
        self.ax.set_xlabel("spatial x")
        # self.ax.set_xticks([0, 48, 95])
        # self.ax.set_xticklabels([0, r"$\pi$", r"2$\pi$"])

        self.fig.canvas.mpl_connect("close_event", self.close)
        # Velocity Field
        
        # Show
        if show:
            plt.show(block=False)

    def draw(
        self,
        action: npt.NDArray[np.float32],
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
