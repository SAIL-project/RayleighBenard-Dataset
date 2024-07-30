from abc import ABC
from typing import Any, List

import numpy as np
import numpy.typing as npt
from sympy import Piecewise
import sympy
from spb import plot_piecewise

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
        x_domain = (0, 2*np.pi),
        n_segments_plot = 100
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

        # Create the figure and axes
        plt.rcParams["font.size"] = 15

        self.fig, self.action_ax = plt.subplots(
            figsize=(10, 6),
        )
        self.x_domain = x_domain
        # y axis
        self.action_ax.set_ylabel("Applied temperature")
        # self.ax.set_yticks([0, 32, 63])
        # self.ax.set_yticklabels([-1, 0, 1])
        # X axis
        self.action_ax.set_xlabel("spatial x")
        # self.ax.set_xticks([0, 48, 95])
        # self.ax.set_xticklabels([0, r"$\pi$", r"2$\pi$"])

        self.fig.canvas.mpl_connect("close_event", self.close)
        # Velocity Field
        
        # Show
        if show:
            plt.show(block=False)

    def draw(self, action_effective: Piecewise, y, sim_t) -> Figure:
        """
        Show an action curve or update the action curve.
        """
        # Update the action curve
        self.action_ax.clear()
        # plot here in the axes
        plot_piecewise(
            action_effective,
            (y, float(self.x_domain[0]), float(self.x_domain[1])),
            ax=self.action_ax,
        )
        self.action_ax.set_title(f'Action taken at t={sim_t:.3f}')
        self.action_ax.set_ylabel("Applied temperature")
        self.action_ax.set_xlabel("Spatial x coordinate")
        
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

        return self.fig

    def close(self, event: Event | None = None) -> Any:
        """
        Close the window
        """
        self.closed = True
        plt.close()
        plt.ioff()
