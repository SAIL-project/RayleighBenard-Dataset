from abc import ABC
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from spb import plot_piecewise
from sympy import Piecewise

x, y, tt = sympy.symbols("x,y,t", real=True)


class RBCActionVisualizer(ABC):
    def __init__(
        self,
        show: bool = True,
        x_domain=(0, 2 * np.pi),
        bcT=(2, 1),
        action_limit=0.75,
        n_segments_plot=100,
    ) -> None:
        self.x_domain = x_domain
        self.action_limit = action_limit
        self.bcT = bcT
        # Matplotlib settings
        self.closed = False
        if show:
            matplotlib.use("QtAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        # Create the figure and axes
        plt.rcParams["font.size"] = 10
        self.fig, self.action_ax = plt.subplots(
            figsize=(5, 4),
        )

        # Show
        if show:
            self.fig.canvas.mpl_connect("close_event", self.close)
            plt.show(block=False)

    def draw(self, action_effective: Piecewise, t: float) -> Figure:
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
        self.set_axis(t)
        self.fig.canvas.draw()

        return self.fig

    def set_axis(self, t: float):
        self.action_ax.set_title(f"Action taken at t={t:.3f}")
        self.action_ax.set_ylabel("Applied temperature")
        self.action_ax.set_xlabel("Spatial x coordinate")
        self.action_ax.set_ylim(self.bcT[1], self.bcT[0] + self.action_limit)
        self.fig.tight_layout()

    def close(self, event: Event | None = None) -> Any:
        """
        Close the window
        """
        self.closed = True
        plt.close()
        plt.ioff()
