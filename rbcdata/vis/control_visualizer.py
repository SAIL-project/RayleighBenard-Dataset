from abc import ABC
from typing import Any

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import Event
    from matplotlib.figure import Figure
except ImportError:
    print("Matplotlib not found, visualization is not available")


class ControlVisualizer(ABC):
    def __init__(
        self,
        size: int,
        show: bool = True,
    ) -> None:
        # Matplotlib settings
        self.closed = False
        if show:
            matplotlib.use("QtAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        self.fig, self.ax = plt.subplots()
        (self.hl,) = self.ax.plot(range(size), np.ones(size))
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel("spatial x")
        self.ax.set_ylabel("control")
        self.ax.grid(True)

        # Show
        self.fig.canvas.mpl_connect("close_event", self.close)
        if show:
            plt.show(block=False)

    def draw(self, control) -> Figure:
        self.hl.set_ydata(control)
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
