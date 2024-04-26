from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure


class ImageVisualizer:
    def __init__(
        self,
        size: Tuple[int],
        vrange: Tuple[float, float],
        show: float = True,
        cmap: str = "coolwarm",
        title: str = "",
        ax_args=None,
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
        self.fig, (self.ax, self.cbar) = plt.subplots(
            1,
            2,
            gridspec_kw={
                "width_ratios": (0.9, 0.02),
                "wspace": 0.05,
            },
            figsize=(10, 6),
        )

        # Show empty image
        self.image = self.ax.imshow(
            np.zeros(size),
            cmap=cmap,
            aspect="equal",
            vmin=vrange[0],
            vmax=vrange[1],
        )
        self.title = title

        # Set axis labels
        if ax_args is not None:
            self.ax.set(**ax_args)

        # Set color bar
        self.fig.colorbar(
            self.image,
            cax=self.cbar,
            orientation="vertical",
            ticks=[vrange[0], 0, vrange[1]],
        )
        # self.cbar.set_yticklabels([-0.1, 0, 0.1, 0.2])

        # Show
        self.fig.canvas.mpl_connect("close_event", self.close)
        if show:
            plt.show(block=False)

    def draw(self, data: npt.NDArray[np.float32], t: float | None = None) -> Figure:
        """
        Show an image or update the image being shown
        """
        self.image.set_array(data)

        # Set title
        title = self.title
        if t is not None:
            title += f" t={round(t, 3)}"
        self.ax.set_title(title, loc="left")

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


class ConvectionVisualizer(ImageVisualizer):
    def __init__(
        self,
        show: float = True,
    ) -> None:
        ax_args = {
            "title": "Convection",
            "ylabel": "spatial y",
            "yticks": [0, 32, 63],
            "yticklabels": [-1, 0, 1],
            "xlabel": "spatial x",
            "xticks": [0, 48, 95],
            "xticklabels": [0, r"$\pi$", r"2$\pi$"],
        }
        super().__init__(
            size=(64, 96), vrange=(-0.1, 0.2), cmap="coolwarm", show=show, ax_args=ax_args
        )


class CylinderVisualizer(ImageVisualizer):
    def __init__(
        self,
        vrange: Tuple[float, float],
        title: str = "Cylinder",
        show: float = True,
    ) -> None:
        ax = {
            "title": title,
        }
        super().__init__(size=(128, 512), vrange=vrange, cmap="coolwarm", show=show, ax_args=ax)
