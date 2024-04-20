from typing import Optional

import firedrake as fd
import matplotlib
import numpy as np
from hydrogym.core import CallbackBase, PDEBase
from matplotlib import pyplot as plt


class CylinderVisCallback(CallbackBase):

    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)

        # matplotlib stuff
        matplotlib.use("QtAgg")
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7.5, 3))
        self.ax.set_xlim([-2, 9])
        self.ax.set_ylim([-2, 2])
        plt.show(block=False)

    def __call__(self, iter: int, t: float, flow: PDEBase):
        if super().__call__(iter, t, flow):
            self.render(flow)

    def render(self, flow: PDEBase):
        levels = np.linspace(-3, 3, 10)
        fd.tricontourf(
            flow.vorticity(),
            levels=levels,
            axes=self.ax,
            cmap="RdBu",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class H5DatasetCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)

    def __call__(self, iter: int, t: float, flow: PDEBase):
        if super().__call__(iter, t, flow):
            # vort = flow.vorticity()
            # mesh = vort.function_space().mesh()
            # data = vort.dat.data
            pass
