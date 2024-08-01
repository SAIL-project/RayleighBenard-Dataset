from typing import List, Optional

from matplotlib import pyplot as plt
from tqdm import tqdm

from rbcdata.utils.rbc_field import RBCField
from rbcdata.vis.rbc_action_visualizer import RBCActionVisualizer
from rbcdata.vis.rbc_field_visualizer import RBCFieldVisualizer


class CallbackBase:
    def __init__(self, interval: int = 1):
        self.interval = interval

    def __call__(self, env, obs, reward, info) -> bool:
        return info["step"] % self.interval == 0

    def close(self):
        pass


class RBCVisCallback(CallbackBase):
    def __init__(
        self,
        size: List[int],
        vmin: float,
        vmax: float,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.window = RBCFieldVisualizer(
            size=size,
            vmin=vmin,
            vmax=vmax,
        )

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            state = env.get_state()
            self.window.draw(
                state[RBCField.T],
                state[RBCField.UX],
                state[RBCField.UY],
                info["t"],
            )

    def close(self):
        self.window.close()


class TqdmCallback(CallbackBase):
    def __init__(
        self,
        total: int,
        position: int = 0,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.pbar = tqdm(
            total=total,
            leave=False,
            position=position,
        )

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            t = info["t"]
            self.pbar.update(t - self.pbar.n)

    def close(self):
        self.pbar.close()


class LogNusseltNumberCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.nusselts = []
        self.time = []

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            self.nusselts.append(env.simulation.compute_nusselt())
            self.time.append(info["t"])

    def close(self):
        fig, ax = plt.subplots()

        # Plot lift
        ax.set_xlabel("time")
        ax.set_ylabel("Nusselt Number")
        ax.plot(self.time, self.nusselts)
        ax.tick_params(axis="y")

        ax.grid()
        fig.savefig("nusselt.png")


class ControlVisCallback(CallbackBase):
    def __init__(
        self,
        x_domain,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.window = RBCActionVisualizer(x_domain=x_domain)

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            self.window.draw(env.action_effective, info["t"])

    def close(self):
        self.window.close()
