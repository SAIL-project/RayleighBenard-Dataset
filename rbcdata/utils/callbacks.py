import os
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from rbcdata.sim.rbc_env import RayleighBenardEnv
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

    def average(self, window: int = 30):
        return np.mean(self.nusselts[-window:])

    def close(self):
        df = pd.DataFrame({"nusselt": np.array(self.nusselts), "time": np.array(self.time)})
        # Save nusselt numbers to file
        df.to_hdf("nusselt.h5", key="df", mode="w")
        # Plot nusselt number
        fig, ax = plt.subplots()

        # Plot lift
        ax.set_xlabel("time")
        ax.set_ylabel("Nusselt Number")
        ax.plot(self.time, self.nusselts)
        ax.tick_params(axis="y")

        ax.grid()
        fig.savefig("nusselt.png")
        plt.close(fig)


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


class LogActionCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.data = []

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            datum = dict(zip(range(len(env.action)), env.action))
            datum["t"] = info["t"]
            self.data.append(datum)

    def close(self):
        pd.DataFrame(self.data).to_hdf("actions.h5", key="df", mode="w")


class LogDatasetCallback(CallbackBase):
    def __init__(
        self,
        env: RayleighBenardEnv,
        seed: int,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)

        # Create dataset on disk
        file_name = f"dataset/ra{env.cfg.ra}/rbc{seed}.h5"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.file = h5py.File(file_name, "w")

        # Create datasets for Temperature and velocity field
        steps = env.episode_steps // interval
        self.states = self.file.create_dataset(
            "states",
            (steps, 3, env.cfg.N[0], env.cfg.N[1]),
            chunks=(10, 3, env.cfg.N[0], env.cfg.N[1]),
            compression="gzip",
            dtype=np.float32,
        )
        self.actions = self.file.create_dataset(
            "action",
            (steps, env.action_segments),
            chunks=(10, env.action_segments),
            compression="gzip",
            dtype=np.float32,
        )

        # Save commonly used parameters of the simulation
        self.file.attrs["seed"] = seed
        self.file.attrs["steps"] = steps
        self.file.attrs["episode_length"] = env.episode_length
        self.file.attrs["dt"] = env.cfg.dt * interval
        self.file.attrs["N"] = env.cfg.N
        self.file.attrs["ra"] = env.cfg.ra
        self.file.attrs["pr"] = env.cfg.pr
        self.file.attrs["bcT"] = env.cfg.bcT
        self.file.attrs["domain"] = np.array(env.simulation.domain).astype(np.float32)
        self.file.attrs["action_segments"] = env.action_segments
        self.file.attrs["action_limit"] = env.action_limit
        self.file.attrs["action_duration"] = env.action_duration
        self.file.attrs["action_start"] = env.action_start

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            idx = info["step"] // self.interval
            self.states[idx] = env.get_state()
            self.actions[idx] = env.get_action()

    def close(self):
        self.file.close()
