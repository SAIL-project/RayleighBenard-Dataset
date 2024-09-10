import os
from typing import List, Optional

import h5py
import hydra
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from rbcdata.env.rbc_env import RayleighBenardEnv
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


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[CallbackBase]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[CallbackBase] = []

    if not callbacks_cfg:
        print("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


class RBCVisCallback(CallbackBase):
    def __init__(
        self,
        size: List[int],
        bcT: List[float],
        action_limit: float,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.window = RBCFieldVisualizer(
            size=size,
            vmin=bcT[1],
            vmax=bcT[0] + action_limit,
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
            state = env.simulation.state
            self.nusselts.append(env.simulation.compute_nusselt(state))
            self.time.append(info["t"])

    def close(self):
        df = pd.DataFrame({"nusselt": np.array(self.nusselts), "time": np.array(self.time)})
        # Save nusselt numbers to file
        df.to_hdf("nusselt.h5", key="df", mode="w")
        # Plot nusselt number
        fig, ax = plt.subplots()

        # Plot lift
        ax.set_xlabel("time")
        ax.set_ylabel("Nusselt Number")
        ax.set_ylim(0, 5)
        ax.plot(self.time, self.nusselts)
        ax.tick_params(axis="y")

        ax.grid()
        fig.savefig("nusselt.png")
        plt.close(fig)


class ControlVisCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.window = RBCActionVisualizer()

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
        seed: int,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.seed = seed
        self.initialized = False

    def initialize(self, env: RayleighBenardEnv):
        # Create dataset on disk
        file_name = f"dataset/ra{env.cfg.ra}/rbc{self.seed}.h5"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.file = h5py.File(file_name, "w")

        # Create datasets for Temperature and velocity field
        steps = env.episode_steps // self.interval
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
        self.file.attrs["seed"] = self.seed
        self.file.attrs["steps"] = steps
        self.file.attrs["episode_length"] = env.episode_length
        self.file.attrs["dt"] = env.cfg.dt * self.interval
        self.file.attrs["N"] = env.cfg.N
        self.file.attrs["ra"] = env.cfg.ra
        self.file.attrs["pr"] = env.cfg.pr
        self.file.attrs["bcT"] = env.cfg.bcT
        self.file.attrs["domain"] = np.array(env.simulation.domain).astype(np.float32)
        self.file.attrs["action_segments"] = env.action_segments
        self.file.attrs["action_limit"] = env.action_limit
        self.file.attrs["action_duration"] = env.action_duration
        self.file.attrs["action_start"] = env.action_start
        # Set initialized flag
        self.initialized = True

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            if not self.initialized:
                self.initialize(env)

            idx = info["step"] // self.interval - 1
            self.states[idx] = env.get_state()
            self.actions[idx] = env.get_action()

    def close(self):
        self.file.close()


class SweepMetricCallback(CallbackBase):
    def __init__(
        self,
        action_start: float,
        action_end: float,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.action_start = action_start
        self.action_end = action_end
        self.nusselt_start = []
        self.nusselt_end = []

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            if info["t"] < self.action_start:
                return
            elif info["t"] < self.action_end:
                self.nusselt_start.append(env.simulation.compute_nusselt(obs))
            else:
                self.nusselt_end.append(env.simulation.compute_nusselt(obs))

    def result(self):
        return {
            "nu_mean_action": np.mean(self.nusselt_start).item(),
            "nu_std_action": np.std(self.nusselt_start).item(),
            "nu_mean_end": np.mean(self.nusselt_end).item(),
            "nu_std_end": np.std(self.nusselt_end).item(),
        }

    def close(self):
        with open("sweep_metric.yml", "w") as outfile:
            yaml.dump(self.result(), outfile)
