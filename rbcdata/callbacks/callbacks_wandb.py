import logging
import pathlib
import tempfile
from typing import Optional

import numpy as np
import wandb
from matplotlib import animation
from matplotlib import pyplot as plt

from rbcdata.callbacks.callbacks import CallbackBase


class LogNusseltNumberCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
        nr_episodes: int = 1,
    ):
        super().__init__(interval=interval)
        for idx in range(nr_episodes):
            wandb.define_metric(f"ep{idx}/time")
            wandb.define_metric(
                f"ep{idx}/nusselt_state", step_metric=f"ep{idx}/time", summary="mean"
            )
            wandb.define_metric(
                f"ep{idx}/nusselt_obs", step_metric=f"ep{idx}/time", summary="mean"
            )

    def __call__(self, env, obs, reward, info, episode_idx):
        if super().__call__(env, obs, reward, info):
            # state = env.simulation.state
            # nusselt = env.simulation.compute_nusselt(state)
            wandb.log(
                {
                    f"ep{episode_idx}/time": info["t"],
                    f"ep{episode_idx}/nusselt_state": info["nusselt"],
                    f"ep{episode_idx}/nusselt_obs": info["nusselt_obs"],
                }
            )


class LogVisualizationCallback(CallbackBase):
    def __init__(
        self,
        action_limit: float,
        fps: int = 4,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        self.action_limit = action_limit
        self.fps = fps
        self.screens = []
        self.actions = []

    def __call__(self, env, obs, reward, info, episode_idx=0):
        if super().__call__(env, obs, reward, info):
            self.screens.append(env.render().transpose(2, 0, 1))
            self.ep_idx = episode_idx

    def reset(self):
        if self.ep_idx is not None:
            wandb.log(
                {
                    f"ep{self.ep_idx}/visualization": wandb.Video(
                        np.asarray(self.screens), fps=self.fps, format="mp4"
                    )
                }
            )
        self.screens = []
        self.actions = []


class LogActionCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        wandb.define_metric("sim_time")
        wandb.define_metric("run/action", step_metric="sim_time")

        # plot
        self.actions = []

        # suppress matplotlib logging
        logger = logging.getLogger("matplotlib.animation")
        logger.setLevel(logging.ERROR)

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            action = env.last_action
            self.actions.append(action)
            # plot action
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.set_xlabel("segements")
            ax.set_ylabel("amplitude")
            ax.set_ylim(-1.1, 1.1)
            ax.tick_params(axis="y")
            ax.grid()
            # save container for video
            ax.plot(range(len(action)), action, color="blue")
            im = wandb.Image(fig, caption="action")
            plt.close(fig)
            # log to wandb
            wandb.log(
                {
                    "sim_time": info["t"],
                    "run/action": im,
                }
            )

    def close(self):
        # plot actions
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlabel("segements")
        ax.set_ylabel("amplitude")
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis="y")
        ax.grid()
        artists = []
        for action in self.actions:
            artists.append(ax.plot(range(len(action)), action, color="blue"))
        ani = animation.ArtistAnimation(fig=fig, artists=artists)
        writer = animation.FFMpegWriter(fps=2)
        path = pathlib.Path(f"{tempfile.gettempdir()}/rbcdata").resolve()
        path.mkdir(parents=True, exist_ok=True)
        path = f"{path}/actions.mp4"
        ani.save(path, writer=writer)

        # wandb
        vid = wandb.Video(path, caption="actions")
        wandb.log({"run/video_actions": vid})
