import logging
import pathlib
import tempfile
from typing import Optional

from matplotlib import animation
from matplotlib import pyplot as plt

import wandb
from rbcdata.callbacks.callbacks import CallbackBase
from rbcdata.utils.rbc_field import RBCField


class LogNusseltNumberCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        wandb.define_metric("sim_time")
        wandb.define_metric("run/nusselt", step_metric="sim_time")

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            # state = env.simulation.state
            # nusselt = env.simulation.compute_nusselt(state)
            wandb.log(
                {
                    "sim_time": info["t"],
                    "run/nusselt": info["nusselt"],
                    "run/nusselt_obs": info["nusselt_obs"],
                }
            )


class LogVisualizationCallback(CallbackBase):
    def __init__(
        self,
        action_limit: float,
        video: bool = True,
        interval: Optional[int] = 1,
    ):
        super().__init__(interval=interval)
        wandb.define_metric("sim_time")
        wandb.define_metric("run/visualization", step_metric="sim_time")
        self.action_limit = action_limit
        self.sequence = []
        self.video = video

        # suppress matplotlib logging
        logger = logging.getLogger("matplotlib.animation")
        logger.setLevel(logging.ERROR)

    def __call__(self, env, obs, reward, info):
        if super().__call__(env, obs, reward, info):
            state = env.simulation.state

            images = []
            for field in [RBCField.T, RBCField.UY, RBCField.UX]:
                fig, _, _ = self.plot_field(state, field)
                images.append(wandb.Image(fig, caption=field.name))
                plt.close(fig)
            self.sequence.append(state)
            wandb.log({"run/visualization": images, "sim_time": info["t"]})

    def close(self):
        if self.video:
            print("Generating videos...", end="")
            # generate videos
            videos = []
            for field in [RBCField.T, RBCField.UX, RBCField.UY]:
                videos.append(
                    wandb.Video(
                        self.sequence2video(self.sequence, "state", field),
                        caption=field.name,
                    )
                )

            # log to wandb
            for i, field in enumerate([RBCField.T, RBCField.UX, RBCField.UY]):
                wandb.log({f"run/video_{field.name}": videos[i]})
            print(" done.")

    def plot_field(self, x, field: RBCField):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_axis_off()
        if field == RBCField.T:
            vmin, vmax = 1, 2
        else:
            vmin, vmax = None, None

        im = ax.imshow(x[field], cmap="coolwarm", vmin=vmin, vmax=vmax)

        return fig, ax, im

    def sequence2video(
        self,
        sequence,
        caption: str,
        field: RBCField,
        colormap="coolwarm",
        fps=2,
    ) -> str:
        # set up path
        path = pathlib.Path(f"{tempfile.gettempdir()}/rbcdata").resolve()
        path.mkdir(parents=True, exist_ok=True)
        # config fig
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_axis_off()

        if colormap == "binary":
            vmin, vmax = None, None
        elif field == RBCField.T:
            vmin, vmax = 1, 2
        else:
            vmin, vmax = None, None

        # create video
        artists = []
        steps = len(sequence)
        for i in range(steps):
            artists.append(
                [ax.imshow(sequence[i][field], cmap=colormap, vmin=vmin, vmax=vmax)],
            )
        ani = animation.ArtistAnimation(fig, artists, blit=True)

        # save as mp4
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
        path = path / f"video_{field}_{caption}.mp4"
        ani.save(path, writer=writer)
        plt.close(fig)
        return str(path)


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
