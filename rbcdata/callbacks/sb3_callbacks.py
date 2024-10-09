import os
from typing import Optional

import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, Video
from stable_baselines3.common.vec_env import VecEnv


class NusseltCallback(BaseCallback):
    def __init__(
        self,
        freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        for info in infos:
            self.logger.record_mean("train/nusselt_obs", info["nusselt_obs"])
            self.logger.record_mean("train/nusselt", info["nusselt"])
        return True


class EvaluationCallback(BaseCallback):
    def __init__(
        self,
        env: VecEnv,
        save_model: bool = False,
        save_path: Optional[str] = None,
        freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.freq = freq
        self.env = env
        self.save_model = save_model
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.n_calls % self.freq == 0:
            self.logger.info(f"Evaluating model at {self.num_timesteps} timesteps")
            nr_envs = self.env.num_envs
            rewards_list = []

            # env loop
            obs = self.env.reset()
            done = np.zeros(nr_envs)
            while not done.any():
                # env step
                actions, _ = self.model.predict(obs, deterministic=True)
                _, rewards, done, infos = self.env.step(actions)
                for id in range(nr_envs):
                    rewards_list.append(rewards[id])
                    self.logger.record_mean("eval/reward", rewards[id])
                    self.logger.record_mean("eval/nusselt", infos[id]["nusselt"])
                    self.logger.record_mean("eval/nusselt_obs", infos[id]["nusselt_obs"])

            # check for new best model
            mean_reward = np.mean(rewards_list)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.logger.info(f"New best model with mean reward {mean_reward}")
                if self.save_model:
                    self.model.save(os.path.join(self.save_path, "best_model"))


class EvaluationVisualizationCallback(BaseCallback):
    def __init__(
        self,
        env: VecEnv,
        freq: int = 1,
        path: Optional[str] = ".",
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.freq = freq
        self.env = env
        self.path = path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.n_calls % self.freq == 0:
            self.logger.info(f"Visualizing model at {self.num_timesteps} timesteps")
            # data
            screens = []
            actions = []
            nusselts = []
            # env loop
            obs = self.env.reset()
            done = np.zeros(1)
            while not done.any():
                # env step
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = self.env.step(action)
                screen = self.env.render()

                # save data
                nusselts.append(infos[0]["nusselt"])
                screens.append(screen.transpose(2, 0, 1))
                actions.append(action.squeeze())

            # episode stats
            self.plot_episode_nusselt(nusselts)
            self.plot_actions(actions)
            self.logger.record(
                "eval/video",
                Video(torch.from_numpy(np.asarray([screens])), fps=2),
                exclude=("stdout", "log", "json", "csv"),
            )

    def plot_episode_nusselt(self, nusselts):
        # Plot nusselt number
        fig, ax = plt.subplots()
        # Plot lift
        ax.set_xlabel("time")
        ax.set_ylabel("Nusselt Number")
        ax.set_ylim(1, 4)
        ax.plot(range(len(nusselts)), nusselts)
        ax.tick_params(axis="y")
        ax.grid()
        self.logger.record(
            "eval/nusselt_plot",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )

    def plot_actions(self, actions):
        # Plot nusselt number
        fig, ax = plt.subplots()
        # Plot amplitude
        ax.set_xlabel("segements")
        ax.set_ylabel("amplitude")
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis="y")
        ax.grid()
        # plot actions
        artists = []
        for action in actions:
            container = ax.plot(range(len(action)), action, color="blue")
            artists.append(container)

        ani = animation.ArtistAnimation(fig=fig, artists=artists)
        writer = animation.FFMpegWriter(fps=2)
        ani.save(f"{self.path}/actions_{self.n_calls}.mp4", writer=writer)

        self.logger.record(
            "eval/action_plot",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
