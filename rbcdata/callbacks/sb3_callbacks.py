import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Figure, Video
from supersuit import pettingzoo_env_to_vec_env_v1

from rbcdata.env.rbc_env import RayleighBenardEnv
from rbcdata.env.rbc_ma_env import RayleighBenardMultiAgentEnv
from rbcdata.env.wrapper.ma_flatten import ma_flatten


class RBCEvaluationCallback(BaseCallback):
    def __init__(
        self,
        env: RayleighBenardMultiAgentEnv | RayleighBenardEnv,
        freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.freq = freq
        if isinstance(env, RayleighBenardMultiAgentEnv):
            self.env = pettingzoo_env_to_vec_env_v1(
                ma_flatten(env),
            )
        else:
            self.env = make_vec_env(FlattenObservation(env))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.n_calls % self.freq == 0:
            # episode stats
            nusselts = []
            screens = []
            # env loop
            obs, _ = self.env.reset()
            tnc = np.array([0])
            while not tnc.any():
                # env step
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, _, tnc, infs = self.env.step(action)
                screen = self.env.render()

                # stats
                info = infs[0]
                nusselt = info["nusselt"]
                nusselt_obs = info["nusselt_obs"]

                # logging
                nusselts.append(nusselt)
                screens.append(screen.transpose(2, 0, 1))
                self.logger.record_mean("eval/reward", reward.mean())
                self.logger.record_mean("eval/nusselt", nusselt)
                self.logger.record_mean("eval/nusselt_obs", nusselt_obs)

            # episode stats
            self.log_episode_nusselt(nusselts)
            self.logger.record(
                "eval/video",
                Video(torch.from_numpy(np.asarray([screens])), fps=2),
                exclude=("stdout", "log", "json", "csv"),
            )

    def log_episode_nusselt(self, nusselts):
        # Plot nusselt number
        fig, ax = plt.subplots()
        # Plot lift
        ax.set_xlabel("time")
        ax.set_ylabel("Nusselt Number")
        ax.set_ylim(0, 5)
        ax.plot(range(len(nusselts)), nusselts)
        ax.tick_params(axis="y")
        ax.grid()
        self.logger.record(
            "eval/nusselt_plot",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
