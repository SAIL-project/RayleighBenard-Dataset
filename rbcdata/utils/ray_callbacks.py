import datetime
import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from rbcdata.envs.rbc_ma_env import RayleighBenardMultiAgentEnv


class LogCallback(DefaultCallbacks):
    def __init__(self):
        session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"logs/ray/Session{session_id}"

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # self.log(f"Starting episode {episode.episode_id}")
        # Nusselt Number
        episode.user_data["nusselts"] = []

    def on_episode_step(self, *, worker, base_env: BaseEnv, episode, env_index, **kwargs):
        # self.log(f"Step {episode.total_env_steps} episode {episode.episode_id}")
        # Nusselt Number
        env: RayleighBenardMultiAgentEnv = base_env.envs[0]
        episode.user_data["nusselts"].append(env.get_global_nusselt())

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index):
        # self.log(f"Ending episode {episode.episode_id}")
        # Nusselt Number
        nusselts = episode.user_data["nusselts"]
        episode.custom_metrics["nusselt_mean"] = np.mean(nusselts)
        episode.custom_metrics["nusselt_var"] = np.var(nusselts)
        self.log_episode_nusselt(nusselts, episode.episode_id)

    def on_train_result(self, *, algorithm, metrics_logger: MetricsLogger, result):
        # TM: What is metrics_logger doing?
        return
        custom_metrics = result[ENV_RUNNER_RESULTS]["custom_metrics"]
        # Log mean of nusselt number across all episodes
        for key in ["nusselt_mean_mean", "nusselt_var_mean"]:
            if key in custom_metrics:
                metrics_logger.log_value(key, custom_metrics[key])

    def on_sample_end(
        self,
        *,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        samples: SampleBatch,
        worker: EnvRunner | None = None,
        **kwargs,
    ) -> None:
        self.log(f"On sample end... Collected {len(samples)} samples")

    def on_evaluate_start(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger | None = None,
    ) -> None:
        self.log("Starting evaluation...")

    def on_evaluate_end(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger | None = None,
        evaluation_metrics: dict,
    ) -> None:
        self.log("Ending evaluation...")

    def log(self, msg):
        logger = logging.getLogger("ray")
        logger.info(msg)

    def log_episode_nusselt(self, nusselts, index):
        Path(f"{self.log_dir}/episodes").mkdir(parents=True, exist_ok=True)
        # Plot nusselt number
        fig, ax = plt.subplots()
        # Plot lift
        ax.set_xlabel("time")
        ax.set_ylabel("Nusselt Number")
        ax.set_ylim(0, 5)
        ax.plot(range(len(nusselts)), nusselts)
        ax.tick_params(axis="y")
        ax.grid()
        fig.savefig(f"{self.log_dir}/episodes/nusselt_{index}.png")
        plt.close(fig)
