import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

from rbcdata.envs.rbc_ma_env import RayleighBenardMultiAgentEnv


class LogCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        print(f"Starting episode in env {env_index} on worker {worker.worker_index}")
        # Nusselt Number
        episode.user_data["nusselts"] = []

    def on_episode_step(self, *, worker, base_env: BaseEnv, episode, env_index, **kwargs):
        env: RayleighBenardMultiAgentEnv = base_env.envs[0]
        print(f"Step {episode.total_env_steps} in env {env_index} on worker {worker.worker_index}")
        # Nusselt Number
        episode.user_data["nusselts"].append(env.get_global_nusselt())

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index):
        nusselt = np.mean(episode.user_data["nusselts"])
        episode.custom_metrics["nusselt"] = nusselt
        episode.hist_data["nusselts"] = episode.user_data["nusselts"]

    def on_train_result(self, *, algorithm, metrics_logger, result):
        # TODO finish
        custom_metrics = result[ENV_RUNNER_RESULTS]["custom_metrics"]
        pole_angle = custom_metrics["pole_angle"]
        var = np.var(pole_angle)
        mean = np.mean(pole_angle)
        custom_metrics["pole_angle_var"] = var
        custom_metrics["pole_angle_mean"] = mean
        # We are not interested in these original values
        del custom_metrics["pole_angle"]
        del custom_metrics["num_batches"]
