import numpy as np
import os
from os.path import join
import logging

import hydra
from omegaconf import DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig

from rbcdata.sim.rbc_env import RayleighBenardEnv

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="run_SA")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    save_dir = join(cfg.output_dir, "model")
    os.makedirs(save_dir, exist_ok=True)
    
    env = RayleighBenardEnv(cfg)
    # check_env(env)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100)
    model.save(save_dir, "PPOmodelRBC")
    # Random agent, before training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
    
# # Perform some random actions on the agent
# obs, info = env.reset()
# n_steps = 10
# for _ in range(n_steps):
#     # random action
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if truncated or terminated:
#         obs, info = env.reset()