import glob
import os
import time

import hydra
import rootutils
import wandb
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo import MlpPolicy
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1
from wandb.integration.sb3 import WandbCallback

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.callbacks.sb3_callbacks import EvaluationCallback
from rbcdata.env.rbc_ma_env import RayleighBenardMultiAgentEnv
from rbcdata.env.wrapper.ma_flatten import ma_flatten

# TODO: run without local
# TODO: plot local nusselt number
# TODOL check checkpoint


def train_marl(cfg: DictConfig) -> None:
    # logging
    tmp_path = "logging/"
    logger = configure(tmp_path, ["stdout", "log", "json", "tensorboard"])

    # environment
    env = RayleighBenardMultiAgentEnv(cfg.env)
    env = ma_flatten(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(
        env, cfg.nr_envs, num_cpus=cfg.nr_envs, base_class="stable_baselines3"
    )

    eval_env = RayleighBenardMultiAgentEnv(cfg.eval.env, render_mode="rgb_array")

    # callbacks
    callback = CallbackList(
        [
            EvaluationCallback(eval_env, freq=cfg.eval.freq),
            WandbCallback(
                verbose=1,
            ),
        ]
    )

    # Train a single model to play as each agent
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        n_steps=cfg.ppo.n_steps,
        stats_window_size=cfg.ppo.stats_window_size,
        learning_rate=cfg.ppo.learning_rate,
        ent_coef=cfg.ppo.ent_coef,
        batch_size=cfg.ppo.batch_size,
    )
    model.set_logger(logger)
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # save and close the environment
    model.save(f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()


def eval_marl():
    # Evaluation environment
    env = ma_flatten(RayleighBenardMultiAgentEnv())
    observations, infos = env.reset(seed=42)

    # Load agent
    latest_policy = max(glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime)
    PPO.load(latest_policy)

    # Evaluate
    print(f"\nStarting evaluation on {str(env.metadata['name'])}")
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f"Observations: {observations}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
        print(f"Infos: {infos}")
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


@hydra.main(version_base=None, config_path="config", config_name="marl")
def main(cfg: DictConfig) -> None:
    run = wandb.init(
        project="sb3-multi-agent",
        config=dict(cfg),
        sync_tensorboard=True,
    )

    train_marl(cfg)

    run.finish()


if __name__ == "__main__":
    main()
