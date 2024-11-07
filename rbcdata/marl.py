import time
from os.path import join

import hydra
import rootutils
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
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


@hydra.main(version_base=None, config_path="config", config_name="marl")
def main(cfg: DictConfig) -> None:
    # Configure logging
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    # wandb
    wandb.init(
        project="sb3-multi-agent",
        config=dict(cfg),
        sync_tensorboard=True,
        dir=cfg.output_dir,
    )
    # sb3 logging
    logger = configure(join(cfg.output_dir, "log"), ["stdout", "log", "json", "tensorboard"])
    logger.info(f"Set log directory to {cfg.output_dir}")

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


if __name__ == "__main__":
    main()
