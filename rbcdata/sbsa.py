import logging
import os
from os.path import join

import hydra
import torch
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from rbcdata.callbacks.sb3_callbacks import NusseltCallback
from rbcdata.env.rbc_env import RayleighBenardEnv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="sbsa")
def main(cfg: DictConfig) -> None:
    # Configure logging
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    # wandb
    run = wandb.init(
        project="sb3-single-agent",
        config=dict(cfg),
        sync_tensorboard=True,
        dir=cfg.output_dir,
        tags=cfg.tags,
        notes=cfg.notes,
    )
    # sb3 logging
    logger = configure(join(cfg.output_dir, "log"), ["stdout", "log", "json", "tensorboard"])
    logger.info(f"Set log directory to {cfg.output_dir}")

    # Construct the evaluation and training environments
    def create_env(env_cfg, render_mode=None):
        env = RayleighBenardEnv(
            env_cfg, render_mode=render_mode, reward_shaping=cfg.sb3.reward_shaping
        )
        env = FlattenObservation(env)
        env = FrameStackObservation(env, cfg.sb3.frame_stack)
        return env

    train_env = make_vec_env(
        lambda: create_env(cfg.train_env),
        cfg.sb3.nr_processes,
        vec_env_cls=SubprocVecEnv,
    )

    test_env = make_vec_env(
        lambda: create_env(cfg.test_env),
        cfg.sb3.nr_eval_processes,
        vec_env_cls=SubprocVecEnv,
    )

    # Parameters
    steps_per_iteration = cfg.sb3.ppo.episodes_update * int(
        cfg.train_env.episode_length / cfg.train_env.action_duration
    )

    if cfg.sb3.model == "ppo":
        nr_neurons = cfg.sb3.ppo.nr_neurons
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[nr_neurons, nr_neurons], vf=[nr_neurons, nr_neurons]),
        )
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            n_steps=steps_per_iteration,
            learning_rate=cfg.sb3.ppo.lr,
            batch_size=cfg.sb3.ppo.batch_size,
            gamma=cfg.sb3.ppo.gamma,
            ent_coef=cfg.sb3.ppo.ent_coef,
            verbose=1,
        )
    elif cfg.sb3.model == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            ent_coef=cfg.sb3.sac.ent_coef,
            verbose=1,
        )

    # Callbacks
    dir_model = join(cfg.output_dir, "model")
    dir_log = join(cfg.output_dir, "log")
    # train checkpoint

    os.makedirs(dir_model, exist_ok=True)
    checkpoint_cb_training = CheckpointCallback(
        save_freq=cfg.sb3.train_checkpoint_every
        * int(cfg.train_env.episode_length / cfg.train_env.action_duration),
        save_path=dir_model,
        name_prefix="PPO_train",
    )

    # evaluation callback
    eval_cb = EvalCallback(
        test_env,
        best_model_save_path=dir_model,
        log_path=dir_log,
        eval_freq=cfg.sb3.eval_every * steps_per_iteration,
        deterministic=True,
        render=False,
    )

    callbacks = [
        NusseltCallback(),
        eval_cb,
        checkpoint_cb_training,
        WandbCallback(
            verbose=1,
        ),
    ]

    # Train the model
    model.set_logger(logger)
    model.learn(total_timesteps=cfg.sb3.train_steps, progress_bar=True, callback=callbacks)

    train_env.close()
    test_env.close()
    run.finish()


if __name__ == "__main__":
    main()
