import logging
import os
from os.path import join

import hydra
import wandb
from gymnasium.wrappers.flatten_observation import FlattenObservation
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from stable_baselines3 import PPO

# Callbacks
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from rbcdata.callbacks.sb3_callbacks import (
    EvaluationCallback,
    EvaluationVisualizationCallback,
    NusseltCallback,
)
from rbcdata.env.rbc_env import RayleighBenardEnv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="sarl")
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
    )
    # sb3 logging
    logger = configure(join(cfg.output_dir, "log"), ["stdout", "log", "json", "tensorboard"])
    logger.info(f"Set log directory to {cfg.output_dir}")

    # Construct the evaluation and training environments
    train_env = make_vec_env(
        lambda: FlattenObservation(RayleighBenardEnv(cfg.train_env)),
        cfg.sb3.nr_processes,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="fork"),
    )

    eval_env = make_vec_env(
        lambda: FlattenObservation(RayleighBenardEnv(cfg.eval_env)),
        cfg.sb3.nr_eval_processes,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="fork"),
    )

    viz_env = make_vec_env(
        lambda: FlattenObservation(RayleighBenardEnv(cfg.eval_env, render_mode="rgb_array")),
        1,
        vec_env_cls=DummyVecEnv,
    )

    # Parameters
    steps_per_iteration = cfg.sb3.ppo.episodes_update * int(
        cfg.train_env.episode_length / cfg.train_env.action_duration
    )

    # Construct the agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.sb3.ppo.lr,
        verbose=1,
        n_steps=steps_per_iteration,
        batch_size=cfg.sb3.ppo.batch_size,
        gamma=cfg.sb3.ppo.gamma,
        ent_coef=cfg.sb3.ppo.ent_coef,
    )

    # Callbacks
    # train checkpoint
    dir_model = join(cfg.output_dir, "model")
    os.makedirs(dir_model, exist_ok=True)
    checkpoint_callback_training = CheckpointCallback(
        save_freq=cfg.sb3.train_checkpoint_every
        * int(cfg.train_env.episode_length / cfg.train_env.action_duration),
        save_path=dir_model,
        name_prefix="PPO_train",
    )

    # evaluation callback
    eval_callback = EvaluationCallback(
        env=eval_env,
        save_model=True,
        save_path=dir_model,
        freq=cfg.sb3.eval_every * steps_per_iteration,
    )

    video_dir = join(cfg.output_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    vis_callback = EvaluationVisualizationCallback(
        env=viz_env,
        freq=cfg.sb3.eval_every * steps_per_iteration,
        path=video_dir,
    )

    callbacks = [
        NusseltCallback(),
        vis_callback,
        eval_callback,
        checkpoint_callback_training,
        WandbCallback(
            verbose=1,
        ),
    ]

    # Train the model
    model.set_logger(logger)
    model.learn(total_timesteps=cfg.sb3.train_steps, progress_bar=True, callback=callbacks)

    # Right now the best model is saved upon best evaluation score.
    # We don't need to save the model again.
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    # logger.info(f"Training completed after {time.time() - start} seconds.
    # Mean evaluation reward {mean_reward:.2f} +/- {std_reward:.2f}")
    # model.save(final_model_dir, "PPOmodelRBC")
    train_env.close()
    run.finish()


if __name__ == "__main__":
    main()
