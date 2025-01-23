import logging
import os
from os.path import join

import hydra
import wandb
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from rbcdata.callbacks.sb3_callbacks import NusseltCallback
from rbcdata.env.rbc_env import RayleighBenardEnv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="sbsa")
def main(cfg: DictConfig) -> None:
    # Configure logging
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    # wandb
    # run = wandb.init(
    #     project="sb3-single-agent",
    #     config=dict(cfg),
    #     sync_tensorboard=True,
    #     dir=cfg.output_dir,
    # )
    # sb3 logging
    logger = configure(join(cfg.output_dir, "log"), ["stdout", "log", "json", "tensorboard"])
    logger.info(f"Set log directory to {cfg.output_dir}")

    # Construct the evaluation and training environments
    def create_env(env_cfg, render_mode=None):
        env = RayleighBenardEnv(env_cfg, render_mode=render_mode)
        env = FlattenObservation(env)
        env = FrameStackObservation(env, cfg.sb3.frame_stack)
        return env

    train_env = make_vec_env(
        lambda: create_env(cfg.train_env),
        cfg.sb3.nr_processes,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = make_vec_env(
        lambda: create_env(cfg.eval_env),
        cfg.sb3.nr_eval_processes,
        vec_env_cls=SubprocVecEnv,
    )

    # Parameters
    steps_per_iteration = cfg.sb3.ppo.episodes_update * int(
        cfg.train_env.episode_length / cfg.train_env.action_duration
    )

    # Construct the agent
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=steps_per_iteration,
        learning_rate=cfg.sb3.ppo.lr,
        batch_size=cfg.sb3.ppo.batch_size,
        gamma=cfg.sb3.ppo.gamma,
        ent_coef=cfg.sb3.ppo.ent_coef,
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
        eval_env,
        best_model_save_path=dir_model,
        log_path=dir_log,
        eval_freq=cfg.sb3.eval_every * steps_per_iteration,
        deterministic=True,
        render=False,
    )

    # eval_callback = EvaluationCallback(
    #    env=eval_env,
    #    save_model=True,
    #    save_path=dir_model,
    #    freq=cfg.sb3.eval_every * steps_per_iteration,
    # )

    # video_dir = join(cfg.output_dir, "video")
    # os.makedirs(video_dir, exist_ok=True)
    # vis_callback = EvaluationVisualizationCallback(
    #     env=viz_env,
    #     freq=cfg.sb3.eval_every * steps_per_iteration,
    #     path=video_dir,
    # )

    callbacks = [
        NusseltCallback(),
        #  vis_callback,
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
    eval_env.close()
    run.finish()


if __name__ == "__main__":
    main()
