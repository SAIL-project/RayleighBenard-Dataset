import numpy as np
import os
from os.path import join
import logging
import time

import hydra
from omegaconf import DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig

from rbcdata.sim.rbc_env import RayleighBenardEnv

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Callbacks
from stable_baselines3.common.callbacks import CheckpointCallback
from rbcdata.utils.callbacks_sb import LogNusseltCallback, EvalCallback
from stable_baselines3.common.callbacks import EvalCallback as eval_cb

logger = logging.getLogger(__name__)

# Configuration stable baselines. TODO move to config file
NR_PROCESSES = 6
TRAIN_STEPS = 120*1000
EVAL_EPS = 1 * NR_PROCESSES
EVAL_EVERY = 2    # Evaluate every so many training steps. Probably best if it is a multiple of NR_PROCESSES, because train_steps increases by NR_PROCESSES.
ALGO = PPO

#TODO Note that things like Nusselt number can also be returned in the info dict of the environment. This is not yet used here.

@hydra.main(version_base=None, config_path="config", config_name="run_SA")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    dir_model_besteval = join(cfg.output_dir, "model_besteval")
    dir_model_checkpoint_train = join(cfg.output_dir, "model_checkpoint_train")
    os.makedirs(dir_model_besteval, exist_ok=True)
    os.makedirs(dir_model_checkpoint_train, exist_ok=True)
    
    # Construct the evaluation and training environments
    eval_env = make_vec_env(lambda: RayleighBenardEnv(cfg),
                            NR_PROCESSES,
                            vec_env_cls=SubprocVecEnv,
                            vec_env_kwargs=dict(start_method='fork')
    )

    # eval_env = RayleighBenardEnv(cfg) # Single environment for evaluation

    train_env = make_vec_env(lambda: RayleighBenardEnv(cfg),
                             NR_PROCESSES,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs=dict(start_method='fork')
    )
    # train_env.reset()
    model = PPO("MlpPolicy",
                train_env,
                learning_rate=3e-4,
                verbose=1,
                n_steps=40,
                batch_size=30,
                gamma=0.95
    )

    checkpoint_callback_eval = CheckpointCallback(
        save_freq=4,
        save_path=dir_model_besteval,
        name_prefix="PPOmodelRBC_besteval",
    )
    checkpoint_callback_eval.init_callback(model)
    
    checkpoint_callback_training = CheckpointCallback(
        save_freq=200,
        save_path=dir_model_checkpoint_train,
        name_prefix="PPOmodelRBC"
    )

    logNusselt = LogNusseltCallback(1)
    logNusselt.init_callback(model)
    
    eval_callback = eval_cb(
        eval_env=eval_env,
        callback_on_new_best=checkpoint_callback_eval,
        callback_after_eval=logNusselt,
        n_eval_episodes=EVAL_EPS,
        eval_freq=EVAL_EVERY,
        deterministic=True
    )

    # eval_callback = EvalCallback(
    #     eval_env=eval_env,
    #     callback_on_new_best=checkpoint_callback_eval,
    #     callback_after_eval=logNusselt,
    #     n_eval_episodes=EVAL_EPS,
    #     eval_freq=EVAL_EVERY,
    #     deterministic=True,
    #     render=False
    # )

    # callbacks = [LogNusseltCallback(10*NR_PROCESSES), eval_callback, checkpoint_callback_training]
    callbacks = [eval_callback]

    # start = time.time()
    model.learn(total_timesteps=TRAIN_STEPS,
                progress_bar=True,
                callback=callbacks
    )

    # Right now the best model is saved upon best evaluation score. We don't need to save the model again.
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    # logger.info(f"Training completed after {time.time() - start} seconds. Mean evaluation reward {mean_reward:.2f} +/- {std_reward:.2f}")
    # model.save(final_model_dir, "PPOmodelRBC")
    train_env.close()


if __name__ == "__main__":
    main()