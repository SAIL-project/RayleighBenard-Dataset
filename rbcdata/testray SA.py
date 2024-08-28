# File for testing single agent reinforcement learning with Raylib
# General python imports
import os
import rootutils
import logging
import hydra    # for loading the configuration
from omegaconf import DictConfig  # datatype for the configuration
from os.path import join

# All Raylib imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env  # allows to register custom environments, every ray worker will have access to it

import gymnasium as gym

from hydra.core.hydra_config import HydraConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
# Environment imports
from rbcdata.sim.rbc_env import RayleighBenardEnv   # the environment that the single agent will interact with

logger = logging.getLogger(__name__)

def checkpoints_to_nodes():
    """
    This function is able to move the local checkpoints to all the nodes in the Ray cluster,
    that need to have access to them for resetting their environments.
    """
    logger.warning("TODO: Not implemented the moving of the checkpoints for remote nodes yet")


def logging_setup_func():
    # ray_rllib_logger = logging.getLogger("ray.rllib")
    logger = logging.getLogger("ray")
    logger.setLevel(logging.DEBUG)


@hydra.main(version_base=None, config_path="config", config_name="run_SA")
def run_env(cfg: DictConfig) -> None:
    """
    This function runs the environment with a single agent using the PPO algorithm from Raylib library using the given config.
    """
    output_dir = HydraConfig.get().runtime.output_dir

    if cfg.sim.load_checkpoint_path is not None and cfg.ray.env_runners > 0:
        checkpoints_to_nodes()

    # Define a Ray runtime_env
    # runtime_env = {"conda": "./environment_rbcdata.yml"}  # (Michiel): this line is used when we need to install dependencies on other worker nodes
    # This line is used if the worker nodes can use an already installed environment from the filesystem
    runtime_env = {
        "conda": "/home/michiel/miniconda3/envs/rbcdata",
        "worker_process_setup_hook": logging_setup_func
    }    
    print(cfg)
    # Structure of the code (based on examples of RLlib)
    # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html  for handling dependencies on the Ray cluster
    # ray.init()  # initialize the Ray runtime 
    # initialize the Ray runtime using the above dictionary. # TODO Look at logging in Ray workers later
    # This actually initiates the so-called Ray Driver
    context = ray.init(
        runtime_env=runtime_env,
        logging_level="info",
        log_to_driver=False,
    ) 
    # setup logger for the driver process
    logging_setup_func()

    logger.info(f"Ray runtime initialized with context: {context}")

    # We need to define an environment (RayleighBenardEnvironment)
    # An RLlib environment or Gym environment consists of: action space (all possible actions), state space
    # an observation by the agent of certain parts of the state (all possible states), and a reward function (feedback agent receives per action).
    # TODO we should probably implement probes in the RayleighBenard environment to get the state of the system, which is then observed by the agent.
    # This is because PPO can probably not handle the full state of the system, as it is a relatively large grid.

    # Do we need to register the environment with gym if we also register it with Raylib?
    # gym.register("RayleighBenardEnv", RayleighBenardEnv)  # register the environment with gym, so that it can be used by RLlib

    # reset the environment to the initial state
    # obs, info = env.reset() # TODO maybe load a checkpoint here from the created checkpoints. Reset is probably handled by Raylib itself.
    register_env("RayleighBenardEnv", lambda config: RayleighBenardEnv(config))

    # We need to define a policy for the single agent that controls the heaters in the RayleighBenard environment
    # TODO I read PPO is robust, so check recommended parameters for our setting, Petting Zoo provides these parameters.
    algo_config = PPOConfig()
    # training params discount factor, learning rate, TODO check if we need to set more parameters like train_batch_size, kl_coeff, etc.
    algo_config = algo_config.training(
        gamma=cfg.rl.ppo.gamma,
        lr=cfg.rl.ppo.lr,
        train_batch_size=int(cfg.sim.episode_length / cfg.sim.dt) * cfg.ray.env_runners,    # these are the total steps (or actions) (total among all workers) that are used for one policy update session (which consists of multiple minibatch updates)
        sgd_minibatch_size=100,  # the number of steps (or actions) that are used for one SGD update
        num_sgd_iter=10,           # This refers to the number of traversals though the complete training batch for updating the policy.
        shuffle_sequences=True,  # shuffle the sequences of experiences in the training batch, this is by default already true.
        entropy_coeff=0.01,  # the entropy coefficient for the policy, which is used to encourage exploration
    ) 
    # TODO Michiel: I think what happens: In each batch the Learner worker takes a permutation of the batch, then splits it into chunks of size sgd_minibatch_size, and then traverses all the chunks, updating the policy when processing each chunk.
    # TODO how to set the neural network details (I see by default a 2-layered network with 256 neurons per layer is used, which could be a overkill for the first tests).
    # In the Ray PPO documentation they speak about a Learner worker, which is the worker that updates the policy based on the experiences gathered by the EnvRunner workers. Right now we have 1 learner worker and multiple env runners.
    algo_config = algo_config.resources(num_gpus=0) # we don't use GPU for now, TODO check if we can use GPU for policy updates
    algo_config = algo_config.env_runners(
        num_env_runners=cfg.ray.env_runners, rollout_fragment_length=int(cfg.sim.episode_length / cfg.sim.dt)
    ) # for simplicity first we use 1, later on should be more parallel using more workers for the rollouts.
    # Next, in the config.environment, we specify the environment options as well
    algo_config = algo_config.environment(
        env="RayleighBenardEnv",
        env_config={
           "sim": cfg.sim,
           "action_segments": cfg.action_segments,
           "action_limit": cfg.action_limit,
           "action_duration": cfg.action_duration,
           "output_dir": output_dir,
        }
    )    # here I set the environment for where this policy acts in
    algo_config = algo_config.framework("torch")  # we use PyTorch as the framework for the policy

    logger.info('Working directory for the local Ray program: ' + os.getcwd())
    # TODO FYI, using a Tuner from ray, one can optimize the hyperparameters of the policy
    # the optimization space should not have to be too large for PPO, as it is a robust algorithm

    # We build the policy that was configured above into an RLlib Algorithm
    # TODO we use use_copy=False right now, because it gives problems with deepcopying the RayleighBenardEnv due to the MPI usage.
    # But we may or may not need to use it for the parallel case. However, the documentation says it is only used for recycling the policy in test cases.
    rrlib_algo = algo_config.build(use_copy=False)

    # The algorithms work on the SampleBatch or MultiAgentBatch types. In our case we have a single agent, so we use SampleBatch.
    # They store trajectories of experiences from the environment. The policy is updated based on these experiences.

    # Only applies to parallel environments: We need an EnvRunnerGroup where the 
    # remote workers run the environment and collect experiences and the local
    # worker updates the policy based on these experiences in a SGD fashion with repeated updates, possibly on a GPU.

    # Next we train the policy on the environment:
    # The next line is a function that will take care of the whole training loop inside
    nr_iters = 1000 # is the number of batch updates that the Learner Worker will do
    for i in range(nr_iters):   # each iteration will acquire the nr of data (see config above) and update the policy using multiple mini batches and traversals through the data
        train_info = rrlib_algo.train()  # TODO check if we need to pass the number of iterations or episodes to train
        logger.info(f"Training iteration {i + 1}/{nr_iters} completed. Avg reward throughout rollout process: {train_info['env_runners']['episode_reward_mean']}")
        # Note that train_info gets saved to the ~/ray_results directory
        save_result = rrlib_algo.save(join(output_dir, 'ray_algocheckpoint'))
        logger.info(f"Saved the policy to {save_result}")



# def main(cfg: DictConfig) -> None:
#     return run_env(cfg=cfg)
# 
if __name__ == "__main__":
    run_env()

# Here we can evaluate the policy on the environment:
# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
# 
#     if termination or truncation:
#         action = None
#     else:
#         action = env.action_space(agent).sample()   # insert policy here
#     
#     env.step(action)    # apply action to the environment
# env.close()

# config = (
#     PPOConfig()
#     .environment("Taxi-v3")
#     .env_runners(num_env_runners=2)
#     .framework("torch")
#     .training(model={"fcnet_hiddens": [64, 64]})
#     .evaluation(evaluation_num_env_runners=1)
# )


# algo = config.build()

# for _ in range(5):
#     result = algo.train()
#     print(result)

# algo.evaluate()


# config = (
#     PPOConfig()
#     .with_updates(
#         {
#             "num_workers": 1,
#             "num_envs_per_worker": 1,
#             "rollout_fragment_length": 100,
#             "train_batch_size": 400,
#             "num_sgd_iter": 30,
#             "lr": 5e-5,
#             "gamma": 0.99,
#             "lambda": 0.95,
#             "clip_param": 0.2,
#             "vf_clip_param": 10.0,
#             "entropy_coeff": 0.0,
#             "model": {
#                 "fcnet_hiddens": [256, 256],
#                 "fcnet_activation": "tanh",
#             },
#         }
#     )
#     .with_updates(
#         {
#             "env": "RayleighBenardEnv",
#             "env_config": {
#                 "sim_cfg": {
#                     "episode_length": 1000,
#                     "dt": 0.01,
#                     "bcT": (0.0, 1.0),
#                     "N": (64, 64),
#                     "L": (1.0, 1.0),
#                     "Pr": 1.0,
#                     "Ra": 1.0e4,
#                     "nu": 1.0e-3,
#                 },
#                 "action_segments": 10,
#                 "action_limit": 0.75,
#                 "action_duration": 1.0,
#                 "action_start": 0.0,
#             },
#         }
#     )
# )