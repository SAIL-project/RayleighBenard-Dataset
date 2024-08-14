# File for testing single agent reinforcement learning with Raylib
# General python imports
import rootutils
import logging
import hydra    # for loading the configuration
from omegaconf import DictConfig  # datatype for the configuration

# All Raylib imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env  # allows to register custom environments, every ray worker will have access to it

import gymnasium as gym

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
# Environment imports
from rbcdata.sim.rbc_env import RayleighBenardEnv   # the environment that the single agent will interact with


def run_env(cfg: DictConfig) -> None:
    # Structure of the code (based on examples of RLlib)
    ray.init()  # TODO see whether this call is necessary. It initializes the Ray runtime supposedly.

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
    config = PPOConfig()
    config = config.training(gamma=0.9, lr=1e-2) # discount factor, learning rate, TODO check if we need to set more parameters like train_batch_size, kl_coeff, etc.
    config = config.resources(num_gpus=0) # we don't use GPU for now, TODO check if we can use GPU for policy updates
    config = config.env_runners(num_env_runners=0) # for simplicity first we use 1, later on should be more parallel using more workers for the rollouts.
    # Next, in the config.environment, we specify the environment options as well
    config = config.environment(
        env="RayleighBenardEnv",
        env_config={
           "sim_cfg": cfg.sim,
           "action_segments": cfg.action_segments,
           "action_limit": cfg.action_limit
        }
    )    # here I set the environment for where this policy acts in
    config = config.framework("torch")  # we use PyTorch as the framework for the policy
    # config = config.training()   # for now we don't set any training parameters, we use the default ones


    # TODO FYI, using a Tuner from ray, one can optimize the hyperparameters of the policy
    # the optimization space should not have to be too large for PPO, as it is a robust algorithm

    # We build the policy that was configured above into an RLlib Algorithm
    # TODO we use use_copy=False right now, because it gives problems with deepcopying the RayleighBenardEnv due to the MPI usage.
    # But we may or may not need to use it for the parallel case. However, the documentation says it is only used for recycling the policy in test cases.
    rrlib_algo = config.build(use_copy=False)

    # The algorithms work on the SampleBatch or MultiAgentBatch types. In our case we have a single agent, so we use SampleBatch.
    # They store trajectories of experiences from the environment. The policy is updated based on these experiences.

    # Only applies to parallel environments: We need an EnvRunnerGroup where the 
    # remote workers run the environment and collect experiences and the local
    # worker updates the policy based on these experiences in a SGD fashion with repeated updates, possibly on a GPU.

    # Next we train the policy on the environment:
    rrlib_algo.train()  # TODO check if we need to pass the number of iterations or episodes to train


@hydra.main(version_base=None, config_path="config", config_name="run_SA")
def main(cfg: DictConfig) -> None:
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Run the environment and reinforcement learning algorithm using specified configuration
    return run_env(cfg=cfg)

if __name__ == "__main__":
    main()

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