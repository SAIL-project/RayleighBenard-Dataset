# General python imports
import rootutils

# All Raylib imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray.tune.registry import register_env  # allows to register custom environments, every ray worker will have access to it

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
# Environment imports
# from pettingzoo.sisl import waterworld_v4
from rbcdata.sim.rbc_env_MP import RayleighBenardEnvMP

# Structure of the code (based on examples of RLlib)
ray.init()  # TODO see whether this call is necessary. It initializes the Ray runtime supposedly.

# We need to an environment (RayleighBenardEnvirontment)
# env = waterworld_v4.env(render_mode='human')
# An RLlib environment consists of: action space (all possible actions), state space
# an observation by the agent of certain parts of the state (all possible states), and a reward function (feedback agent receives per action)
env = RayleighBenardEnvMP()  # TODO

# TODO check if we need to register the environment, seems to be necessary for parallel environments
# register_env("RayleighBenardEnv", lambda config: RayleighBenardEnvMP())

# reset the environment to the initial state
env.reset()

# We need to define a policy for the agent, we use PPO.
# TODO I read PPO is robust, so check recommended parameters for our setting
config = (PPOConfig().training(gamma=0.9, lr=1e-2)
    .environment(env) # here I set the environment for where this policy acts in
    .resources(num_gpus=0)  # TODO check if we can use GPU for policy updates
    .env_runners=(num_env_runners=1) # TODO for simplicity first we use 1, later on should be more parallel
    .callbacks(MemoryTrackingCallbacks)  # TODO check if we need this
)
# TODO FYI, using a Tuner from ray, one can optimize the hyperparameters of the policy
# the optimization space should not be too large for PPO, as it is a robust algorithm

# TODO For our case, we need a MultiAgentPolicy, as we have multiple agents in the environment

# We build the policy that was configured above into an RLlib Algorithm
rrlib_algo = config.build()

# The algorithms work on the SampleBatch or MultiAgentBatch types,
# which store trajectories of experiences from the environment

# Only applies to parallel environments: We need an EnvRunnerGroup where the 
# remote workers run the environment and collect experiences and the local
# worker updates the policy based on the experiences.

# Next we train the policy on the environment:
# TODO

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