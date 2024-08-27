import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.policy.policy import Policy
import yaml
import logging
import numpy as np
from omegaconf import DictConfig  # datatype for the configuration
from rbcdata.utils.callbacks import RBCVisCallback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rbcdata.sim.rbc_env import RayleighBenardEnv   # the environment that the single agent will interact with

# TODO what is the evaluate() function used for in the Algorithm class?

# Hydra output directory that we will load a checkpoint from
experiment_dir = 'logs/runs/08-27-12-39-11/'

# ray.init()
# Here we just restore the policy from the Algorithm checkpoint
policy_SA = Policy.from_checkpoint(experiment_dir + 'ray_algocheckpoint/policies/default_policy')
# read config
with open(experiment_dir + '.hydra/config.yaml') as file:
    config = DictConfig(yaml.safe_load(file))
config['output_dir'] = './tmp/'
logger.info(f"Loaded config from {experiment_dir}.hydra/config.yaml")

visCallback = RBCVisCallback(size=config.sim.N, bcT=config.sim.bcT, action_limit=config.action_limit, interval=1)

# instantiate the environment
env = RayleighBenardEnv(config)
obs, info = env.reset()

apply_policy = True

# Evaluate the agent (policy) in the environment
for i in range(3 * int(config.sim.episode_length / config.action_duration)):
    if apply_policy:
        action = policy_SA.compute_single_action(obs)
        action = action[0]
    else:
        action = np.full(config.action_segments, config.sim.bcT[0])
    logger.info(f"Step {i}: action={action}")
    obs, reward, closed, truncated, info = env.step(action)
    logger.info(f"Step {i}, after applying action: reward={reward}, Nusselt={env.simulation.compute_nusselt(False)}, closed={closed}, truncated={truncated}, info={info}")
    visCallback(env, obs, reward, info)