from stable_baselines3 import PPO
from os.path import join
import yaml
import logging
import numpy as np
from omegaconf import DictConfig  # datatype for the configuration
from rbcdata.utils.callbacks import RBCVisCallback
import argparse

from rbcdata.sim.rbc_env import RayleighBenardEnv   # the environment that the single agent will interact with

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Inspect a trained agent in the Rayleigh-Benard environment')
parser.add_argument('--experiment_dir', type=str, help='Directory of the experiment with the trained policy and the used configuration.')
parser.add_argument('--apply_policy', type=bool, default=True, help='Whether to apply the policy or just look at constant case.')
args = parser.parse_args()

# Hydra output directory that we will load a checkpoint from
experiment_dir = args.experiment_dir 

# read config
with open(join(experiment_dir, '.hydra/config.yaml')) as file:
    config = DictConfig(yaml.safe_load(file))
config['output_dir'] = './tmp/' # TODO remove this, if save_checkpoint=False RayleighBenardEnv should not expect an output dir (env doesn't write in that case)
config.sim.save_checkpoint = False
logger.info(f"Loaded config from {experiment_dir}.hydra/config.yaml")

test_env = RayleighBenardEnv(config, nusselt_logging=True)

# Here we just restore the policy from the checkpoint
# policy_SA = PPO.load(join(experiment_dir, 'model_checkpoint_train/PPOmodelRBC_16000_steps'), env=test_env)
policy_SA = PPO.load(join(experiment_dir, 'PPOmodelRBC_besteval_40000_steps'), env=test_env)

visCallback = RBCVisCallback(size=config.sim.N, bcT=config.sim.bcT, action_limit=config.action_limit, interval=1)

# instantiate the environment
obs, info = test_env.reset()

apply_policy = args.apply_policy
evalfortraindurations = 3   # evaluation for this many train episode durations.
# Evaluate the agent (policy) in the environment
for i in range(evalfortraindurations * int(config.sim.episode_length / config.action_duration)):
    if apply_policy:
        action = policy_SA.predict(obs, deterministic=True) # returns the action as well as the hidden state which we don't use here
        action = action[0]
        print(action)
    else:
        action = np.full(config.action_segments, config.sim.bcT[0])
    # logger.info(f"Step {i}: action={action}")
    obs, reward, closed, truncated, info = test_env.step(action)
    # logger.info(f"Step {i}, after applying action: reward={reward}, Nusselt={env.simulation.compute_nusselt(False)}, closed={closed}, truncated={truncated}, info={info}")
    visCallback(test_env, obs, reward, info)