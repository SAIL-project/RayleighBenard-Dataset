from stable_baselines3 import PPO
from os.path import join, isfile
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import partial
from omegaconf import DictConfig  # datatype for the configuration
from rbcdata.utils.callbacks import RBCVisCallback
import argparse
from os import listdir
import re
import rbcdata.vis.rbc_field_visualizer as rbc_field_visualizer
from rbcdata.utils.rbc_field import RBCField
import matplotlib.animation as animation
from rbcdata.sim.rbc_env import RayleighBenardEnv   # the environment that the single agent will interact with
from rbcdata.vis.utils import animate_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Inspect a trained agent in the Rayleigh-Benard environment')
parser.add_argument('--experiment_dir', type=str, help='Directory of the experiment with the trained policy and the used configuration.')
parser.add_argument('--apply_policy', type=bool, default=True, help='Whether to apply the policy or just look at constant case.')
parser.add_argument('--train', type=bool, default=False, help='Whether to apply the latest training checkpoint')
parser.add_argument('--file', type=str, default=None, help='File name of the checkpoint to apply, if none is give the last checkpoint is used in train or validation folder is used')
parser.add_argument('--val', type=bool, default=False, help='Whether to apply the latest (best) validation checkpoint')
parser.add_argument('--save_ani', type=bool, default=False, help='Whether to save an animation of the result')

args = parser.parse_args()

if args.train and args.val or (not args.train and not args.val):
    raise ValueError("One of --train and --val should be set to True.")

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
# search for the last checkpoint in the folder and load that one
dir = 'model_checkpoint_train' if args.train else 'model_besteval'
if args.file is None:
    pattern = r'\d+'
    checkpoint_files = [f for f in listdir(join(experiment_dir, dir)) if isfile(join(experiment_dir, dir, f))]
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(re.search(pattern, x).group()))
else:
    checkpoint_files = [args.file]
policy_SA = PPO.load(join(experiment_dir, dir, checkpoint_files[-1]), env=test_env)
logger.info(f"Loading checkpoint {checkpoint_files[-1]}")

visCallback = RBCVisCallback(size=config.sim.N, bcT=config.sim.bcT, action_limit=config.action_limit, interval=1)

evalfortraindurations = 0.25   # evaluation for this many train episode durations.
# Nr steps to predict
nr_steps = evalfortraindurations * int(config.sim.episode_length / config.action_duration)
nr_steps = int(nr_steps)
# save observations for later animation
frames = np.zeros((nr_steps, 3, 64, 96))
nusselts = np.zeros(nr_steps)

# instantiate the environment
obs, info = test_env.reset()

# def func(frame, *fargs) -> iterable_of_artists
def update_vis(frame, window: rbc_field_visualizer.RBCFieldVisualizer, sequence, nusselts):
    # frame is just an index
    # update the AxesImage
    window.draw(sequence[frame, RBCField.T], sequence[frame, RBCField.UX], sequence[frame, RBCField.UY], 0)
    window.ax.set_title("", loc='left')
    window.ax.set_title(f'Nusselt {nusselts[frame]:.2f}', loc='left')

apply_policy = args.apply_policy
# Evaluate the agent (policy) in the environment
for i in range(nr_steps):
    if apply_policy:
        action = policy_SA.predict(obs, deterministic=True) # returns the action as well as the hidden state which we don't use here
        action = action[0]
        print(action)
    else:
        action = np.full(config.action_segments, config.sim.bcT[0])
    # logger.info(f"Step {i}: action={action}")
    obs, reward, closed, truncated, info = test_env.step(action)
    frames[i] = test_env.get_state()
    nusselts[i] = np.mean(test_env.nusselt_window)
    # logger.info(f"Step {i}, after applying action: reward={reward}, Nusselt={env.simulation.compute_nusselt(False)}, closed={closed}, truncated={truncated}, info={info}")
    # visCallback(test_env, obs, reward, info)

window = rbc_field_visualizer.RBCFieldVisualizer(vmin=1, vmax=2)
ani = animation.FuncAnimation(
    fig=window.fig,
    func=partial(update_vis, window=window, sequence=frames, nusselts=nusselts),
    frames=nr_steps,
    interval=500
)
# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=2,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('ppo_result.gif', writer=writer)
plt.show()