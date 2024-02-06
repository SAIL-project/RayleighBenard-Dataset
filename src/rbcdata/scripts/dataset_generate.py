import math
import os
import pathlib
import time

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from rbcdata.sim import RayleighBenardEnv


def create_dataset(cfg: DictConfig, seed: int, path: pathlib.Path) -> None:
    # Set up gym environment
    env = RayleighBenardEnv(cfg=cfg.environment)
    _, info = env.reset(seed=seed)
    # Generate cartesian coords
    x_min, x_max, y_min, y_max = env.domain
    N1, N2 = env.N
    axis_x = np.linspace(x_min, x_max, num=N2)
    axis_y = np.linspace(y_min, y_max, num=N1)
    axis_t = np.arange(0, cfg.env.episode_length, step=env.dt)
    spatial_mesh = np.array(env.simulation.X)
    # Create dataset on disk
    file_name = f"{path}/{cfg.gen.name}/ra{cfg.env.ra}/rbc{seed}.h5"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = h5py.File(file_name, "w")
    # Save axes and cartesian coords
    file.create_dataset("axis_x", axis_x.shape, data=axis_x)
    file.create_dataset("axis_y", axis_y.shape, data=axis_y)
    file.create_dataset("axis_t", axis_t.shape, data=axis_t)
    file.create_dataset("spatial_mesh", spatial_mesh.shape, data=spatial_mesh)
    # Create datasets for Temperature and velocity field
    dt_ratio = math.floor(cfg.gen.dt / cfg.env.dt)
    assert cfg.gen.dt / cfg.env.dt == dt_ratio, "dt must be a multiple of dt_sim"
    frames = math.floor(cfg.env.episode_length / cfg.gen.dt)
    dataset = file.create_dataset(
        "data", (frames, 3, N1, N2), chunks=True, dtype=np.float32
    )
    dataset_stats = file.create_dataset(
        "stats", (frames, 2), chunks=True, dtype=np.float32
    )
    # Save commonly used parameters of the simulation
    file.attrs["length"] = cfg.env.episode_length
    file.attrs["samples"] = frames
    file.attrs["domain"] = env.domain
    file.attrs["N"] = env.N
    file.attrs["dt"] = cfg.gen.dt
    file.attrs["seed"] = seed
    # Run simulation
    while True:
        step = info["step"] - env.cook_steps
        state = env.get_state()
        stats = env.get_statistics()
        # Save observations for every dt and not dt_sim
        if step % (dt_ratio) == 0:
            idx = math.floor(step / dt_ratio)
            dataset[idx] = state
            dataset_stats[idx] = stats
        # Simulation step
        action = env.action_space.sample()  # TODO should be 0 action
        _, _, terminated, truncated, info = env.step(action)
        # Termination criterion
        if terminated or truncated:
            break
    # Close
    env.close()


@hydra.main(version_base=None, config_path="../config", config_name="generate")
def main(cfg: DictConfig) -> None:
    # Generate Dataset
    unique = math.floor(time.time())
    path = pathlib.Path(f"{cfg.path}/{unique}")
    for i in tqdm(range(cfg.count), position=0, leave=True):
        create_dataset(cfg=cfg, seed=cfg.base_seed + i, path=path)


if __name__ == "__main__":
    main()
