import math
import os
import pathlib

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

    # Create dataset on disk
    file_name = f"{path}/ra{cfg.environment.Ra}/rbc{seed}.h5"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = h5py.File(file_name, "w")

    # Create datasets for Temperature and velocity field
    dt_ratio = math.floor(cfg.dt / cfg.environment.dt)
    assert cfg.dt / cfg.environment.dt == dt_ratio, "dt must be a multiple of dt_sim"
    dataset = file.create_dataset(
        "data",
        (env.steps, 3, env.cfg.N[0], env.cfg.N[1]),
        chunks=True,
        dtype=np.float32,
    )

    # Save commonly used parameters of the simulation
    file.attrs["seed"] = seed
    file.attrs["steps"] = env.steps
    file.attrs["episode_length"] = env.cfg.episode_length
    file.attrs["cook_length"] = env.cfg.cook_length
    file.attrs["N"] = env.cfg.N
    file.attrs["ra"] = env.cfg.Ra
    file.attrs["pr"] = env.cfg.Pr
    file.attrs["dt"] = cfg.dt
    file.attrs["bcT"] = env.cfg.bcT
    file.attrs["domain"] = env.cfg.domain

    # Run simulation
    while True:
        step = info["step"] - env.cook_steps
        state = env.get_state()
        # Save observations for every dt and not dt_sim
        if step % (dt_ratio) == 0:
            idx = math.floor(step / dt_ratio)
            dataset[idx] = state
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
    path = pathlib.Path(f"{cfg.path}")
    for i in tqdm(range(cfg.count), position=0, leave=True):
        create_dataset(cfg=cfg, seed=cfg.base_seed + i, path=path)


if __name__ == "__main__":
    main()
