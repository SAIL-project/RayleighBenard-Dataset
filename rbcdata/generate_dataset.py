import math
import os
import pathlib
import time

import h5py
import hydra
import numpy as np
import rootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.sim.rbc_env import RayleighBenardEnv


def create_dataset(cfg: DictConfig, seed: int, path: pathlib.Path, num: int) -> None:
    # Set up gym environment
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        segments=cfg.segments,
        action_scaling=cfg.action_scaling,
        action_duration=cfg.action_duration,
        tqdm_position=2 * num + 1,
    )

    # Create dataset on disk
    file_name = f"{path}/ra{cfg.sim.ra}/rbc{seed}.h5"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = h5py.File(file_name, "w")

    # Create datasets for Temperature and velocity field
    dataset = file.create_dataset(
        "data",
        (env.env_steps, 3, env.cfg.N[0], env.cfg.N[1]),
        chunks=(10, 3, env.cfg.N[0], env.cfg.N[1]),
        compression="gzip",
        dtype=np.float32,
    )
    actions = file.create_dataset(
        "action",
        (env.env_steps, cfg.segments),
        chunks=(10, cfg.segments),
        compression="gzip",
        dtype=np.float32,
    )

    # Save commonly used parameters of the simulation
    file.attrs["seed"] = seed
    file.attrs["steps"] = env.env_steps
    file.attrs["episode_length"] = env.cfg.episode_length
    file.attrs["cook_length"] = env.cfg.cook_length
    file.attrs["N"] = env.cfg.N
    file.attrs["ra"] = env.cfg.ra
    file.attrs["pr"] = env.cfg.pr
    file.attrs["bcT"] = env.cfg.bcT
    file.attrs["domain"] = np.array(env.simulation.domain).astype(np.float32)
    file.attrs["segments"] = cfg.segments
    file.attrs["action_scaling"] = cfg.action_scaling
    file.attrs["action_duration"] = cfg.action_duration

    # Reset simulation
    _, info = env.reset(seed=seed)
    # Save initial state and action
    action = env.get_action()
    actions[0] = action
    dataset[0] = env.get_state()
    # Run simulation
    while True:
        # introduce random actions after action_start time
        if math.floor(info["step"] / cfg.action_duration) >= cfg.action_start:
            action = env.action_space.sample()

        # Simulation step
        _, _, terminated, truncated, info = env.step(action)
        # Termination criterion
        if terminated or truncated:
            break

        # Save state and action
        idx = info["step"]
        dataset[idx] = env.get_state()
        actions[idx] = env.get_action()

    # Close
    env.close()


@hydra.main(version_base=None, config_path="config", config_name="generate")
def main(cfg: DictConfig) -> None:
    # Generate Dataset
    num = HydraConfig.get().job.num
    path = pathlib.Path(f"{cfg.path}")
    time.sleep(num / 10)
    pbar = tqdm(total=cfg.count, desc=f"Generating Dataset {num}", position=2 * num, leave=False)
    for i in range(cfg.count):
        create_dataset(cfg=cfg, seed=cfg.base_seed + i, path=path, num=num)
        pbar.update(1)


@hydra.main(version_base=None, config_path="config", config_name="generate")
def main_debug(cfg: DictConfig) -> None:
    create_dataset(cfg=cfg, seed=cfg.base_seed, path=pathlib.Path(f"{cfg.path}"), num=0)


if __name__ == "__main__":
    main()
