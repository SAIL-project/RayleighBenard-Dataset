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
    env = RayleighBenardEnv(cfg=cfg.sim, tqdm_position=2 * num + 1)

    # Create dataset on disk
    file_name = f"{path}/ra{cfg.sim.ra}/rbc{seed}.h5"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = h5py.File(file_name, "w")

    # Create datasets for Temperature and velocity field
    dt_ratio = math.floor(cfg.dt / cfg.sim.dt)
    assert cfg.dt / cfg.sim.dt == dt_ratio, "dt must be a multiple of dt_sim"
    dataset = file.create_dataset(
        "data",
        (env.steps, 3, env.cfg.N[0], env.cfg.N[1]),
        chunks=(10, 3, env.cfg.N[0], env.cfg.N[1]),
        compression="gzip",
        dtype=np.float32,
    )

    # Save commonly used parameters of the simulation
    file.attrs["seed"] = seed
    file.attrs["steps"] = env.steps
    file.attrs["episode_length"] = env.cfg.episode_length
    file.attrs["cook_length"] = env.cfg.cook_length
    file.attrs["N"] = env.cfg.N
    file.attrs["ra"] = env.cfg.ra
    file.attrs["pr"] = env.cfg.pr
    file.attrs["dt"] = cfg.dt
    file.attrs["bcT"] = env.cfg.bcT
    file.attrs["domain"] = env.cfg.domain

    # Run simulation
    _, info = env.reset(seed=seed)
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


if __name__ == "__main__":
    main()
