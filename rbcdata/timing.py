import os
import time

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from rbcdata.env.rbc_env import RayleighBenardEnv


def run_env(cfg: DictConfig) -> None:
    print(os.cpu_count())

    env = RayleighBenardEnv(env_config=cfg.env, render_mode=cfg.render_mode)
    obs, info = env.reset()
    # time
    start = time.time()
    # Run environment
    for i in range(30):
        # Simulation step
        obs, reward, terminated, truncated, info = env.step()
        print(f"step={i}")
        if terminated or truncated:
            break
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    # Close
    env.close()


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    return run_env(cfg=cfg)


if __name__ == "__main__":
    main()
