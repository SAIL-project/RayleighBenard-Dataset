import hydra
from omegaconf import DictConfig

from rbcdata.sim import RayleighBenardEnv


def run_env(cfg: DictConfig, seed: int) -> None:
    # Set up gym environment
    env = RayleighBenardEnv(cfg=cfg.environment, modshow=40, render_mode="live")
    _, _ = env.reset(seed=seed)

    # Run simulation
    while True:
        # Simulation step
        action = env.action_space.sample()  # TODO should be 0 action
        _, _, terminated, truncated, _ = env.step(action)
        # Termination criterion
        if terminated or truncated:
            break
    # Close
    env.close()


@hydra.main(version_base=None, config_path="../config", config_name="run")
def main(cfg: DictConfig) -> None:
    run_env(cfg=cfg, seed=cfg.seed)


if __name__ == "__main__":
    main()
