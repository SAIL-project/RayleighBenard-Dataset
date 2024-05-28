import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig
# from rbcdata.utils.rbc_field import RBCField

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.sim.rbc_env import RayleighBenardEnv


def run_env(cfg: DictConfig) -> None:
    # Set up gym environment
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        segments=cfg.segments,
        action_scaling=cfg.action_scaling,
        action_duration=cfg.action_duration,
        modshow=cfg.modshow,
        render_mode=cfg.render_mode,
    )
    _, _ = env.reset(seed=cfg.seed)


    # Run simulation
    while True:
        # if env.env_steps == 4:
        #     env.simulation.bcT[0] = 3
        # Simulation step
        action = np.linspace(-1, 1, 10)
        # action = -1 + np.random.rand(cfg.segments) * 2
        # action = np.array([1] * cfg.segments)
        # action = np.array([-cfg.action_scaling] * cfg.segments)
        _, _, terminated, truncated, _ = env.step(action)
        print(env.action_effective)
        state = env.get_state()
        print(state[-1][-1])
        # Termination criterion
        if terminated or truncated:
            break
    # Close
    env.close()

    


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    run_env(cfg=cfg)


if __name__ == "__main__":
    main()
