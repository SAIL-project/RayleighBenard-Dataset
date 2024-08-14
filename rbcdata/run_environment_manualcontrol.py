import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.envs.rbc_env import RayleighBenardEnv

# In this environment I am applying a manual control between the cells, in order to see
# if it will break the two-cell setup.


def run_env(cfg: DictConfig) -> None:
    # Set up gym environment
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        nr_segments=cfg.nr_segments,
        action_scaling=cfg.action_scaling,
        action_duration=cfg.action_duration,
        modshow=cfg.modshow,
        render_mode=cfg.render_mode,
    )
    # _, _ = env.reset(seed=cfg.seed, filename='../06-14-15-21-39/shenfun/ra10000/RB_2D')
    _, _ = env.reset(seed=cfg.seed, filename="../06-14-20-37-38/shenfun/ra10000/RB_2D")
    # _, _ = env.reset(seed=cfg.seed)

    fig, ax = plt.subplots()
    # Run simulation
    while True:

        # action = np.zeros(cfg.nr_segments)
        action = -1 * np.ones(cfg.nr_segments)
        action[3] = 1
        action[8] = 1
        _, _, terminated, truncated, _ = env.step(action)
        print(
            f"Nusselt nr. full state: {env.simulation.compute_nusselt(from_obs=False):.4f}, from \
            observations: {env.simulation.compute_nusselt(from_obs=True):.4f}"
        )
        # print(env.action_effective)
        state = env.__get_state()
        # Find out if last row temperate of state is approximately equal to Piecewise action
        ax.clear()
        ax.plot(state[-1][-1])
        ax.plot(state[-1][-2])
        fig.canvas.draw()
        # Termination criterion
        if terminated or truncated:
            break
    # Close
    env.close()


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)  # numpy seed
    run_env(cfg=cfg)


if __name__ == "__main__":
    main()
