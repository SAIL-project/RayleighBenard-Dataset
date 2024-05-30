import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig
import matplotlib.pyplot as plt
# from rbcdata.utils.rbc_field import RBCField

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.sim.rbc_env import RayleighBenardEnv


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
    _, _ = env.reset(seed=cfg.seed)


    fig, ax = plt.subplots()
    # Run simulation
    while True:
        if env.env_step == 2:
            env.simulation.bcT_avg = (3, 1)
        if env.env_step > 20:
            action = np.zeros(cfg.nr_segments)
            action[:cfg.nr_segments // 2] = -1 
            action[cfg.nr_segments // 2:] = 1
        else: 
            action = -1 + np.random.rand(cfg.nr_segments) * 2
        # Simulation step
        # action = np.linspace(-1, 1, cfg.nr_segments)
        # action = np.array([1] * cfg.segments)
        # action = np.array([-cfg.action_scaling] * cfg.segments)
        _, _, terminated, truncated, _ = env.step(action)
        # print(env.action_effective)
        state = env.get_state()
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
    run_env(cfg=cfg)


if __name__ == "__main__":
    main()