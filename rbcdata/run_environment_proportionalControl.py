import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.envs.rbc_env import RayleighBenardEnv

# In this environment I am applying a proportional control between the cells, in order to see
# if it will break the two-cell setup.
# Strategy, measure the temperature in the middle row of the simulation, apply proportional
# control like in Tang and Bau using these values.


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

    N = cfg.sim.N
    Tb = cfg.sim.bcT[0]
    Tu = cfg.sim.bcT[1]
    Tr = (Tb + Tu) / 2  # the reference temperature for the controller
    nr_heaters = cfg["nr_segments"]
    heater_skip = N[1] // nr_heaters

    fig, ax = plt.subplots()
    # Run simulation
    while True:
        # ideas, integral term, max min.
        state = env.__get_state()
        sensorT_midrow = state[
            -1, N[0] // 2 - 1, :
        ]  # temperature sensor measurement of the middle row of the layer
        error = sensorT_midrow - Tr
        ind_max = np.argmax(np.abs(error))
        extreme_error = error[ind_max]
        action = -error / np.abs(extreme_error)
        action_heaters = np.zeros(nr_heaters)
        for i in range(nr_heaters):
            action_heaters[i] = np.max(action[i * heater_skip : i * heater_skip + heater_skip])
        # group the values per heater
        ax.clear()
        ax.plot(action_heaters)

        # At this point: Internal model of the environment used for planning the next action vs.
        # no internal model of the environment used for planning the next action
        # (Model-based vs. model-free)
        _, _, terminated, truncated, _ = env.step(
            action_heaters
        )  # perform the action in the environment
        print(
            f"Nusselt nr. full state: {env.simulation.compute_nusselt(from_obs=False):.4f}, \
                from observations: {env.simulation.compute_nusselt(from_obs=True):.4f}"
        )
        # print(env.action_effective)
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
