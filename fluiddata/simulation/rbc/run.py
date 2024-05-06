import time
from omegaconf import DictConfig

from fluiddata.simulation.rbc.rbc_env import RayleighBenardEnv


def run_rbc(sim: DictConfig):
    # Define system
    env = RayleighBenardEnv(
        sim_cfg=sim,
        render_mode="live",
    )

    # Run simulation
    while True:
        # Simulation step
        _, _, terminated, truncated, _ = env.step(env.action)
        # Termination criterion
        if terminated or truncated:
            break

        time.sleep(0.1)

    # Close
    env.close()
