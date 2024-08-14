import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.envs.rbc_env import RayleighBenardEnv

# here I am trying to implement a model-free RL method that tries to reduce the Nusselt
# number by applying temperature fluctuations on the bottom wall.


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
    # Action loop in which each iteration applies an action (temperature profile at bottom)
    # for cfg.action_duration seconds to the environment.
    # TODO(s) Can we build a Q-table where we have for instance latent representations of the
    # state as "s" and as "a" the 1024 different actions that can be taken in that state. We
    # can also cluster together similar states based on distance.
    # TODO More appropriate distance measures to evaluate the similarity between states, for
    # example based on Fourier or the perceptual distance measure.
    # I guess you should write an outer loop that resets the states and performs the current
    # policy or tryout for the new episode.

    # number of episodes the learn over. TODO put hard-coded parameter in a configuration file
    nr_episodes = 100
    for i in range(nr_episodes):
        # perform an episode performing the actions according to some policy.
        while True:
            action = np.zeros(cfg.nr_segments)
            _, _, terminated, truncated, _ = env.step(action)
            print(
                f"Nusselt nr. full state: {env.simulation.compute_nusselt(from_obs=False):.4f}, \
                    from observations: {env.simulation.compute_nusselt(from_obs=True):.4f}"
            )
            # print(env.action_effective)
            env.__get_state()
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
