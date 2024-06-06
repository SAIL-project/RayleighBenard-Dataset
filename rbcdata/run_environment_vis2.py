import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from functools import partial

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.sim.rbc_env import RayleighBenardEnv
from rbcdata.utils.rbc_field import RBCField
from rbcdata.vis import RBCFieldVisualizer
from rbcdata.vis import RBCConvectionVisualizer

def update_vis(frame, window: RBCFieldVisualizer, sequence):
    # frame is just an index
    # update the AxesImage
    window.draw(sequence[frame, RBCField.T], sequence[frame, RBCField.UX], sequence[frame, RBCField.UY], frame * 1)


def update_vis_convec(frame, window: RBCConvectionVisualizer, sequence):
    # frame is just an index
    # update the AxesImage
    window.draw(sequence[frame], frame * 1)


def run_env(cfg: DictConfig) -> None:
    # Set up gym environment
    np.random.seed(0)
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        nr_segments=cfg.nr_segments,
        action_scaling=cfg.action_scaling,
        action_duration=cfg.action_duration,
        modshow=cfg.modshow,
        render_mode=cfg.render_mode,
    )
    _, _ = env.reset(seed=cfg.seed)

    # start = 0
    end = 50    
    
    observed_steps = np.zeros((end, 3, 64, 96))
    observed_steps_convection = np.zeros((end, 64, 96))
    minConv = 5000  # records the minimum convection value throughout the episode
    maxConv = -5000 # recors the maximum convection value throughout the episode

    i = 0
    # Run simulation
    while True:
        # Simulation step
        action = np.array([-cfg.action_scaling] * cfg.nr_segments)
        # action[0] = cfg.C
        _, _, terminated, truncated, _ = env.step(action)
        # get state and draw in window
        state = env.get_state()
        observed_steps[i] = state
        observed_steps_convection[i] = state[0] * state[-1]
        minConv_i = np.min(observed_steps_convection[i])
        maxConv_i = np.max(observed_steps_convection[i])
        if maxConv < maxConv_i:
            maxConv = maxConv_i
        if minConv > minConv_i:
            minConv = minConv_i
        
        # Termination criterion
        if terminated or truncated:
            break
        i += 1
        print(i)
        if i >= end:
            break
    # Close
    env.close()

    window = RBCFieldVisualizer(vmin=1, vmax=2)
    print(minConv)
    print(maxConv)
    window_convection = RBCConvectionVisualizer(vmin=minConv, vmax=maxConv)
    ani = animation.FuncAnimation(fig=window.fig, func=partial(update_vis, window=window, sequence=observed_steps), frames=len(observed_steps), interval=500)
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=2,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save('exampleRBC.gif', writer=writer)
    plt.show()

    ani2 = animation.FuncAnimation(fig=window_convection.fig, func=partial(update_vis_convec, window=window_convection, sequence=observed_steps_convection), frames=len(observed_steps_convection), interval=500)
    writer2 = animation.PillowWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    ani2.save('exampleRBCconvec.gif', writer=writer2)
    plt.show()


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    cfg['sim']['cook_length'] = 1
    cfg['action_duration'] = 1
    print(cfg)
    run_env(cfg=cfg)


if __name__ == "__main__":
    main()
