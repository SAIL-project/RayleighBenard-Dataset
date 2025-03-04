import gymnasium as gym

from rbcdata.control.controller import Controller


def integrate(
    env: gym.Env,
    callbacks: list[callable] = [],
    controller: Controller = None,
    seed: int | None = None,
    episode_idx: int = 0,
):
    # Set up gym environment
    obs, info = env.reset(seed=seed)
    action = None
    # Run environment
    while True:
        # Controller
        if controller is not None:
            action = controller(env, obs, info)
        # Simulation step
        obs, reward, terminated, truncated, info = env.step(action)
        # Render
        env.render()
        # Termination criterion
        if terminated or truncated:
            break
        # Callbacks
        for callback in callbacks:
            callback(env, obs, reward, info, episode_idx=episode_idx)

    # close environment and callbacks
    for callback in callbacks:
        callback.reset()
