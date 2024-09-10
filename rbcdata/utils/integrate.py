import gymnasium as gym

from rbcdata.control.controller import Controller


def integrate(
    env: gym.Env,
    callbacks: list[callable] = [],
    controller: Controller = None,
    seed: int | None = None,
    render: bool = False,
):
    # Set up gym environment
    if render:
        env.render_mode = "human"
    obs, info = env.reset(seed=seed)
    action = env.action
    # Run environment
    while True:
        # Controller
        if controller is not None:
            action = controller(env, obs, info)
        # Simulation step
        obs, reward, terminated, truncated, info = env.step(action)
        # Render
        if render:
            env.render()
        # Termination criterion
        if terminated or truncated:
            break
        # Callbacks
        for callback in callbacks:
            callback(env, obs, reward, info)

    # Close
    env.close()
    for callback in callbacks:
        callback.close()
