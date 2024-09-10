import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv
from supersuit import observation_lambda_v0


def ma_flatten(env):
    assert isinstance(env, AECEnv) or isinstance(
        env, ParallelEnv
    ), "pad_observations_v0 only accepts an AECEnv or ParallelEnv"
    assert hasattr(
        env, "possible_agents"
    ), "environment passed to pad_observations must have a possible_agents list."

    # observation lambda
    return observation_lambda_v0(
        env,
        lambda obs, obs_space: flatten_observation(obs),
        lambda obs_space: flatten_space(obs_space),
    )


def flatten_space(space):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.flatten_space(space)

    else:
        raise NotImplementedError(f"flatten_space not implemented for {type(space)}")


def flatten_observation(obs):
    return obs.flatten()
