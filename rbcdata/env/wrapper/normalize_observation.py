from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, WrapperObsType


class NormalizeObservation(gym.ObservationWrapper[WrapperObsType, ActType, ObsType]):
    """Normalize the observation to image range [0, 255]"""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
    ):
        gym.ObservationWrapper.__init__(self, env)

        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        # TODO hardcoded right now
        self.min_vals = [-1.3, -1.3, 1]
        self.max_vals = [1.3, 1.3, 2.75]

    def observation(self, obs: ObsType) -> Any:
        # Ensure the image is in float format
        obs = obs.astype(np.float32)

        # Normalize each channel
        for c in range(obs.shape[2]):
            obs[..., c] = (
                255 * (obs[..., c] - self.min_vals[c]) / (self.max_vals[c] - self.min_vals[c])
            )

        # Clip values to be in the range [0, 255]
        obs = np.clip(obs, 0, 255)

        return obs.astype(np.uint8)
