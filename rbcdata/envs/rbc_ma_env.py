from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from rbcdata.envs.rbc_env import RayleighBenardEnv


class RayleighBenardMultiAgentEnv(MultiAgentEnv):
    def __init__(
        self,
        env: RayleighBenardEnv,
        beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.env = env
        self.beta = beta
        self._agent_ids = set([str(idx) for idx in range(env.action_segments)])

        # segment index mapping
        self.segment_mapping = np.array_split(np.arange(env.cfg.N_obs[1]), env.action_segments)

        # spaces
        self.observation_space = gym.spaces.Dict(
            spaces={
                index: gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3, env.cfg.N[0], env.cfg.N[1]),  # Segmentize the observation space
                    dtype=np.float32,
                )
                for index in self._agent_ids
            }
        )
        self.action_space = gym.spaces.Dict(
            spaces={
                index: gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                for index in self._agent_ids
            }
        )

        # assertions
        # same number of probes per segment
        self.probes_per_segment = env.cfg.N_obs[1] / self.env.action_segments
        assert (
            self.probes_per_segment.is_integer()
        ), "Number of probes must be divisible by segments"
        # observation can be centered
        center_index = (self.env.action_segments - 1) / 2
        assert (center_index * self.probes_per_segment).is_integer(), "Centering is not possible"
        # TODO

    def reset(
        self, seed: Optional[int], options: Optional[dict] = None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        return self.__obs_dict(obs), self.__2dict(info)

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # Apply action
        action = self.__dict2array(action_dict)
        obs, _, terminated, truncated, info = self.env.step(action)

        return (
            self.__obs_dict(obs),
            self.__reward_dict(obs),
            terminated,
            truncated,
            self.__2dict(info),
        )

    def __reward_dict(self, obs: Any):
        return {index: self.__reward(obs, int(index)) for index in self._agent_ids}

    def __obs_dict(self, obs) -> MultiAgentDict:
        return {index: self.__recenter_observation(obs, int(index)) for index in self._agent_ids}

    def __2dict(self, obj) -> MultiAgentDict:
        return {index: obj for index in self._agent_ids}

    def __dict2array(self, obj) -> list:
        return np.array([obj[index] for index in self._agent_ids])

    def __reward(self, obs, index):
        # global nusselt number
        nu_global = self.env.simulation.compute_nusselt(obs)
        # local nusselt number
        local_obs = obs[:, :, self.segment_mapping[index]]
        nu_local = self.env.simulation.compute_nusselt(local_obs)

        return nu_global - self.beta * nu_local

    def __recenter_observation(self, obs, index):
        center_idx = (self.env.action_segments - 1) / 2
        shift = int((center_idx - index) * self.probes_per_segment)
        return np.roll(obs, shift, axis=2)
