from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import EnvConfigDict, MultiAgentDict

from rbcdata.envs.rbc_env import RayleighBenardEnv


class RayleighBenardMultiAgentEnv(MultiAgentEnv):

    BETA = 0.01
    FLATTEN = True

    def __init__(
        self,
        env_config: Optional[EnvConfigDict] = None,
    ) -> None:
        super().__init__()
        if env_config is None:
            env_config = {}
        self.env = RayleighBenardEnv(env_config)

        # environment configuration
        self.beta = env_config.get("beta", self.BETA)
        self.flatten = env_config.get("flatten", self.FLATTEN)

        # agent configuration
        self._agent_ids = set([str(idx) for idx in range(self.env.action_segments)])
        self.segment_mapping = np.array_split(
            np.arange(self.env.size_obs[1]), self.env.action_segments
        )

        # spaces
        if self.flatten:
            obs_shape = (3 * self.env.size_obs[0] * self.env.size_obs[1],)
        else:
            obs_shape = (3, self.env.size_obs[0], self.env.size_obs[1])

        self.observation_space = gym.spaces.Dict(
            spaces={
                index: gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs_shape,
                    dtype=np.float32,
                )
                for index in self._agent_ids
            }
        )
        self._obs_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            spaces={
                index: gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                )
                for index in self._agent_ids
            }
        )
        self._action_space_in_preferred_format = True

        # assertions
        # same number of probes per segment
        self.probes_per_segment = self.env.size_obs[1] / self.env.action_segments
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
            self.__2dict(terminated, include_all=True),
            self.__2dict(truncated, include_all=True),
            self.__2dict(info),
        )

    def get_global_nusselt(self):
        return self.env.get_nusselt()

    def __reward_dict(self, obs: Any):
        return {index: self.__reward(obs, int(index)) for index in self._agent_ids}

    def __obs_dict(self, obs) -> MultiAgentDict:
        return {index: self.__recenter_observation(obs, int(index)) for index in self._agent_ids}

    def __2dict(self, obj, include_all=False) -> MultiAgentDict:
        dict = {index: obj for index in self._agent_ids}
        if include_all:
            dict["__all__"] = obj
        return dict

    def __dict2array(self, obj) -> list:
        return np.array([obj[index] for index in self._agent_ids]).squeeze()

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
        recentered = np.roll(obs, shift, axis=2)

        if self.flatten:
            return recentered.flatten().astype(np.float32)
        return recentered.astype(np.float32)
