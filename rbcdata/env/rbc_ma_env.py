import functools
from copy import copy
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from rbcdata.env.rbc_env import RayleighBenardEnv


class RayleighBenardMultiAgentEnv(ParallelEnv):
    metadata = {
        "name": "rbc_ma_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    BETA = 0.2
    FLATTEN = True

    def __init__(
        self,
        env_config: Optional[Dict] = {},
        render_mode: Optional[str] = None,
    ) -> None:
        # environment configuration
        self.env = RayleighBenardEnv(env_config, render_mode)
        self.beta = env_config.get("beta", self.BETA)
        self.flatten = env_config.get("flatten", self.FLATTEN)
        self.timestep = None
        self.render_mode = None

        # agent configuration
        self.possible_agents = [str(idx) for idx in range(self.env.action_segments)]
        self.segment_mapping = np.array_split(
            np.arange(self.env.size_obs[1]), self.env.action_segments
        )

        # spaces
        self.observation_spaces = dict(
            zip(self.possible_agents, [self.env.observation_space for _ in self.possible_agents])
        )
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.action_spaces = dict(
            zip(self.possible_agents, [action_space for _ in self.possible_agents])
        )

        # assertions
        # same number of probes per segment
        self.probes_per_segment = self.env.size_obs[1] / self.env.action_segments
        assert (
            self.probes_per_segment.is_integer()
        ), "Number of probes must be divisible by segments"
        # observation can be centered
        center_index = (self.env.action_segments - 1) / 2
        assert (center_index * self.probes_per_segment).is_integer(), "Centering is not possible"

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        # Reset environment
        obs, info = self.env.reset(seed=seed, options=options)
        return self.__obs_dict(obs), self.__2dict(info)

    def step(self, action_dict: Dict):
        # Apply action
        action = self.__dict2array(action_dict)
        obs, _, terminated, truncated, info = self.env.step(action)
        self.timestep += 1

        return (
            self.__obs_dict(obs),
            self.__reward_dict(obs),
            self.__2dict(terminated, include_all=True),
            self.__2dict(truncated, include_all=True),
            self.__2dict(info),
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self) -> np.ndarray:
        return self.env.simulation.get_state()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def __reward_dict(self, obs: Any) -> Dict:
        return {index: self.__reward(obs, int(index)) for index in self.agents}

    def __obs_dict(self, obs) -> Dict:
        return {index: self.__recenter_observation(obs, int(index)) for index in self.agents}

    def __2dict(self, obj, include_all=False) -> Dict:
        dict = {index: obj for index in self.agents}
        if include_all:
            dict["__all__"] = obj
        return dict

    def __dict2array(self, obj) -> list:
        return np.array([obj[index] for index in self.agents]).squeeze()

    def __reward(self, obs, index):
        # max nusselt number
        nu_max = 3
        # global nusselt number
        nu_g = self.env.simulation.compute_nusselt(obs)
        # local nusselt number
        local_obs = obs[:, :, self.segment_mapping[index]]
        nu_l = self.env.simulation.compute_nusselt(local_obs)

        # reward calculation
        return nu_max - (1 - self.beta) * nu_g - self.beta * nu_l

    def __recenter_observation(self, obs, index):
        center_idx = (self.env.action_segments - 1) / 2
        shift = int((center_idx - index) * self.probes_per_segment)
        recentered = np.roll(obs, shift, axis=2)
        return recentered.astype(np.float32)
