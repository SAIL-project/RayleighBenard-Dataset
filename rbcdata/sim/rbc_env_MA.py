import copy
from typing import Any, Dict, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sympy

from rbcdata.config import RBCSimConfig
from rbcdata.sim.rayleighbenard2d import RayleighBenard
from rbcdata.sim.tfunc import Tfunc

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]

from ray.rllib.env.multi_agent_env import MultiAgentEnv

x, y, tt = sympy.symbols("x,y,t", real=True)


class RayleighBenardEnv(MultiAgentEnv):
    # gym.Env[RBCAction, RBCObservation] # TODO previously extended the gym env, look for action and observation space definitions.
    """
    Multi-Agent env in which each agent controls a different segment of the bottom boundary.
    All agents step simultaneously in this environment.
    Each agent has the same action space, but different local observation spaces.
    """
    reward_range = (-float("inf"), float("inf"))
    def __init__(
        self,
        sim_cfg: RBCSimConfig,
        action_segments: int = 10,
        action_limit: float = 0.75,
        action_duration: float = 1.0,
        action_start: float = 0.0,
        fraction_length_smoothing=0.1,
    ) -> None:
        """
        Initialize the environment with the given configuration. 
        Args:
            sim_cfg: The configuration for the PDE simulation.
            action_segments: the number of heaters on the bottom boundary, i.e. the number of agents.
            action_limit: The maximum fluctuation of each agent's output on top of the average temperature.
            action_duration: The duration of the action in relative simulation time.
            action_start: The starting absolute simulation time of the action. 
            fraction_length_smoothing: The fraction of the length of the domain to smooth the action.
        """
        super().__init__()

        # Env configuration
        self.cfg = sim_cfg
        self.episode_length = sim_cfg.episode_length
        self.episode_steps = int(sim_cfg.episode_length / sim_cfg.dt)
        self.closed = False

        # Action configuration
        self.action_limit = action_limit
        self.action_duration = action_duration
        self.action_segments = action_segments
        self.action_start = action_start

        # TODO: See if we need a mapping here from agent ID to its action space
        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_segments,), dtype=np.float32)

        # TODO see if we need a mapping here from agent ID to its observation space
        # TODO How important is the observation space for the agents?
        # Because the agents only use the local reward signal to learn, which is computed from the local observation.
        # Maybe only the reward function uses the observation space?
        # Observation Space
        self.observation_space = gym.spaces.Box(
            sim_cfg.bcT[1],
            sim_cfg.bcT[0] + action_limit,
            shape=(
                1,
                sim_cfg.N[0] * sim_cfg.N[1] * 3,
            ),
            dtype=np.float32,
        )

        # PDE configuration
        self.simulation = RayleighBenard(
            N_state=(sim_cfg.N[0], sim_cfg.N[1]),
            Ra=sim_cfg.ra,
            Pr=sim_cfg.pr,
            dt=sim_cfg.dt,
            bcT=(sim_cfg.bcT[0], sim_cfg.bcT[1]),
            filename=sim_cfg.checkpoint_path,
        )
        # The Tfunc class is used to compute the effective action on the domain, respecting the action limit.
        # and smoothing the action over a fraction of the domain length.
        self.t_func = Tfunc(
            segments=action_segments,
            domain=self.simulation.domain,
            action_limit=action_limit,
            bcT_avg=self.simulation.bcT_avg,
            x=y,
            fraction_length_smoothing=fraction_length_smoothing,
        )

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None, filename=None
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        # init PDE simulation
        self.t, self.tstep = self.simulation.initialize(
            filename=filename, np_random=self._np_random, rand=0.000001
        )
        self.simulation.assemble()
        self.simulation.step(self.t, self.tstep)

        # Reset action
        self.action = np.array([0.0])
        self.action_effective = None  # TODO sympy zero

        return self.get_obs(), self.__get_info()

    def step(self, action: RBCAction) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        """
        Function to perform one step of the environment using action "action", i.e.
        (state(t), action(t)) -> state(t+1)
        """
        truncated = False
        # Apply action
        self.action = action
        self.action_effective = self.t_func.apply_T(copy.deepcopy(action))
        self.simulation.update_actuation((self.action_effective, self.simulation.bcT[1]))

        self.t, self.tstep = self.simulation.step(tstep=self.tstep, t=self.t)

        # Check for truncation
        if self.t >= self.episode_length:
            truncated = True

        # TODO note that the second return value is the reward, to be implemented still.
        # The reward should also be returned as a dict, with the agent ID as key.
        return self.get_obs(), 0, self.closed, truncated, self.__get_info()

    def close(self) -> None:
        self.closed = True

    def get_obs(self) -> RBCObservation:
        """
        Returns a dictionary that maps agent IDs to their local observations.
        """
        return self.simulation.obs_flat.astype(np.float32)

    def get_state(self) -> RBCObservation:
        return self.simulation.state.astype(np.float32)

    def get_action(self) -> RBCAction:
        return self.action

    def get_reward(self) -> float:
        return float(-self.simulation.compute_nusselt())

    def __get_info(self) -> dict[str, Any]:
        return {"step": self.tstep, "t": round(self.t, 8)}