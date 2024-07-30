from typing import Any, Dict, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sympy
import math

from rbcdata.config import RBCSimConfig
from rbcdata.sim.rayleighbenard2d import RayleighBenard
from rbcdata.sim.tfunc import Tfunc

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]

x, y, tt = sympy.symbols("x,y,t", real=True)


class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):
    reward_range = (-float("inf"), float("inf"))

    def __init__(
        self,
        sim_cfg: RBCSimConfig,
        segments: int = 10,
        action_limit: float = 0.75,
        action_duration: int = 1,
        fraction_length_smoothing=0.1
    ) -> None:
        super().__init__()

        # Env configuration
        self.cfg = sim_cfg
        self.segments = segments
        self.action_limit = action_limit
        self.solver_steps_action = math.floor(action_duration / sim_cfg.dt)        # simulation steps taken for one action
        self.sim_steps = round(sim_cfg.episode_length / sim_cfg.dt)         # simulation steps taken in one episode (after cooking)
        self.env_steps = math.floor(self.sim_steps / self.solver_steps_action)     # The total number of actions taken over the whole episode
        self.cook_steps = round(sim_cfg.cook_length / sim_cfg.dt)           # The number simulation steps for cooking
        self.closed = False

        # Action configuration, starting temperatures
        self.temperature_segments = np.ones(segments) * sim_cfg.bcT[0]

        # The reinforcement learning should take actions between [-1, 1] on the bottom segments according to Vignon...
        self.action_space = gym.spaces.Box(
            -1,
            1,
            shape=(segments,),
            dtype=np.float32,
        )

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
        self.t_func = Tfunc(
            segments=segments,
            domain=self.simulation.domain,
            action_limit=action_limit,
            fraction_length_smoothing=fraction_length_smoothing
        )

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None, filename=None
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        # init PDE simulation
        self.sim_t, self.sim_step = self.simulation.initialize(filename=filename)
        self.sim_t = 0.0
        self.sim_step = 0
        self.env_step = 0
        self.simulation.assemble()
        self.simulation.step(self.sim_t, self.sim_step)

        # Reset action
        self.action = np.array([0.0] * self.segments)
        self.action_effective = np.array([0.0] * self.segments)

        return self.get_obs(), self.__get_info()

    def step(self, action: RBCAction) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        """
        Function to perform one step of the environment using action "action", i.e.
        (state(t), action(t)) -> state(t+1)
        """
        truncated = False
        # Apply action
        self.action = action
        for i in range(self.segments):
            self.temperature_segments[i] = action[i]    # apply given temperature value to each segment
        self.action_effective = self.t_func.apply_T(self.temperature_segments, x=y, bcT_avg=self.simulation.bcT_avg)   # Returns Sympy Piecewise for the action
        self.simulation.update_actuation((self.action_effective, 1))
        self.sim_t, self.sim_step = self.simulation.step(tstep=self.sim_step, t=self.sim_t)

        # Check for truncation
        self.env_step += 1
        if self.env_step >= self.env_steps:
            truncated = True

        return self.get_obs(), 0, self.closed, truncated, self.__get_info()

    def close(self) -> None:
        self.closed = True

    def get_obs(self) -> RBCObservation:
        return self.simulation.obs_flat.astype(np.float32)

    def get_state(self) -> RBCObservation:
        return self.simulation.state.astype(np.float32)

    def get_action(self) -> RBCAction:
        return self.action

    def get_reward(self) -> float:
        return float(-self.simulation.compute_nusselt())

    def __get_info(self) -> dict[str, Any]:
        return {"step": self.env_step, "t": round(self.sim_t, 7)}
