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

x, y, tt = sympy.symbols("x,y,t", real=True)


class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):
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

        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_segments,), dtype=np.float32)

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
        self.t, self.tstep = self.simulation.initialize(filename=filename)
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
        return {"step": self.tstep, "t": round(self.t, 8)}
