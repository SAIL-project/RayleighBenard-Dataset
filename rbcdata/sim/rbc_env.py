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
        segments: int,
        action_limit: float,
    ) -> None:
        super().__init__()

        # Env configuration
        self.cfg = sim_cfg
        self.segments = segments
        self.action_limit = action_limit
        self.env_steps = round(sim_cfg.episode_length / sim_cfg.dt)
        self.closed = False

        # Action configuration
        self.dicTemp = {}
        # starting temperatures
        for i in range(segments):
            self.dicTemp["T" + str(i)] = sim_cfg.bcT[1]

        self.action_space = gym.spaces.Box(
            -action_limit,
            action_limit,
            shape=(segments,),
            dtype=np.float32,
        )

        # Observation Space
        self.observation_space = gym.spaces.Box(
            sim_cfg.bcT[1],
            sim_cfg.bcT[0],
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
            nb_seg=segments, domain=self.simulation.domain, action_scaling=action_limit
        )

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        # init PDE simulation and do one step
        self.sim_t, self.sim_step = self.simulation.initialize()
        self.env_step = 0
        self.simulation.assemble()
        self.simulation.step(self.sim_t, self.sim_step)

        # Reset action
        self.action = np.array([0.0] * self.segments)

        return self.get_obs(), self.__get_info()

    def step(self, action: RBCAction) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        truncated = False
        # Apply action
        if not np.array_equiv(action, self.action):
            self.action = action
            for i in range(self.segments):
                self.dicTemp.update({"T" + str(i): action[i]})
            self.simulation.update_actuation((self.t_func.apply_T(dicTemp=self.dicTemp, x=y), 1))
        # PDE step
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
