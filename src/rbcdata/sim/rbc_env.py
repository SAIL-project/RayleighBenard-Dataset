import math
import time
from typing import Any, Dict, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from rbcdata.config import RBCEnvConfig
from rbcdata.utils.rbc_field import RBCField
from rbcdata.utils.rbc_simulation_params import RBCSimulationParams
from rbcdata.vis import RBCFieldVisualizer

from .rayleighbenard2d import RayleighBenard2D

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]


class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):
    metadata = {"render_modes": ["live", "rgb_array"]}
    reward_range = (-float("inf"), float("inf"))

    def __init__(
        self,
        cfg: RBCEnvConfig,
        savestatistics: bool = False,
        modshow: int = 20,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        # Env configuration
        self.cfg = cfg
        self.steps = round(cfg.episode_length / cfg.dt)
        self.cook_steps = round(cfg.cook_length / cfg.dt)

        # PDE configuration
        unique = math.floor(time.time() * 1000)
        sim_params = RBCSimulationParams(
            N=cfg.N,
            Ra=cfg.Ra,
            Pr=cfg.Pr,
            dt=cfg.dt,
            bcT=cfg.bcT,
            domain=cfg.domain,
            filename=f"{cfg.ckpt_path}/{unique}/ra{cfg.Ra}/RB_2D",
        )
        self.simulation = RayleighBenard2D(sim_params)

        # Action configuration
        self.action_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(1, 10),
            dtype=np.float32,
        )

        # Observation Space
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(
                1,
                cfg.N[0] * cfg.N[1] * 3,
            ),  # *3 because of dim of velocity and temperature field
            dtype=np.float32,
        )

        # Statistics
        self.savestatistics = savestatistics
        self.nusselts = np.zeros(self.steps)
        self.energies = np.zeros(self.steps)

        # Render configuration
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.modshow = modshow
        if render_mode == "live":
            self.window = RBCFieldVisualizer(
                size=cfg.N, vmin=cfg.bcT[1], vmax=cfg.bcT[0], show=True, show_u=True
            )
        self.render_mode = render_mode

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        # init PDE simulation
        self.t, self.tstep = self.simulation.initialize()
        self.simulation.assemble()

        # init visualizer
        if self.render_mode is not None:
            self.visualizer = None

        # cook system
        for _ in tqdm(
            range(self.cook_steps), desc="Cooking Time", position=1, leave=False
        ):
            self.t, self.tstep = self.simulation.step(self.t, self.tstep)
            self.__save_statistics()
            if self.tstep > 1:
                self.render(cooking=True)

        # Reset progress bar
        self.pbar = tqdm(total=self.steps, position=1, leave=False)
        self.pbar.set_description("Episode")

        return self.get_obs(), self.__get_info()

    def step(
        self, action: RBCAction
    ) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        truncated = False
        # PDE stepping
        for _ in range(self.cfg.solver_steps):
            self.t, self.tstep = self.simulation.step(tstep=self.tstep, t=self.t)
            self.__save_statistics()

            # Check for truncation
            self.pbar.update(1)
            if self.tstep >= self.steps + self.cook_steps:
                truncated = True
                break

            # Update vis
            self.render()

        return self.get_obs(), 0, False, truncated, self.__get_info()

    def render(self, cooking: bool = False) -> None:
        if self.render_mode == "live" and self.tstep % self.modshow == 0:
            state = self.get_state()
            self.window.draw(
                state[RBCField.T],
                state[RBCField.UX],
                state[RBCField.UY],
                self.tstep * self.cfg.dt,
                cooking=cooking,
            )

    def close(self) -> None:
        self.simulation.clean()
        if self.render_mode == "live":
            self.window.close()

    def get_obs(self) -> RBCObservation:
        return self.simulation.obs_flat.astype(np.float32)

    def get_state(self) -> RBCObservation:
        return self.simulation.state.astype(np.float32)

    def get_reward(self) -> float:
        return -self.simulation.compute_nusselt()

    def get_statistics(self) -> npt.NDArray[np.float32]:
        return np.array(
            [
                self.simulation.compute_nusselt(),
                self.simulation.compute_kinematic_energy(),
            ]
        )

    def __save_statistics(self) -> None:
        if self.savestatistics and self.tstep > 0:
            self.nusselts[self.tstep] = self.simulation.compute_nusselt()
            self.energies[self.tstep] = self.simulation.compute_kinematic_energy()

    def __get_info(self) -> dict[str, Any]:
        return {"step": self.tstep, "t": round(self.t, 7)}
