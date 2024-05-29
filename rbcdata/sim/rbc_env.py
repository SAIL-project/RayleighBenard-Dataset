import math
from typing import Any, Dict, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sympy
from tqdm import tqdm

from rbcdata.config import RBCSimConfig
from rbcdata.sim.rayleighbenard2d import RayleighBenard
from rbcdata.sim.tfunc import Tfunc
from rbcdata.utils.rbc_field import RBCField
from rbcdata.vis import RBCFieldVisualizer
from rbcdata.vis import RBCActionVisualizer

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]

x, y, tt = sympy.symbols("x,y,t", real=True)


class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):
    """
    Class manages a 
    """
    metadata = {"render_modes": ["live", "rgb_array"]}
    reward_range = (-float("inf"), float("inf"))

    def __init__(
        self,
        sim_cfg: RBCSimConfig,
        segments: int = 10,
        action_scaling: float = 0.75,
        action_duration: int = 1,
        modshow: int = 20,
        render_mode: str | None = None,
        tqdm_position: int = 0,
    ) -> None:
        super().__init__()

        # Env configuration
        self.cfg = sim_cfg
        self.segments = segments
        self.action_scaling = action_scaling
        self.solver_steps = math.floor(action_duration / sim_cfg.dt)
        self.sim_steps = round(sim_cfg.episode_length / sim_cfg.dt)
        self.env_steps = math.floor(self.sim_steps / self.solver_steps)
        self.cook_steps = round(sim_cfg.cook_length / sim_cfg.dt)
        self.closed = False

        # Progress bar
        self.pbar = tqdm(
            total=self.cook_steps + self.sim_steps, leave=False, position=tqdm_position
        )

        # Action configuration
        self.dicTemp = {}
        # starting temperatures
        for i in range(segments):
            self.dicTemp["T" + str(i)] = sim_cfg.bcT[0]

        # TODO Should this be -1, 1?
        # self.action_space = gym.spaces.Box(
        #     -action_scaling,
        #     action_scaling,
        #     shape=(segments,),
        #     dtype=np.float32,
        # )

        self.action_space = gym.spaces.Box(
            -1,
            1,
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
            nb_seg=segments, domain=self.simulation.domain, action_scaling=action_scaling
        )

        # Render configuration
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.modshow = modshow
        if render_mode == "live":
            self.window = RBCFieldVisualizer(
                size=sim_cfg.N,
                vmin=sim_cfg.bcT[1],
                vmax=sim_cfg.bcT[0] + action_scaling,
                show=True,
                show_u=True,
            )

            # self.action_window = RBCActionVisualizer(True, self.simulation.domain[1], n_segments_plot=100) 
        self.render_mode = render_mode

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        # init PDE simulation
        self.sim_t, self.sim_step = self.simulation.initialize()
        self.env_step = 0
        self.simulation.assemble()

        # init visualizer
        if self.render_mode is not None:
            self.visualizer = None

        # cook system
        self.pbar.set_description("Cook System...")
        for _ in range(self.cook_steps):
            self.sim_t, self.sim_step = self.simulation.step(self.sim_t, self.sim_step)
            if self.sim_step > 1:
                self.render(cooking=True)
            self.pbar.update(1)

        # Reset progress bar description
        self.pbar.set_description("Episode")

        # Reset action
        self.action = np.array([0.0] * self.segments)
        self.action_effective = np.array([0.0 * self.segments])

        return self.get_obs(), self.__get_info()

    def step(self, action: RBCAction) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        """
        Function to perform one step of the environment using action "action", i.e.
        (state(t), action(t)) -> state(t+1)
        """
        truncated = False
        # Apply action
        self.action = action # TODO this should be set to the action that is truly applied after t_func?
        for i in range(self.segments):
            self.dicTemp.update({"T" + str(i): action[i]})  # apply the value to each segment
        self.action_effective = self.t_func.apply_T(dicTemp=self.dicTemp, x=y, bcT_avg=self.simulation.bcT_avg)   # Returns Sympy Piecewise for the action
        # print(self.t_func.apply_T(dicTemp=self.dicTemp, x=y))
        # print(self.dicTemp)
        self.simulation.update_actuation((self.action_effective, 1))
        # PDE stepping, simulates the system while performing the action for action_duration nr. of steps
        for _ in range(self.solver_steps):
            self.sim_t, self.sim_step = self.simulation.step(tstep=self.sim_step, t=self.sim_t)
            self.pbar.update(1)

        # Check for truncation
        self.env_step += 1
        if self.env_step >= self.env_steps:
            truncated = True

        # Update vis
        self.render()

        return self.get_obs(), 0, self.closed, truncated, self.__get_info()

    def render(self, cooking: bool = False) -> None:
        # TODO: isn't it better to make the second test in units of env_step, i.e. env_step % modshow == 0?
        if self.render_mode == "live" and self.sim_step % self.modshow == 0:
            state = self.get_state()
            self.window.draw(
                state[RBCField.T],
                state[RBCField.UX],
                state[RBCField.UY],
                self.sim_step * self.cfg.dt,
                cooking=cooking,
            )
            
            # TODO for now only show applied action if not cooking, but could also show when cooking
            # if not cooking:
            #    self.action_window.draw(self.action_effective)

    def close(self) -> None:
        self.closed = True
        if self.render_mode == "live":
            self.window.close()

    def get_obs(self) -> RBCObservation:
        return self.simulation.obs_flat.astype(np.float32)

    def get_state(self) -> RBCObservation:
        return self.simulation.state.astype(np.float32)

    def get_action(self) -> RBCAction:
        return self.action

    def get_reward(self) -> float:
        return float(-self.simulation.compute_nusselt())

    def __get_info(self) -> dict[str, Any]:
        return {"step": self.env_step, "sim_step": self.sim_step, "sim_t": round(self.sim_t, 7)}
