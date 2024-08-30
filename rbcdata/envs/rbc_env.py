import copy
from typing import Any, Dict, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sympy

from rbcdata.envs.sim.rayleighbenard2d import RayleighBenard
from rbcdata.envs.sim.tfunc import Tfunc

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]

x, y, tt = sympy.symbols("x,y,t", real=True)


class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):

    EPISODE_LENGTH = 300
    SIZE_STATE = [64, 96]
    SIZE_OBS = [8, 48]
    RA = 10_000
    PR = 0.7
    DT = 0.05
    SOLVER_STEPS = 20
    BCT = [2, 1]
    BASE_PATH = "logs/environment"
    CHECKPOINT = None
    WRITE_CHECKPOINT = False
    ACTION_LIMIT = 0.75
    ACTION_DURATION = 1.0
    ACTION_SEGMENTS = 12
    ACTION_START = 0.0
    FRACTION_LENGTH_SMOOTHING = 0.1

    def __init__(
        self,
        env_config: Dict,
    ) -> None:
        super().__init__()
        # env runner config
        self.worker_index = env_config.get("worker_index", 0)
        self.vector_index = env_config.get("vector_index", 0)

        # write checkpoint path
        write_checkpoint = env_config.get("write_checkpoint", self.WRITE_CHECKPOINT)
        base_path = env_config.get("base_path", self.BASE_PATH)
        path = f"{base_path}/worker_{self.worker_index}_vector_{self.vector_index}/shenfun"
        if not write_checkpoint:
            path = None

        # initialize from checkpoint path
        self.checkpoint = env_config.get("checkpoint", self.CHECKPOINT)

        # simulation config
        self.episode_length = env_config.get("episode_length", self.EPISODE_LENGTH)
        self.size_state = env_config.get("size_state", self.SIZE_STATE)
        self.size_obs = env_config.get("size_obs", self.SIZE_OBS)
        self.ra = env_config.get("ra", self.RA)
        self.pr = env_config.get("pr", self.PR)
        self.dt = env_config.get("dt", self.DT)
        self.solver_steps = env_config.get("solver_steps", self.SOLVER_STEPS)
        self.bcT = env_config.get("bcT", self.BCT)

        # action config
        self.action_limit = env_config.get("action_limit", self.ACTION_LIMIT)
        self.action_duration = env_config.get("action_duration", self.ACTION_DURATION)
        self.action_segments = env_config.get("action_segments", self.ACTION_SEGMENTS)
        self.action_start = env_config.get("action_start", self.ACTION_START)
        self.fraction_length_smoothing = env_config.get(
            "fraction_length_smoothing", self.FRACTION_LENGTH_SMOOTHING
        )

        # Env configuration
        self.episode_steps = int(self.episode_length / self.dt)
        self.closed = False

        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.action_segments,), dtype=np.float32)

        # Observation Space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                3,
                self.size_obs[0],
                self.size_obs[1],
            ),
            dtype=np.float32,
        )

        # PDE configuration
        self.simulation = RayleighBenard(
            N_state=tuple(self.size_state),
            N_obs=tuple(self.size_obs),
            Ra=self.ra,
            Pr=self.pr,
            dt=self.dt,
            bcT=tuple(self.bcT),
            checkpoint=path,
        )
        self.t_func = Tfunc(
            segments=self.action_segments,
            domain=self.simulation.domain,
            action_limit=self.action_limit,
            bcT_avg=self.simulation.bcT_avg,
            x=y,
            fraction_length_smoothing=self.fraction_length_smoothing,
        )

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        # init PDE simulation
        self.t, self.tstep = self.simulation.initialize(
            checkpoint=self.checkpoint, np_random=self._np_random, rand=0.000001
        )
        self.simulation.assemble()
        self.simulation.step()

        # Reset action
        self.action = np.array([0.0])
        self.action_effective = None  # TODO sympy zero

        return self.__get_obs(), self.__get_info()

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

        for _ in range(self.solver_steps):
            self.t, self.tstep = self.simulation.step(tstep=self.tstep, t=self.t)

        # Check for truncation
        if self.t >= self.episode_length:
            truncated = True

        return self.__get_obs(), self.__get_reward(), self.closed, truncated, self.__get_info()

    def close(self) -> None:
        self.closed = True

    def get_state(self) -> RBCObservation:
        return self.simulation.get_state().astype(np.float32)

    def get_action(self) -> RBCAction:
        return self.action

    def get_nusselt(self) -> float:
        return self.simulation.compute_nusselt(self.__get_obs())

    def __get_obs(self) -> RBCObservation:
        return self.simulation.get_obs().astype(np.float32)

    def __get_reward(self, from_obs=False) -> float:
        if from_obs:
            state = self.__get_obs()
        else:
            state = self.get_state()
        return float(-self.simulation.compute_nusselt(state))

    def __get_info(self) -> dict[str, Any]:
        return {"step": self.tstep, "t": self.t}
