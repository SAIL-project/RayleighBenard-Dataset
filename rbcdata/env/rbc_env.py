import glob
import logging
from os.path import exists, isdir, isfile, join
from typing import Any, Dict, Optional, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sympy
from gymnasium.error import DependencyNotInstalled
from hydra.utils import to_absolute_path

from rbcdata.env.sim.rayleighbenard2d import RayleighBenard
from rbcdata.env.sim.tfunc import Tfunc
from rbcdata.utils.rbc_field import RBCField
from rbcdata.vis.utils import colormap

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]

x, y, tt = sympy.symbols("x,y,t", real=True)


class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }
    logger = logging.getLogger(__name__)

    EPISODE_LENGTH = 300
    SIZE_STATE = [64, 96]
    SIZE_OBS = [8, 48]
    RA = 10_000
    PR = 0.7
    DT = 0.05
    BCT = [2, 1]
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
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the Rayleigh-Benard environment with the given configuration Dictionary.
        """
        super().__init__()
        # write checkpoint path
        write_checkpoint = env_config.get("write_checkpoint", self.WRITE_CHECKPOINT)
        self.path = "shenfun"
        if not write_checkpoint:
            self.path = None

        # initialize from checkpoint path
        self.checkpoint = to_absolute_path(env_config.get("checkpoint", self.CHECKPOINT))
        self.load_checkpoint_files = []
        if self.checkpoint is not None:
            self.logger.info(f"Loading checkpoint from {self.checkpoint}")
            if not exists(self.checkpoint):
                raise ValueError(f"Path to checkpoint does not exist: {self.checkpoint}")
            elif isdir(self.checkpoint):
                self.load_checkpoint_files = glob.glob(join(self.checkpoint, "*.h5"))
                if len(self.load_checkpoint_files) == 0:
                    raise ValueError(f"No checkpoint files found in directory: {self.checkpoint}")
            elif isfile(self.checkpoint):
                self.load_checkpoint_files = [self.checkpoint]
            else:
                raise ValueError(
                    f"Invalid path to checkpoint file or directory: {self.checkpoint}"
                )

        # simulation config
        self.episode_length = env_config.get("episode_length", self.EPISODE_LENGTH)
        self.size_state = env_config.get("size_state", self.SIZE_STATE)
        self.size_obs = env_config.get("size_obs", self.SIZE_OBS)
        self.ra = env_config.get("ra", self.RA)
        self.pr = env_config.get("pr", self.PR)
        self.dt = env_config.get("dt", self.DT)
        self.bcT = env_config.get("bcT", self.BCT)

        # action config
        self.action_limit = env_config.get("action_limit", self.ACTION_LIMIT)
        self.action_duration = env_config.get("action_duration", self.ACTION_DURATION)
        self.action_segments = env_config.get("action_segments", self.ACTION_SEGMENTS)
        self.action_start = env_config.get("action_start", self.ACTION_START)
        self.fraction_length_smoothing = env_config.get(
            "fraction_length_smoothing", self.FRACTION_LENGTH_SMOOTHING
        )
        self.solver_steps = int(self.action_duration / self.dt)

        # Env configuration
        self.obs_list = []
        self.episode_steps = int(self.episode_length / self.dt)
        self.closed = False

        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.action_segments,), dtype=np.float32)

        # Observation Space
        lows = np.stack(
            [
                np.ones(self.size_obs) * (-np.inf),
                np.ones(self.size_obs) * (-np.inf),
                np.ones(self.size_obs) * self.bcT[1],
            ],
            axis=0,
        )
        highs = np.stack(
            [
                np.ones(self.size_obs) * np.inf,
                np.ones(self.size_obs) * np.inf,
                np.ones(self.size_obs) * self.bcT[0] + self.action_limit,
            ],
            axis=0,
        )
        self.observation_space = gym.spaces.Box(
            lows,
            highs,
            shape=(
                3,
                self.size_obs[0],
                self.size_obs[1],
            ),
            dtype=np.float32,
        )

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 768
        self.screen_height = 512
        self.screen = None
        self.clock = None

        # Warnings and TODOs
        self.logger.warning(
            "Reward scaling in env currently only implemented with values for Ra=1e4, maybe \
                suboptimal for other values."
        )

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        """Resets the environment to an initial state. If seed is provided, it will be used
        to seed the environment.
        If filename is provided, it will be used to load the initial state from a checkpoint."""
        super().reset(seed=seed)
        # PDE configuration
        self.simulation = RayleighBenard(
            N_state=tuple(self.size_state),
            N_obs=tuple(self.size_obs),
            Ra=self.ra,
            Pr=self.pr,
            dt=self.dt,
            bcT=tuple(self.bcT),
            checkpoint=self.path,
        )
        self.t_func = Tfunc(
            segments=self.action_segments,
            domain=self.simulation.domain,
            action_limit=self.action_limit,
            bcT_avg=self.simulation.bcT_avg,
            x=y,
            fraction_length_smoothing=self.fraction_length_smoothing,
        )

        # load checkpoint file
        filename = None
        if len(self.load_checkpoint_files) > 0:
            # choose a random checkpoint file to load from
            file_idx = self.np_random.choice(len(self.load_checkpoint_files))
            filename = self.load_checkpoint_files[file_idx]
        # initialize the simulation from a file or randomly depending on whether filename is
        # none or not
        if filename is not None and filename.endswith(".chk.h5"):
            filename = filename[:-7]  # cut the extension off

        # init PDE simulation
        self.t, self.tstep = self.simulation.initialize(
            checkpoint=filename, np_random=self._np_random, rand=0.000001
        )
        self.simulation.assemble()
        self.simulation.step()
        self.obs_list = []

        # Reset action
        self.last_action = np.array([0.0])
        self.action_effective = None  # TODO sympy zero

        # Logging
        if filename is None:
            self.logger.info(f"Environment reset to random init: t={self.t}")
        else:
            self.logger.info(f"Environment reset from checkpoint file {filename}: t={self.t}")

        return self.__get_obs(), self.__get_info()

    def step(self, action: RBCAction) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        """
        Function to perform one step of the environment using action "action", i.e.
        (state(t), action(t)) -> state(t+1)
        """
        truncated = False
        # Apply action
        self.last_action = action
        self.action_effective = self.t_func.apply_T(action)
        self.simulation.update_actuation((self.action_effective, self.simulation.bcT[1]))

        for _ in range(self.solver_steps):
            self.t, self.tstep = self.simulation.step(tstep=self.tstep, t=self.t)

        # Check for truncation
        if self.tstep >= self.episode_steps:
            truncated = True

        self.last_obs = self.__get_obs()
        self.last_reward = self.__get_reward()
        self.last_info = self.__get_info()

        return self.last_obs, self.last_reward, self.closed, truncated, self.last_info

    def close(self) -> None:
        self.closed = True

    def get_state(self) -> RBCObservation:
        return self.simulation.get_state().astype(np.float32)

    def get_action(self) -> RBCAction:
        return self.last_action

    def __get_obs(self) -> RBCObservation:
        return self.simulation.get_obs().astype(np.float32)

    def __get_reward(self) -> float:
        obs = self.__get_obs()
        neg_nusselt_nr = float(-self.simulation.compute_nusselt(obs))
        # TODO scaling is currently only implemented with values for Ra=1e4, maybe
        # suboptimal for other values
        reward = (
            neg_nusselt_nr + 2.67
        ) / 2.67  # TODO find out more about what the lowest achievable Nusselt number is
        return reward

    def __get_info(self) -> dict[str, Any]:
        return {
            "step": self.tstep,
            "t": self.t,
            "state": self.get_state(),
            "nusselt_obs": self.simulation.compute_nusselt(self.__get_obs()),
            "nusselt": self.simulation.compute_nusselt(self.get_state()),
        }

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Rayleigh Benard Convection")
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Data
        data = self.get_state()[RBCField.T]
        data = np.transpose(data)
        data = colormap(data, vmin=self.bcT[1], vmax=self.bcT[0] + self.action_limit)

        if self.render_mode == "human":
            canvas = pygame.Surface((self.size_state[1], self.size_state[0]))
            pygame.surfarray.blit_array(canvas, data)

            # scale canvas
            canvas = pygame.transform.scale(canvas, (self.screen_width, self.screen_height))
            self.screen.blit(canvas, (0, 0))

            # Show screen
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return data.transpose(1, 0, 2)
