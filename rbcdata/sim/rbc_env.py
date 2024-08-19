import copy
import logging
from typing import Any, Dict, Tuple, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sympy

import glob
from os import listdir
from os.path import isdir, isfile, join

from rbcdata.config import RBCSimConfig
from rbcdata.sim.rayleighbenard2d import RayleighBenard
from rbcdata.sim.tfunc import Tfunc

RBCAction: TypeAlias = npt.NDArray[np.float32]
RBCObservation: TypeAlias = npt.NDArray[np.float32]

x, y, tt = sympy.symbols("x,y,t", real=True)

logger = logging.getLogger(__name__)

class RayleighBenardEnv(gym.Env[RBCAction, RBCObservation]):
    reward_range = (-float("inf"), float("inf"))

    
    def __init__(self, config: Dict) -> None:
        """
        Initialize the Rayleigh-Benard environment with the given configuration Dictionary.
        Note that I had to change the constructor to work with Ray and custom environments
        Further, the config dict should have the following handy properties in addition added by Ray runtime:
        num_env_runners, worker_index, vector_index, and remote.
        """
        super().__init__()
        print(__name__)
        self.config = config    # This is the Ray config dictionary passed to the environment
        sim_cfg = config['sim_cfg']
        # handle the default values
        action_segments = config.get('action_segments', 10)
        action_limit = config.get('action_limit', 0.75)
        action_duration = config.get('action_duration', 1.0)
        action_start = config.get('action_start', 0.0)
        fraction_length_smoothing = config.get('fraction_length_smoothing', 0.1)

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
        nr_probes = sim_cfg.N_obs[0] * sim_cfg.N_obs[1]
        # For every dimension we need to define min and max values
        # The structure of the observations is as follows: first the two velocity components, then the temperature.
        lows = np.concatenate([np.repeat(-np.inf, nr_probes * 2), np.repeat(sim_cfg.bcT[1], nr_probes)])
        highs = np.concatenate([np.repeat(np.inf, nr_probes * 2), np.repeat(sim_cfg.bcT[0] + action_limit, nr_probes)])
        self.observation_space = gym.spaces.Box(
            lows,
            highs,
            shape=(nr_probes * 3,),
            dtype=np.float32,
        )

        load_checkpoint_path = sim_cfg.load_checkpoint_path # Path to checkpoint file or directory for loading checkpoints on env.reset()
        self.load_checkpoint_files = []
        if sim_cfg.load_checkpoint_path is not None:
            if isdir(load_checkpoint_path):
                self.load_checkpoint_files = glob.glob(join(load_checkpoint_path, "*.h5")) 
                if len(self.load_checkpoint_files) == 0:
                    raise ValueError(f"No checkpoint files found in directory: {sim_cfg.load_checkpoint_path}")
            elif isfile(load_checkpoint_path): 
                self.load_checkpoint_files = [sim_cfg.load_checkpoint_path]
            else:
                raise ValueError(f"Invalid path to checkpoint file or directory: {sim_cfg.load_checkpoint_path}")


        # PDE configuration
        self.simulation = RayleighBenard(
            N_state=(sim_cfg.N[0], sim_cfg.N[1]),
            N_obs=(sim_cfg.N_obs[0], sim_cfg.N_obs[1]),
            Ra=sim_cfg.ra,
            Pr=sim_cfg.pr,
            dt=sim_cfg.dt,
            bcT=(sim_cfg.bcT[0], sim_cfg.bcT[1]),
            filename=sim_cfg.save_checkpoint_path,
        )
        self.t_func = Tfunc(
            segments=action_segments,
            domain=self.simulation.domain,
            action_limit=action_limit,
            bcT_avg=self.simulation.bcT_avg,
            x=y,
            fraction_length_smoothing=fraction_length_smoothing,
        )

        logger.warning("Reward scaling in env currently only implemented with values for Ra=1e4, maybe suboptimal for other values.")

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None, filename=None
    ) -> Tuple[RBCObservation, Dict[str, Any]]:
        """Resets the environment to an initial state. If seed is provided, it will be used to seed the environment.
        If filename is provided, it will be used to load the initial state from a checkpoint."""
        super().reset(seed=seed)

        # TODO M: If checkpoint is provided, maybe we should consider to set time to 0 as well
        # because is the time that we took the checkpoint meaningful? It may complicate things.
        # init PDE simulation
        # M: we should reset the boundary conditions here as well I guess, before the initialize method is called, becauses it uses the boundary conditions.
        self.simulation.update_actuation(self.simulation.bcT_avg)
        if len(self.load_checkpoint_files) > 0:
            # choose a random checkpoint file to load from
            file_idx = self._np_random.choice(len(self.load_checkpoint_files))
            filename = self.load_checkpoint_files[file_idx]
        # initialize the simulation from a file or randomly depending on whether filename is none or not 
        self.t, self.tstep = self.simulation.initialize(
            filename=filename, np_random=self._np_random, rand=0.000001
        )

        self.simulation.assemble()
        self.simulation.step(self.t, self.tstep)

        # Reset action
        self.action = np.array([0.0])
        self.action_effective = None  # TODO sympy zero

        if filename is None:
            logger.info(f"Environment reset to random init: t={self.t}")
        else:
            logger.info(f"Environment reset from checkpoint file {filename}: t={self.t}")

        return self.get_obs(), self.__get_info()

    def step(self, action: RBCAction) -> Tuple[RBCObservation, float, bool, bool, Dict[str, Any]]:
        """
        Function to perform one step of the environment using action "action", i.e.
        (state(t), action(t)) -> state(t+1)
        """
        truncated = False
        # Apply action
        self.action = action
        # self.action_effective = self.t_func.apply_T(copy.deepcopy(action))  # TODO Why was deepcopy necessary again? There is no writing to the action.
        self.action_effective = self.t_func.apply_T(action) # action_effective is a sympy piecewise expression.
        self.simulation.update_actuation((self.action_effective, self.simulation.bcT[1]))

        # Perform simulation steps for the action duration
        for _ in range(int(self.action_duration / self.simulation.dt)):
            self.t, self.tstep = self.simulation.step(tstep=self.tstep, t=self.t)

        # Check for truncation
        if self.t >= self.episode_length:
            truncated = True

        # Compute the reward
        reward = self.get_reward()
        # scale the reward to [0, 1] approximately. 0 associated with highest Nusselt number, 1 with lowest achievable
        # TODO scaling is currently only implemented with values for Ra=1e4, maybe suboptimal for other values
        reward = (reward + 2.67) / 2.67 # TODO find out more about what the lowest achievable Nusselt number is
        logger.debug(f"Action done: t={self.t}, reward={reward}")
        return self.get_obs(), reward, self.closed, truncated, self.__get_info()

    def close(self) -> None:
        self.closed = True

    def get_obs(self) -> RBCObservation:
        """
        Returns the observation of the current state. Note that the state is observed at the probes only, so it 
        is a partially observed state.
        """
        # TODO Change this so the Y-velocity is the first channel! For now it's OK.
        return self.simulation.obs_flat.astype(np.float32)

    def get_state(self) -> RBCObservation:
        return self.simulation.state.astype(np.float32)

    def get_action(self) -> RBCAction:
        return self.action

    def get_reward(self) -> float:
        return float(-self.simulation.compute_nusselt())

    def __get_info(self) -> dict[str, Any]:
        return {"step": self.tstep, "t": round(self.t, 8)}
