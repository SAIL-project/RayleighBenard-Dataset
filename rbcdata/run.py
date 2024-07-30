import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from rbcdata.sim.rbc_env import RayleighBenardEnv
from rbcdata.utils.callbacks import (
    LogNusseltNumberCallback,
    RBCVisCallback,
    TqdmCallback,
)
from rbcdata.utils.integrate import integrate


def run_env(cfg: DictConfig) -> None:
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        segments=cfg.segments,
        action_limit=cfg.action_limit,
        action_duration=cfg.action_duration,
    )

    callbacks = [
        TqdmCallback(total=cfg.sim.episode_length),
        # RBCVisCallback(
        #     size=cfg.sim.N,
        #     vmin=cfg.sim.bcT[1],
        #     vmax=cfg.sim.bcT[0] + cfg.action_limit,
        #     interval=cfg.interval,
        # ),
        #LogNusseltNumberCallback(interval=1),
    ]

    # Controller
    controller = hydra.utils.instantiate(
        cfg.controller,
        start_time=cfg.action_start,
        control_duration=cfg.action_duration,
        zero_control=np.array([0.0] * env.segments),
    )

    integrate(
        env=env,
        callbacks=callbacks,
        seed=cfg.seed,
        controller=controller,
    )


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    run_env(cfg=cfg)


if __name__ == "__main__":
    main()
