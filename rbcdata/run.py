import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from rbcdata.sim.rbc_env import RayleighBenardEnv
from rbcdata.utils.callbacks import (
    ControlVisCallback,
    LogNusseltNumberCallback,
    RBCVisCallback,
    TqdmCallback,
)
from rbcdata.utils.integrate import integrate


def run_env(cfg: DictConfig) -> None:
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        segments=cfg.action_segments,
        action_limit=cfg.action_limit,
    )

    callbacks = [
        TqdmCallback(total=cfg.sim.episode_length),
        RBCVisCallback(
            size=cfg.sim.N,
            vmin=cfg.sim.bcT[1],
            vmax=cfg.sim.bcT[0] + cfg.action_limit,
            interval=cfg.interval,
        ),
        LogNusseltNumberCallback(interval=1),
        ControlVisCallback(size=cfg.action_segments, interval=cfg.interval),
    ]

    # Controller
    controller = hydra.utils.instantiate(
        cfg.controller,
        start=cfg.action_start,
        duration=cfg.action_duration,
        zero=np.array([0.0] * env.segments),
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
