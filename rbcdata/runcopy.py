from typing import List

import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from rbcdata.sim.rbc_env import RayleighBenardEnv
from rbcdata.utils.callbacks import (
    CallbackBase,
    SweepMetricCallback,
    instantiate_callbacks,
)
from rbcdata.utils.integrate import integrate


def run_env(cfg: DictConfig) -> None:
    env = RayleighBenardEnv(
        sim_cfg=cfg.sim,
        action_segments=cfg.action_segments,
        action_limit=cfg.action_limit,
    )

    # Callbacks
    callbacks: List[CallbackBase] = instantiate_callbacks(cfg.get("callbacks"))
    sweep_metric = SweepMetricCallback(
        action_start=cfg.action_start,
        action_end=cfg.action_end,
    )
    callbacks.append(sweep_metric)

    # Controller
    controller = hydra.utils.instantiate(
        cfg.controller,
        start=cfg.action_start,
        end=cfg.action_end,
        duration=cfg.action_duration,
        zero=np.array([0.0] * env.action_segments),
    )

    # Rollout
    integrate(
        env=env,
        callbacks=callbacks,
        seed=cfg.seed,
        controller=controller,
        checkpoint=cfg.checkpoint,
    )

    return sweep_metric.result()["nu_mean_action"]


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    return run_env(cfg=cfg)


if __name__ == "__main__":
    main()
