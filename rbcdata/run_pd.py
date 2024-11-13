import hydra
import rootutils
import wandb
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.callbacks.callbacks import TqdmCallback
from rbcdata.callbacks.callbacks_wandb import (
    LogActionCallback,
    LogNusseltNumberCallback,
    LogVisualizationCallback,
)
from rbcdata.control.pd_control import PDController
from rbcdata.env.rbc_env import RayleighBenardEnv
from rbcdata.utils.integrate import integrate


@hydra.main(version_base=None, config_path="config", config_name="pdcontrol")
def main(cfg: DictConfig) -> None:
    # Logging
    run = wandb.init(
        project="RayleighBenard-PDControl",
        dir=cfg.paths.output_dir,
        config=dict(cfg),
    )

    # Environment
    env = RayleighBenardEnv(env_config=cfg.env, render_mode=cfg.render_mode)

    # Callbacks
    callbacks = [
        TqdmCallback(total=env.episode_length, interval=cfg.interval),
        LogVisualizationCallback(action_limit=cfg.env.action_limit, interval=cfg.interval),
        LogNusseltNumberCallback(interval=cfg.interval),
        LogActionCallback(interval=cfg.interval),
    ]

    # Controller
    controller = PDController(**cfg.controller)

    # Rollout
    integrate(
        env=env,
        controller=controller,
        callbacks=callbacks,
        seed=cfg.seed,
    )

    # Finish logging
    run.finish()


if __name__ == "__main__":
    main()
