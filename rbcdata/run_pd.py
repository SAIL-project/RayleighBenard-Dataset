import hydra
import wandb
from omegaconf import DictConfig

from rbcdata.callbacks.callbacks import SaveNusseltNumberCallback, TqdmCallback
from rbcdata.callbacks.callbacks_wandb import LogNusseltNumberCallback
from rbcdata.control.pd_control import PDController
from rbcdata.env.rbc_env import RayleighBenardEnv
from rbcdata.utils.integrate import integrate


@hydra.main(version_base=None, config_path="config", config_name="pdcontrol")
def main(cfg: DictConfig) -> None:
    # Logging
    if cfg.baseline:
        tags = ["baseline"]
    else:
        tags = ["pd"]
    tags.append(f"ra{cfg.env.ra}")

    run = wandb.init(
        project="RayleighBenard-PDControl",
        dir=cfg.paths.output_dir,
        config=dict(cfg),
        tags=tags,
    )

    # Environment
    env = RayleighBenardEnv(env_config=cfg.env, render_mode=cfg.render_mode)

    # Callbacks
    callbacks = [
        TqdmCallback(total=env.episode_length, interval=cfg.interval),
        SaveNusseltNumberCallback(log_wandb=True),
        # LogVisualizationCallback(action_limit=cfg.env.action_limit),
        LogNusseltNumberCallback(interval=cfg.interval, nr_episodes=cfg.nr_episodes),
        # LogActionCallback(interval=cfg.interval),
    ]

    # Controller
    if not cfg.baseline:
        controller = PDController(**cfg.controller)
    else:
        controller = None

    # Rollout
    for idx in range(cfg.nr_episodes):
        integrate(
            env=env,
            controller=controller,
            callbacks=callbacks,
            seed=cfg.seed,
            episode_idx=idx,
        )

    # close environment and callbacks
    env.close()
    for callback in callbacks:
        callback.close()

    # Finish logging
    run.finish()


if __name__ == "__main__":
    main()
