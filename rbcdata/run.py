import hydra
from omegaconf import DictConfig

from rbcdata.callbacks.callbacks import TqdmCallback
from rbcdata.env.rbc_env import RayleighBenardEnv
from rbcdata.utils.integrate import integrate


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    env = RayleighBenardEnv(env_config=cfg.env, render_mode=cfg.render_mode)

    # Callbacks
    callbacks = [
        TqdmCallback(total=env.steps, interval=cfg.interval),
    ]

    # Rollout
    integrate(
        env=env,
        callbacks=callbacks,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    main()
