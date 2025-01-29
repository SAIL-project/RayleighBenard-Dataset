from os.path import join

import hydra
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import yaml
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from rbcdata.env.rbc_env import RayleighBenardEnv
from wandb import Table, Video


@hydra.main(version_base=None, config_path="config", config_name="sbsa_eval")
def main(cfg: DictConfig) -> None:
    # Configure logging
    output_dir = HydraConfig.get().runtime.output_dir
    # wandb
    wandb.init(
        project="sb3-single-agent",
        config=dict(cfg),
        sync_tensorboard=True,
        dir=output_dir,
        tags=["eval"],
    )
    # sb3 logging
    logger = configure(join(output_dir, "log"), ["stdout", "log", "json", "tensorboard"])
    logger.info(f"Set log directory to {output_dir}")

    # TODO: new config for evaluation. rn: use train config
    with open(join(cfg.experiment_dir, ".hydra/config.yaml")) as file:
        config = DictConfig(yaml.safe_load(file))
        model_path = join(cfg.experiment_dir, "model/best_model.zip")
    logger.info(f"Loaded config from {cfg.experiment_dir}/.hydra/config.yaml")

    # Get env, wrappers and policy
    env = make_vec_env(
        lambda: FrameStackObservation(
            FlattenObservation(
                RayleighBenardEnv(config.test_env, render_mode="rgb_array"),
            ),
            stack_size=config.sb3.frame_stack,
        ),
        vec_env_cls=DummyVecEnv,
    )
    policy = PPO.load(model_path, env=env)
    policy.set_logger(logger)

    # Enjoy trained agent
    nusselts = []
    times = []
    episode = []
    for idx in range(cfg.nr_episodes):
        logger.info(f"Evaluating model on episode {idx}")
        # data holders
        screens = []
        actions = []
        # logging
        wandb.define_metric(f"ep{idx}/time")
        wandb.define_metric(f"ep{idx}/nusselt_state", step_metric=f"ep{idx}/time", summary="mean")
        wandb.define_metric(f"ep{idx}/nusselt_obs", step_metric=f"ep{idx}/time", summary="mean")
        wandb.define_metric(f"ep{idx}/reward", step_metric=f"ep{idx}/time", summary="mean")
        # reset env
        obs = env.reset()
        dones = np.zeros(1)
        while not dones.any():
            # get next action
            action, _ = policy.predict(obs, deterministic=True)
            # save data
            screens.append(env.render().transpose(2, 0, 1))
            actions.append(action.squeeze())
            # perform step in env
            obs, rewards, dones, info = env.step(action)

            # log data
            wandb.log(
                {
                    f"ep{idx}/time": info[0]["t"],
                    f"ep{idx}/nusselt_state": info[0]["nusselt"],
                    f"ep{idx}/nusselt_obs": info[0]["nusselt_obs"],
                    f"ep{idx}/reward": rewards[0],
                }
            )
            nusselts.append(info[0]["nusselt_obs"])
            times.append(info[0]["t"])
            episode.append(idx)
        # plot data
        plot_actions(actions, output_dir, idx)
        wandb.log(
            {
                f"ep{idx}/video": Video(np.asarray(screens), fps=2, format="mp4"),
            }
        )

    # log overall mean nusselt
    nusselt_mean = np.mean(nusselts)
    logger.info(f"Mean nusselt number: {nusselt_mean}")
    df = pd.DataFrame(
        {
            "nusselt": np.array(nusselts),
            "time": np.array(times),
            "episode": np.array(episode),
        }
    )
    wandb.log({"nusselt_table": Table(dataframe=df)})
    wandb.log({"mean_nusselt": nusselt_mean})
    wandb.run.summary["mean_nusselt"] = nusselt_mean


def plot_actions(actions, out_dir, episode_idx):
    # Plot nusselt number
    fig, ax = plt.subplots()
    # Plot amplitude
    ax.set_xlabel("segements")
    ax.set_ylabel("amplitude")
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis="y")
    ax.grid()
    # plot actions
    artists = []
    for action in actions:
        container = ax.plot(range(len(action)), action, color="blue")
        artists.append(container)

    ani = animation.ArtistAnimation(fig=fig, artists=artists)
    writer = animation.FFMpegWriter(fps=2)
    path = f"{out_dir}/actions_ep{episode_idx}.mp4"
    ani.save(path, writer=writer)
    wandb.log(
        {
            f"ep{episode_idx}/actions": Video(path, format="mp4"),
        }
    )


if __name__ == "__main__":
    main()
