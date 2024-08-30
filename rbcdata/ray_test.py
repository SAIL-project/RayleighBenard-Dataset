import logging
from pprint import pprint

import ray
import rootutils
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.envs.rbc_ma_env import RayleighBenardMultiAgentEnv
from rbcdata.utils.ray_callbacks import LogCallback

ITERATIONS = 50
EPISODE_LENGTH = 50
ENV_RUNNERS = 4


def main() -> None:
    ray.init()
    logger = logging.getLogger("ray")

    register_env("rbc_ma_env", lambda config: RayleighBenardMultiAgentEnv(config))
    algo = (
        PPOConfig()
        .environment(
            env="rbc_ma_env",
            env_config={
                "checkpoint": "data/checkpoints/ra10000/train/baseline42",
                "episode_length": EPISODE_LENGTH,
                "ra": 10_000,
            },
        )
        .env_runners(
            num_env_runners=ENV_RUNNERS,
            num_envs_per_env_runner=1,
            rollout_fragment_length=EPISODE_LENGTH,
            batch_mode="complete_episodes",
            sample_timeout_s=2000,
        )
        .multi_agent(
            policies={"p0"},
            # All agents map to the exact same policy.
            policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        )
        .training(
            train_batch_size=EPISODE_LENGTH * ENV_RUNNERS,
            sgd_minibatch_size=10,
        )
        .evaluation(
            evaluation_duration=1,
            evaluation_interval=ITERATIONS,
        )
        .framework("torch")
        .callbacks(LogCallback)
    ).build()

    # Train the policy
    for idx in range(ITERATIONS):
        logger.info(f"Start Iteration {idx}...")
        algo.train()
    logger.info("Finished Training")
    pprint(eval)


if __name__ == "__main__":
    main()
