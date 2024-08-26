from pprint import pprint

import ray
import rootutils
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.envs.rbc_ma_env import RayleighBenardMultiAgentEnv
from rbcdata.utils.ray_callbacks import LogCallback


def main() -> None:
    ray.init()

    register_env("rbc_ma_env", lambda config: RayleighBenardMultiAgentEnv(config))
    config = (
        PPOConfig()
        .training(
            train_batch_size=128,
        )
        .environment(
            env="rbc_ma_env",
            env_config={
                "episode_length": 10,
            },
        )
        .env_runners(
            num_env_runners=4,
            num_cpus_per_env_runner=2,
        )
        #  .resources(num_gpus=1)
        .multi_agent(
            policies={"p0"},
            # All agents map to the exact same policy.
            policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        )
        .callbacks(LogCallback)
    )

    algo = config.build()
    result = algo.train()
    pprint(result)


if __name__ == "__main__":
    main()
