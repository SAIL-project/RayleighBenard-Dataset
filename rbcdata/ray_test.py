import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.envs.rbc_env import RayleighBenardEnv
from rbcdata.envs.rbc_ma_env import RayleighBenardMultiAgentEnv


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    tmp = RayleighBenardEnv(sim_cfg=cfg.sim)
    env = RayleighBenardMultiAgentEnv(env=tmp)

    env.reset(seed=42)

    action_dict = {str(idx): 0 for idx in range(len(env.get_agent_ids()))}
    env.step(action_dict)


if __name__ == "__main__":
    main()
