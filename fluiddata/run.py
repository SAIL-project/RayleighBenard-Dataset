import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    hydra.utils.call(cfg.env)


if __name__ == "__main__":
    main()
