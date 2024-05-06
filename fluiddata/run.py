import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from fluiddata.simulation.rbc.run import run_rbc


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    # hydra.utils.call(cfg.env)
    run_rbc(cfg.env)


if __name__ == "__main__":
    main()
