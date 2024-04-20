import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from rbcdata.simulation.cylinder.generate import generate_cylinder


@hydra.main(version_base=None, config_path="config", config_name="run_cylinder")
def main(cfg: DictConfig) -> None:
    generate_cylinder(cfg.sim)


if __name__ == "__main__":
    main()
