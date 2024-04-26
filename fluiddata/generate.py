import time

import hydra
import rootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


@hydra.main(version_base=None, config_path="config", config_name="generate")
def main(cfg: DictConfig) -> None:
    num = HydraConfig.get().job.num
    time.sleep(num / 10)
    pbar = tqdm(total=cfg.count, desc=f"Generating Dataset {num}", position=2 * num, leave=False)
    for i in range(cfg.count):
        hydra.utils.call(cfg.generate, seed=cfg.base_seed + i, num=num)
        pbar.update(1)


if __name__ == "__main__":
    main()
