import argparse
import time

import hydra
from omegaconf import DictConfig

from rbcdata import RBCDataset
from rbcdata.utils import RBCField
from rbcdata.vis import RBCConvectionVisualizer, RBCFieldVisualizer


@hydra.main(version_base=None, config_path="../config", config_name="view")
def view_dataset(cfg: DictConfig) -> None:
    # Data
    dataset = RBCDataset(cfg.path, cfg.dataset)

    # Visualize
    vis_state = RBCFieldVisualizer(
        N=dataset.parameters.N,
        spatial_mesh=dataset.parameters.spatial_mesh,
        show_u=True,
        vmin=1,
        vmax=2,
        block=False,
    )
    vis_conv = RBCConvectionVisualizer(
        vmin=-0.1,
        vmax=0.2,
        show=True,
    )

    # Loop
    for idx in range(len(dataset)):
        # check if vis closed
        if vis_conv.closed:
            print("Visualizer closed")
            exit()

        # get fields
        T = dataset[idx][0][RBCField.T].numpy()
        ux = dataset[idx][0][RBCField.UX].numpy()
        uy = dataset[idx][0][RBCField.UY].numpy()
        conv = dataset[idx][0][RBCField.JCONV].numpy()
        # get time step
        t = idx * dataset.cfg.dt
        # visualize
        vis_state.draw(T, ux, uy, t)
        vis_conv.draw(conv, t)

        time.sleep(1 / cfg.fps)


if __name__ == "__main__":
    view_dataset()
