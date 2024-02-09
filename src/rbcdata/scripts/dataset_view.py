import time

import hydra
from omegaconf import DictConfig

from rbcdata.rbc_dataset import RBCDataset
from rbcdata.utils.rbc_field import RBCField
from rbcdata.vis import RBCConvectionVisualizer, RBCFieldVisualizer


@hydra.main(version_base=None, config_path="../config", config_name="view")
def view_dataset(cfg: DictConfig) -> None:
    # Data
    dataset = RBCDataset(cfg.path, cfg.dataset)

    # Visualize
    vis_conv = RBCConvectionVisualizer(
        size=dataset.parameters["N"],
        vmin=-0.1,
        vmax=0.2,
        show=True,
    )
    vis_state = RBCFieldVisualizer(
        size=dataset.parameters["N"],
        vmin=1,
        vmax=2,
        show=True,
        show_u=True,
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
        vis_conv.draw(conv, t)
        vis_state.draw(T, ux, uy, t)

        time.sleep(1 / cfg.fps)


if __name__ == "__main__":
    view_dataset()
