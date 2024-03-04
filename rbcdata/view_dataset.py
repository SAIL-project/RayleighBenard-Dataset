import argparse
import pathlib
import time

from rbcdata.config.dataclass.rbc_dataset_config import RBCDatasetConfig
from rbcdata.utils.rbc_dataset import RBCDataset
from rbcdata.utils.rbc_field import RBCField
from rbcdata.utils.rbc_type import RBCType
from rbcdata.vis import RBCConvectionVisualizer, RBCFieldVisualizer


def view_dataset(path: pathlib.Path) -> None:
    # Data
    config = RBCDatasetConfig(
        sequence_length=1,
        type=RBCType.FULL,
        dt=1,
        start_idx=0,
        end_idx=500,
    )
    dataset = RBCDataset(path, config)

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

        time.sleep(1 / 10)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rbcview",
        description="View RayleighBenardConvection dataset",
    )
    parser.add_argument("filename", help="Path to the dataset")
    args = parser.parse_args()

    view_dataset(pathlib.Path(args.filename))


if __name__ == "__main__":
    main()
