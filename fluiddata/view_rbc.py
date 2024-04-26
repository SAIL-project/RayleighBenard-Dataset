import argparse
import pathlib
import time

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from fluiddata.dataset.rbc_dataset import RBCDataset, RBCField, RBCType
from fluiddata.utils import ConvectionVisualizer, RBCFieldVisualizer


def view_dataset(path: pathlib.Path) -> None:
    # Data
    dataset = RBCDataset(path, type=RBCType.FULL)

    # Visualize
    conv_vis = ConvectionVisualizer()
    state_vis = RBCFieldVisualizer()

    # Loop
    for idx in range(len(dataset)):
        # check if vis closed
        if conv_vis.closed or state_vis.closed:
            exit()
        # Get fields
        state = dataset[idx][0]
        T = state[RBCField.T].numpy()
        ux = state[RBCField.UX].numpy()
        uy = state[RBCField.UY].numpy()
        conv = state[RBCField.JCONV].numpy()
        # get time step
        t = idx * dataset.parameters["dt"]
        # visualize
        conv_vis.draw(conv, t)
        state_vis.draw(T, ux, uy, t)

        time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the dataset")
    args = parser.parse_args()

    view_dataset(pathlib.Path(args.filename))
