import argparse
import pathlib
import time

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from fluiddata.dataset.cylinder_dataset import CylinderDataset, CylinderField
from fluiddata.utils import CylinderVisualizer


def view_dataset(path: pathlib.Path) -> None:
    # Data
    dataset = CylinderDataset(path, sequence_length=1)

    # Visualize
    vort_conv = CylinderVisualizer(vrange=(-5, 5), title="Vorticity")
    magn_vis = CylinderVisualizer(vrange=(0, 1.5), title="Magnitude")

    # Loop
    for idx in range(len(dataset)):
        # check if vis closed
        if vort_conv.closed or magn_vis.closed:
            exit()

        # show field
        state = dataset[idx][0]
        vort_conv.draw(state[CylinderField.VORT], t=idx)
        magn_vis.draw(state[CylinderField.MAGN], t=idx)

        time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the dataset")
    args = parser.parse_args()

    view_dataset(pathlib.Path(args.filename))
