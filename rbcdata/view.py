import argparse
import pathlib
import time

import h5py

from rbcdata.utils.rbc_field import RBCField
from rbcdata.vis import RBCFieldVisualizer


def view_dataset(path: pathlib.Path) -> None:
    file = h5py.File(path, "r")
    states = file["states"]
    vis_state = RBCFieldVisualizer(
        size=file.attrs["N"],
        vmin=file.attrs["bcT"][1],
        vmax=file.attrs["bcT"][0] + file.attrs["action_limit"],
    )

    # Loop
    for idx in range(file.attrs["steps"]):
        # check if vis closed
        if vis_state.closed:
            print("Visualizer closed")
            exit()

        # get fields
        T = states[idx][RBCField.T]
        ux = states[idx][RBCField.UX]
        uy = states[idx][RBCField.UY]

        # get time step
        t = idx * file.attrs["dt"]
        # visualize
        vis_state.draw(T, ux, uy, t)
        time.sleep(1 / 50)


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
