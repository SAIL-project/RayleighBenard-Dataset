import matplotlib


def colormap(value, vmin=1, vmax=2, colormap="turbo"):
    cmap = matplotlib.colormaps[colormap]
    value = (value - vmin) / (vmax - vmin)
    return cmap(value, bytes=True)[:, :, :3]
