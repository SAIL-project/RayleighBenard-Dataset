import matplotlib


def coolwarm_colormap(value, vmin=1, vmax=2):
    cmap = matplotlib.colormaps["coolwarm"]
    value = (value - vmin) / (vmax - vmin)
    return cmap(value, bytes=True)[:, :, :3]
