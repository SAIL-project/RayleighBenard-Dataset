import numpy as np


def normalize_control(
    control: np.ndarray,
    limit: float,
):
    """Normalize control to be within the limits (Beintema et al 2020)"""
    control = np.clip(control, -limit, limit)
    control = control - np.mean(control)
    control = control / max(1, np.max(np.abs(control) / limit))
    return control


def segmentize(input: np.ndarray, segments: int):
    segments = np.array_split(
        input, segments
    )  # TODO how to split if segments is not a factor of array size
    return np.array([np.mean(seg) for seg in segments])
