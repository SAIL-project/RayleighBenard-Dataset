import numpy as np

from rbcdata.utils.rbc_field import RBCField


def normalize_control(
    control: np.ndarray,
    limit: float,
):
    """Normalize control to be within the limits (Beintema et al 2020)"""
    control = np.clip(control, -limit, limit)
    control = control - np.mean(control)
    control = control / max(1, np.max(np.abs(control) / limit))
    return control


def segment_control(control: np.ndarray, segments: int):
    segments = np.array_split(
        control, segments
    )  # TODO how to split if segments is not a factor of array size
    return np.array([np.mean(seg) for seg in segments])


def err_optimal_conductive_state(state):
    """Beintema 2020"""
    pass


def err_midline_temperature(state, Th, Tc):
    """Singer and Bau 1917"""
    mid = int(state[RBCField.T].shape[0] / 2)
    T_mid = state[RBCField.T][mid]
    T_half = 1 / 2 * (Th + Tc)
    T_delta = Th - Tc

    err = (T_mid - T_half) / T_delta
    return err


def err_shadow_graph(state):
    """Howle 1997"""
    pass
