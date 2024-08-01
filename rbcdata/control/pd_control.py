from typing import Any, List

import numpy as np

from rbcdata.control.controller import Controller
from rbcdata.control.utils import err_midline_temperature, segment_control


class PDController(Controller):
    def __init__(
        self,
        kp: float,
        kd: float,
        bcT: List[float],
        limit: float,
        start: float,
        duration: float,
        zero: Any,
    ) -> None:
        super().__init__(start, duration, zero)
        self.kp = kp
        self.kd = kd
        self.bcT = bcT
        self.limit = limit

    def __call__(self, env, obs, info) -> float:
        if super().__call__(env, obs, info):
            err = err_midline_temperature(env.get_state(), self.bcT[0], self.bcT[1])
            control = -self.kp * err  # TODO Derivative term
            control = segment_control(control, env.action_segments)
            self.control = np.clip(control, -1, 1)

        return self.control
