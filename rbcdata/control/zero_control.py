from typing import Any

from rbcdata.control.controller import Controller


class ZeroController(Controller):
    def __init__(
        self,
        start_time: float,
        control_duration: float,
        zero_control: Any,
    ) -> None:
        super().__init__(start_time, control_duration, zero_control)

    def __call__(self, env, obs, info) -> float:
        return self.control
