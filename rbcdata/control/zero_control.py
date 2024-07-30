from typing import Any

from rbcdata.control.controller import Controller


class ZeroController(Controller):
    def __init__(
        self,
        start: float,
        duration: float,
        zero: Any,
    ) -> None:
        super().__init__(start, duration, zero)

    def __call__(self, env, obs, info) -> float:
        return self.control
