from typing import Any

from rbcdata.control.controller import Controller


class ZeroController(Controller):
    def __init__(
        self,
        zero: Any,
    ) -> None:
        super().__init__(start=0, end=0, duration=1, zero=zero)

    def __call__(self, env, obs, info) -> float:
        return self.control
