from abc import ABC
from typing import Any


class Controller(ABC):
    def __init__(
        self,
        start: float,
        end: float,
        zero_control: Any,
    ) -> None:
        # Params
        self.start = start
        self.end = end
        self.last = -10
        self.zero = zero_control
        self.control = zero_control

    def __call__(self, env, obs, info) -> bool:
        # check if the controller should apply a new action
        if info["t"] < self.start:
            return False
        elif info["t"] > self.end:
            self.control = self.zero
            return False
        return True
