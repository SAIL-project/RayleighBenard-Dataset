from abc import ABC
from typing import Any


class Controller(ABC):
    def __init__(
        self,
        start: float,
        end: float,
        duration: float,
        zero: Any,
    ) -> None:
        # Params
        self.start = start
        self.end = end
        self.last = -10
        self.duration = duration
        self.zero = zero
        self.control = zero

    def __call__(self, env, obs, info) -> bool:
        # check if the controller should apply a new action
        if info["t"] < self.start:
            return False
        elif info["t"] > self.end:
            self.control = self.zero
            return False
        elif info["t"] - self.last > self.duration:
            self.last = info["t"]
            return True
        return False
