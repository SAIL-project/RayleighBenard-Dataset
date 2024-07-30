from abc import ABC
from typing import Any


class Controller(ABC):
    def __init__(
        self,
        start: float,
        duration: float,
        zero: Any,
    ) -> None:
        self.time = start
        self.duration = duration
        self.last = -10
        self.control = zero

    def __call__(self, env, obs, info) -> bool:
        if info["t"] < self.time:
            return False
        elif info["t"] - self.last > self.duration:
            self.last = info["t"]
            return True
        return False
