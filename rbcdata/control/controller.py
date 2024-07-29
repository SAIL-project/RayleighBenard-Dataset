from abc import ABC
from typing import Any


class Controller(ABC):
    def __init__(self, start_time: float, control_duration: float, zero_control: Any) -> None:
        self.start_time = start_time
        self.control_duration = control_duration
        self.last_control = -10
        self.control = zero_control

    def __call__(self, env, obs, info) -> bool:
        if info["t"] < self.start_time:
            return False
        elif info["t"] - self.last_control > self.control_duration:
            self.last_control = info["t"]
            return True
        return False
