from enum import Enum, IntEnum


class RBCField(IntEnum):
    UY = 0
    UX = 1
    T = 2
    JCONV = 3


class RBCType(Enum):
    NORMAL = "normal"
    CONVECTION = "convection"
    FULL = "full"
