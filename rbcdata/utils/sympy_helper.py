from typing import Tuple

import sympy


def evalf_tuple(t):
    if isinstance(t, Tuple):
        return tuple(evalf_tuple(x) for x in t)
    elif isinstance(t, sympy.Expr):
        return float(t.evalf())
    else:
        return t
