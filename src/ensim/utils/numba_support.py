from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from numba import jit as numba_jit


def jit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    if getattr(sys, "frozen", False):
        kwargs["cache"] = False
    return numba_jit(*args, **kwargs)
