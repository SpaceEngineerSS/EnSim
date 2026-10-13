"""Numba integration.

Compiled kernels are cached on disk so they are built once per machine, not
once per launch. A packaged (frozen) application cannot write next to its own
modules, so its cache goes to the per-user cache directory instead; Numba's
user-wide locator supports frozen executables.

Without Numba the decorated functions run as plain Python: correct, but much
slower for the flight and equilibrium loops.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from .paths import user_cache_dir

if getattr(sys, "frozen", False) and "NUMBA_CACHE_DIR" not in os.environ:
    # Must be set before numba is imported; numba reads it once.
    os.environ["NUMBA_CACHE_DIR"] = str(user_cache_dir() / "numba")

try:
    from numba import jit as numba_jit
except ImportError:  # pragma: no cover - exercised only without numba installed
    numba_jit = None

HAVE_NUMBA = numba_jit is not None


def jit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """``numba.jit`` when available, otherwise an identity decorator.

    Works both bare (``@jit``) and with options (``@jit(nopython=True)``).
    """
    if numba_jit is not None:
        return numba_jit(*args, **kwargs)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda function: function
