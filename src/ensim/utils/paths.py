"""Per-user directories. Kept free of heavy imports: the splash screen uses it."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def user_cache_dir() -> Path:
    """Per-user cache directory for EnSim."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "EnSim" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "EnSim"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ensim"
