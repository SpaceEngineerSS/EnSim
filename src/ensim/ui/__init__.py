"""Desktop application (PyQt6). Install with ``pip install "ensim[gui]"``."""

import importlib.util

if importlib.util.find_spec("PyQt6") is None:
    raise ImportError('The EnSim desktop application needs PyQt6: pip install "ensim[gui]"')
