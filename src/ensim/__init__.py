"""EnSim rocket propulsion and flight simulation package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ensim")
except PackageNotFoundError:
    __version__ = "3.0.1.dev0"

__all__ = ["__version__"]
