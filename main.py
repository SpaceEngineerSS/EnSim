"""Launcher for a source checkout and for the PyInstaller bundles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ensim.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
