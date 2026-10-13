"""Measure how long the desktop application takes to become usable.

Runs in a fresh interpreter each time so imports are not cached:

    python scripts/bench_startup.py            # three runs, offscreen
    python scripts/bench_startup.py --runs 5

Reported phases: Qt and splash, main window construction, first paint, and
the first engine calculation (which includes loading the Numba cache).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json, os, sys, tempfile, time
from pathlib import Path
t0 = time.perf_counter()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, sys.argv[1])
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from ensim.ui import theme
theme.load_fonts(); theme.apply(app, "console")
from ensim.ui.splash_screen import EnSimSplashScreen
EnSimSplashScreen().show(); app.processEvents()
t_splash = time.perf_counter()
from ensim.ui.settings import Settings
from ensim.ui.tasks import TaskRunner
from ensim.ui.windows.main_window import MainWindow, examples_dir
window = MainWindow(Settings(Path(tempfile.mkdtemp()) / "s.ini"), TaskRunner(synchronous=True))
t_window = time.perf_counter()
window.show(); app.processEvents()
t_paint = time.perf_counter()
window.open_path(examples_dir() / "kerolox_sounding_rocket.ensim", remember=False, confirm=False)
window.run_current(); app.processEvents()
t_engine = time.perf_counter()
print(json.dumps({"splash": t_splash - t0, "window": t_window - t_splash,
                  "paint": t_paint - t_window, "first_engine": t_engine - t_paint,
                  "total": t_engine - t0}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    rows = []
    for _ in range(args.runs):
        output = (
            subprocess.run(
                [sys.executable, "-c", PROBE, str(ROOT / "src")],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()[-1]
        )
        rows.append(json.loads(output))
    keys = list(rows[0])
    print("run  " + "  ".join(f"{k:>12s}" for k in keys))
    for i, row in enumerate(rows, 1):
        print(f"{i:3d}  " + "  ".join(f"{row[k] * 1000:10.0f} ms" for k in keys))


if __name__ == "__main__":
    main()
