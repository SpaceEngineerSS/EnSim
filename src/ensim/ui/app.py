"""Application start-up.

Only Qt and the splash screen are imported before the first paint; the main
window and the numerical modules load behind the splash.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def run(argv: list[str] | None = None) -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from ensim import __version__

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("EnSim")
    app.setOrganizationName("EnSim")
    app.setApplicationVersion(__version__)
    icon_path = Path(__file__).resolve().parents[1] / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    from ensim.ui.splash_screen import EnSimSplashScreen

    splash = EnSimSplashScreen()
    splash.show()
    app.processEvents()

    splash.showMessage("Loading models")
    from ensim.ui.windows.main_window import MainWindow

    window = MainWindow()
    window.show()
    splash.finish(window)
    return app.exec()
