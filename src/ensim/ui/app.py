"""Application start-up.

Only Qt and the splash card are imported before the first paint. The main
window comes next, and the compiled numerical kernels are warmed on a worker
thread once the window is visible, so the first calculation does not stall.
"""

from __future__ import annotations

import logging
import os
import sys


def _warm_up(_ctx) -> None:
    """Compile (or load from cache) the kernels behind a first engine and flight run."""
    from ensim.core.engine import EngineInputs, analyze_engine
    from ensim.core.flight_6dof import simulate_flight_6dof
    from ensim.core.rocket import create_default_rocket

    analyze_engine(EngineInputs("RP-1", "LOX", 2.3, 20e5, 4.0, throat_area=1.4e-4))
    simulate_flight_6dof(create_default_rocket(), 400.0, 250.0, 2.0, max_time=1.0, dt=0.05)


def _use_c_locale() -> None:
    """Point decimals and no digit grouping, whatever the system language.

    A Turkish or German system locale would otherwise show 0,3937 in spin boxes
    and make 1,476 ambiguous.
    """
    from PyQt6.QtCore import QLocale

    locale = QLocale.c()
    locale.setNumberOptions(
        QLocale.NumberOption.OmitGroupSeparator | QLocale.NumberOption.RejectGroupSeparator
    )
    QLocale.setDefault(locale)


def run(argv: list[str] | None = None) -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")
    logging.basicConfig(level=logging.WARNING)

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from ensim import __version__

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv if argv is not None else sys.argv)
    _use_c_locale()
    app.setApplicationName("EnSim")
    app.setOrganizationName("EnSim")
    app.setApplicationVersion(__version__)

    from . import theme
    from .icons import pixmap
    from .settings import Settings

    settings = Settings()
    theme.load_fonts()
    theme.apply(app, settings.theme)
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(pixmap("logo", theme.current().signal, size, 1.0))
    app.setWindowIcon(icon)

    from .splash_screen import EnSimSplashScreen

    splash = EnSimSplashScreen()
    splash.show()
    splash.showMessage("Loading interface")

    from .windows.main_window import MainWindow

    splash.showMessage("Loading thermodynamic data")
    window = MainWindow(settings)
    splash.showMessage("Opening")
    window.show()
    splash.finish(window)
    window.tasks.submit("warm-up", _warm_up, on_result=lambda _r: window.notify("Ready"))

    if settings.first_run:
        settings.mark_seen()
        from .windows.main_window import examples_dir

        folder = examples_dir()
        example = folder / "kerolox_sounding_rocket.ensim" if folder else None
        if example is not None and example.exists():
            window.open_path(example, remember=False, confirm=False)
            window.notify("Opened an example project. F5 runs the current page.")
    return app.exec()
