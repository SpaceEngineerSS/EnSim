"""Interface tests need PyQt6. Without it (a core-only install) they are not collected."""

import importlib.util
import os

import pytest

if importlib.util.find_spec("PyQt6") is None:
    collect_ignore_glob = ["test_*.py"]
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["ENSIM_DISABLE_3D"] = "1"


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/ui" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.gui)


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    from ensim.ui import theme
    from ensim.ui.app import _use_c_locale

    app = QApplication.instance() or QApplication([])
    _use_c_locale()
    theme.load_fonts()
    theme.apply(app, "console")
    return app


@pytest.fixture
def window(qapp, tmp_path):
    """Main window with private settings and tasks that run inline."""
    from ensim.ui.settings import Settings
    from ensim.ui.tasks import TaskRunner
    from ensim.ui.windows.main_window import MainWindow

    win = MainWindow(Settings(tmp_path / "settings.ini"), TaskRunner(synchronous=True))
    win.resize(1366, 768)
    win.show()
    qapp.processEvents()
    yield win
    win.workspace.project._modified = False
    win.close()


@pytest.fixture
def example(window):
    from ensim.ui.windows.main_window import examples_dir

    assert window.open_path(
        examples_dir() / "kerolox_sounding_rocket.ensim", remember=False, confirm=False
    )
    return window
