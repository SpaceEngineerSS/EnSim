"""Headless construction checks for the desktop application shell."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ensim.ui.windows.main_window import MainWindow


def test_main_window_constructs_and_contextual_input_panel_switches():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.tabs.count() == 5
        assert window.engine_tabs.count() == 2
        assert window.engine_tabs.tabText(0) == "Cooling"
        assert window.engine_tabs.tabText(1) == "Optimize"
        assert window.input_panel.fuel_combo.currentData() == "H2"
        assert window.input_panel.oxidizer_combo.currentData() == "O2"
        assert window.input_scroll_area.isHidden() is False

        window.tabs.setCurrentIndex(2)
        assert window.input_scroll_area.isHidden() is True
        window.tabs.setCurrentIndex(1)
        assert window.input_scroll_area.isHidden() is False

        assert not hasattr(window, "replay_bar")
        assert not hasattr(window, "unit_bar")
    finally:
        window.close()
        app.processEvents()
