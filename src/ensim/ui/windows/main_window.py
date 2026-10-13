"""The main window: navigation, sections, menus and the project lifecycle."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ensim import __version__
from ensim.core.project_manager import ProjectFileError
from ensim.utils.units import Quantity, UnitSystem

from .. import motion, theme
from ..sections.analysis import AnalysisSection
from ..sections.engine import EngineSection
from ..sections.flight import FlightSection
from ..sections.log import LogSection
from ..sections.vehicle import VehicleSection
from ..settings import Settings
from ..shell import Header, NavRail, StatusStrip
from ..tasks import TaskRunner
from ..units import UNITS, fmt
from ..workspace import Workspace

PROJECT_FILTER = "EnSim project (*.ensim);;All files (*)"
DOCS_URL = "https://spaceengineerss.github.io/EnSim/"


def examples_dir() -> Path | None:
    """Example projects shipped inside the package."""
    folder = Path(__file__).resolve().parents[2] / "examples"
    return folder if folder.is_dir() else None


class MainWindow(QMainWindow):
    SECTIONS = (EngineSection, VehicleSection, FlightSection, AnalysisSection, LogSection)

    def __init__(self, settings: Settings | None = None, tasks: TaskRunner | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.tasks = tasks or TaskRunner(self)
        self.workspace = Workspace(self.tasks)
        UNITS.set_system(self.settings.units)
        motion.set_enabled(not self.settings.reduce_motion)

        self.setWindowTitle("EnSim")
        self.setMinimumSize(1024, 640)
        self.resize(1360, 820)

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.sections = [cls(self.workspace, self.settings) for cls in self.SECTIONS]
        self.log = self.sections[-1]
        self.nav = NavRail([(s.title, s.icon_name) for s in self.sections])
        outer.addWidget(self.nav)
        column = QVBoxLayout()
        column.setSpacing(0)
        self.header = Header()
        column.addWidget(self.header)
        self.stack = QStackedWidget()
        for section in self.sections:
            self.stack.addWidget(section)
        column.addWidget(self.stack, 1)
        self.status = StatusStrip()
        column.addWidget(self.status)
        outer.addLayout(column, 1)
        self.setCentralWidget(central)

        self._build_menus()
        self._wire()
        self.show_section(0)
        self._update_title()
        geometry = self.settings.geometry("main")
        if geometry is not None:
            self.restoreGeometry(geometry)

    # wiring

    def _wire(self) -> None:
        ws = self.workspace
        self.nav.selected.connect(self.show_section)
        self.status.cancel_requested.connect(self.cancel)
        self.tasks.busy_changed.connect(self.status.set_busy)
        self.tasks.activity.connect(
            lambda _name, fraction, text: self.status.set_progress(fraction, text)
        )
        ws.message.connect(self.notify)
        ws.modified_changed.connect(lambda _m: self._update_title())
        ws.project_loaded.connect(self._project_loaded)
        ws.engine_started.connect(lambda: self.header.set_state("Running…"))
        ws.engine_result.connect(self._engine_done)
        ws.engine_failed.connect(self._engine_failed)

        QShortcut(QKeySequence("F5"), self, activated=self.run_current)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.run_current)
        QShortcut(QKeySequence("Escape"), self, activated=self.cancel)
        for index in range(len(self.sections)):
            QShortcut(
                QKeySequence(f"Ctrl+{index + 1}"),
                self,
                activated=lambda i=index: self.show_section(i),
            )

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self._action("&New", "Ctrl+N", self.new_project))
        file_menu.addAction(self._action("&Open…", "Ctrl+O", self.open_project))
        self.examples_menu = file_menu.addMenu("Open &example")
        self.recent_menu = file_menu.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        self._fill_examples()
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Save", "Ctrl+S", self.save_project))
        file_menu.addAction(self._action("Save &as…", "Ctrl+Shift+S", self.save_project_as))
        file_menu.addSeparator()
        file_menu.addAction(self._action("Export &report…", "Ctrl+E", self.export_report))
        file_menu.addAction(self._action("Export results as &JSON…", None, self.export_json))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Quit", "Ctrl+Q", self.close))

        view = bar.addMenu("&View")
        themes = QActionGroup(self)
        for name, label in (("console", "Dark"), ("paper", "Light")):
            action = self._action(label, None, lambda _checked=False, n=name: self.set_theme(n))
            action.setCheckable(True)
            action.setChecked(self.settings.theme == name)
            themes.addAction(action)
            view.addAction(action)
        view.addSeparator()
        units = QActionGroup(self)
        for system, label in (
            (UnitSystem.SI, "SI units"),
            (UnitSystem.IMPERIAL, "US customary units"),
        ):
            action = self._action(label, None, lambda _checked=False, s=system: self.set_units(s))
            action.setCheckable(True)
            action.setChecked(self.settings.units == system)
            units.addAction(action)
            view.addAction(action)
        view.addSeparator()
        motion_action = self._action("Reduce motion", None, self.toggle_motion)
        motion_action.setCheckable(True)
        motion_action.setChecked(self.settings.reduce_motion)
        view.addAction(motion_action)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction(
            self._action("&Documentation", "F1", lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        )
        help_menu.addAction(
            self._action(
                "Model &limitations",
                None,
                lambda: QDesktopServices.openUrl(QUrl(DOCS_URL + "model-limitations/")),
            )
        )
        help_menu.addSeparator()
        help_menu.addAction(self._action("&About EnSim", None, self.about))

    def _action(self, text: str, shortcut: str | None, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        return action

    def _fill_examples(self) -> None:
        self.examples_menu.clear()
        folder = examples_dir()
        files = sorted(folder.glob("*.ensim")) if folder else []
        for path in files:
            label = path.stem.replace("_", " ").capitalize()
            self.examples_menu.addAction(label, lambda p=path: self.open_path(p, remember=False))
        self.examples_menu.setEnabled(bool(files))

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self.settings.recent_files if p.exists()]
        for path in recent:
            self.recent_menu.addAction(str(path), lambda p=path: self.open_path(p))
        if not recent:
            empty = self.recent_menu.addAction("No recent projects")
            empty.setEnabled(False)

    # navigation

    def show_section(self, index: int) -> None:
        section = self.sections[index]
        section.ensure_built()
        if self.stack.currentWidget() is not section:
            self.stack.setCurrentWidget(section)
            motion.fade_in(section)
        self.nav.set_current(index)

    def current_section(self):
        return self.stack.currentWidget()

    def run_current(self) -> None:
        section = self.current_section()
        if section.run_label:
            section.run()
        else:
            self.notify(f"{section.title} has no single calculation; use the buttons on the page")

    def cancel(self) -> None:
        if self.tasks.busy:
            self.tasks.cancel()
            self.notify("Cancelling…")

    # feedback

    def notify(self, text: str) -> None:
        self.status.show_message(text)
        self.log.add(text)

    def _engine_done(self, result) -> None:
        self._show_engine_summary(result)
        r = result
        elapsed = self.workspace.engine_elapsed
        self.header.set_state(f"Converged in {elapsed * 1000:.0f} ms", "Ok")
        self.notify(
            f"Engine: Tc {r.chamber_temperature:.1f} K, Isp vac {r.vacuum.isp:.2f} s, "
            f"thrust vac {r.vacuum.thrust / 1000:.3f} kN ({elapsed * 1000:.0f} ms)"
        )

    def _show_engine_summary(self, result) -> None:
        r = result
        self.header.set_summary(
            f"{r.inputs.fuel}/{r.inputs.oxidizer}   O/F {r.inputs.of_ratio:g}   "
            f"pc {fmt(UNITS.show(r.inputs.chamber_pressure, Quantity.PRESSURE))} "
            f"{UNITS.unit(Quantity.PRESSURE)}   Isp vac {r.vacuum.isp:.1f} s"
        )

    def _engine_failed(self, message: str) -> None:
        self.header.set_state("Engine failed", "Error")
        self.notify(f"Engine failed: {message}")
        self.sections[0].set_running(False)

    # appearance

    def set_theme(self, name: str) -> None:
        from PyQt6.QtWidgets import QApplication

        self.settings.theme = name
        theme.apply(QApplication.instance(), name)
        self.nav.restyle()
        self.status.restyle()
        for section in self.sections:
            section.restyle()

    def set_units(self, system: UnitSystem) -> None:
        self.settings.units = system
        UNITS.set_system(system)
        for section in self.sections:
            section.restyle()
        if self.workspace.result is not None:
            self._show_engine_summary(self.workspace.result)
            self.sections[0]._show_result(self.workspace.result)

    def toggle_motion(self, checked: bool) -> None:
        self.settings.reduce_motion = checked
        motion.set_enabled(not checked)

    # project lifecycle

    def _update_title(self) -> None:
        name = self.workspace.project.project_name
        modified = self.workspace.project.is_modified
        self.setWindowTitle(f"{name}{' *' if modified else ''} – EnSim")
        self.header.set_project(name, modified)

    def _project_loaded(self) -> None:
        self._update_title()
        self.header.set_summary("")
        self.header.set_state("")
        for section in self.sections:
            if section.built:
                section.load_from_project()

    def _confirm_discard(self) -> bool:
        if not self.workspace.project.is_modified:
            return True
        buttons = (
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save changes to {self.workspace.project.project_name}?",
            buttons,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def new_project(self) -> None:
        if self._confirm_discard():
            self.workspace.new()
            self.notify("New project")

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if path:
            self.open_path(Path(path), confirm=False)

    def open_path(self, path: Path, remember: bool = True, confirm: bool = True) -> bool:
        if confirm and not self._confirm_discard():
            return False
        try:
            self.workspace.load(path)
        except ProjectFileError as error:
            QMessageBox.warning(self, "Open project", str(error))
            return False
        if remember:
            self.settings.add_recent(path)
        self.notify(f"Opened {path.name}")
        return True

    def save_project(self) -> bool:
        if self.workspace.project.current_path is None:
            return self.save_project_as()
        return self._save(None)

    def save_project_as(self) -> bool:
        suggestion = f"{self.workspace.project.project_name}.ensim"
        path, _ = QFileDialog.getSaveFileName(self, "Save project", suggestion, PROJECT_FILTER)
        return bool(path) and self._save(Path(path))

    def _save(self, path: Path | None) -> bool:
        try:
            written = self.workspace.save(path)
        except ProjectFileError as error:
            QMessageBox.warning(self, "Save project", str(error))
            return False
        self.settings.add_recent(written)
        self._update_title()
        self.notify(f"Saved {written.name}")
        return True

    def export_report(self) -> None:
        if self.workspace.result is None:
            self.notify("Run the engine before exporting a report")
            return
        suggestion = f"{self.workspace.project.project_name}.md"
        path, _ = QFileDialog.getSaveFileName(self, "Export report", suggestion, "Markdown (*.md)")
        if path:
            from ..report import write_report

            write_report(self.workspace, Path(path))
            self.notify(f"Report written to {Path(path).name}")

    def export_json(self) -> None:
        if self.workspace.result is None:
            self.notify("Run the engine before exporting results")
            return
        suggestion = f"{self.workspace.project.project_name}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export results", suggestion, "JSON (*.json)")
        if path:
            from ..report import write_json

            write_json(self.workspace, Path(path))
            self.notify(f"Results written to {Path(path).name}")

    def about(self) -> None:
        QMessageBox.about(
            self,
            "About EnSim",
            f"<p><b>EnSim {__version__}</b></p>"
            "<p>Liquid rocket engine analysis and six-degree-of-freedom flight simulation.</p>"
            "<p>Chamber equilibrium by Gibbs minimisation with NASA Glenn thermodynamic data, "
            "frozen nozzle expansion, Barrowman aerodynamics and a rigid-body flight model.</p>"
            "<p>Mehmet Gümüş · MIT licence · "
            "<a href='https://github.com/SpaceEngineerSS/EnSim'>github.com/SpaceEngineerSS/EnSim</a></p>",
        )

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self.settings.set_geometry("main", self.saveGeometry())
        if not self.tasks.shutdown(3000):
            self.notify("Some calculations were still running at exit")
        self.log.detach()
        event.accept()
