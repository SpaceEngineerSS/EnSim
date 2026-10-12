"""State shared by the sections: the open project and the latest results."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ensim.core.engine import EngineInputs, EngineResult, analyze_engine
from ensim.core.frozen_flow import frozen_profile
from ensim.core.geometry import ChamberContour, chamber_contour
from ensim.core.project_manager import ProjectData, ProjectManager
from ensim.core.rocket import Rocket
from ensim.utils.nasa_parser import load_default_database

from .tasks import TaskRunner


@dataclass(frozen=True)
class EngineRun:
    """Everything the Engine section shows for one operating point."""

    result: EngineResult
    contour: ChamberContour
    profile: dict[str, np.ndarray]  # frozen-flow state at every contour station
    elapsed: float  # s


def compute_engine(
    inputs: EngineInputs, contraction_ratio: float, characteristic_length: float, ctx=None
) -> EngineRun:
    """Equilibrium, nozzle, contour and wall profile; runs on a worker thread."""
    start = time.perf_counter()
    result = analyze_engine(inputs)
    if ctx is not None:
        ctx.check()
    contour = chamber_contour(
        inputs.throat_area,
        inputs.expansion_ratio,
        contraction_ratio=contraction_ratio,
        characteristic_length=characteristic_length,
        n_points=80,
    )
    profile = frozen_profile(
        result.equilibrium, load_default_database(), contour.area_ratio, contour.supersonic
    )
    return EngineRun(result, contour, profile, time.perf_counter() - start)


class Workspace(QObject):
    """One open project plus everything computed from it.

    Sections read inputs from :attr:`data`, write them back with
    :meth:`update`, and listen to the signals to refresh.
    """

    project_loaded = pyqtSignal()
    modified_changed = pyqtSignal(bool)
    engine_started = pyqtSignal()
    engine_result = pyqtSignal(object)  # EngineResult
    engine_failed = pyqtSignal(str)
    rocket_changed = pyqtSignal()
    message = pyqtSignal(str)  # short status line text

    def __init__(self, tasks: TaskRunner):
        super().__init__()
        self.tasks = tasks
        self.project = ProjectManager()
        self.project.new_project()
        self.result: EngineResult | None = None
        self.run: EngineRun | None = None
        self.results: dict[str, Any] = {}  # latest result per analysis, for reports
        self.engine_elapsed = 0.0

    @property
    def data(self) -> ProjectData:
        return self.project.data

    # project

    def new(self) -> None:
        self.project.new_project()
        self._reset()

    def load(self, path: Path) -> None:
        self.project.load(path)
        self._reset()

    def save(self, path: Path | None = None) -> Path:
        written = self.project.save(path)
        self.modified_changed.emit(False)
        return written

    def _reset(self) -> None:
        self.result = None
        self.run = None
        self.results.clear()
        self.project_loaded.emit()
        self.modified_changed.emit(False)

    def update(self, section: str, **values: Any) -> None:
        """Set fields of ``engine``, ``rocket``, ``recovery`` or ``environment``."""
        target = getattr(self.data, section)
        changed = False
        for key, value in values.items():
            if getattr(target, key) != value:
                setattr(target, key, value)
                changed = True
        if changed:
            was = self.project.is_modified
            self.project.mark_modified()
            if not was:
                self.modified_changed.emit(True)
            if section in ("rocket", "recovery"):
                self.rocket_changed.emit()

    # engine

    def engine_inputs(self) -> EngineInputs:
        e = self.data.engine
        return EngineInputs(
            fuel=e.fuel,
            oxidizer=e.oxidizer,
            of_ratio=e.of_ratio,
            chamber_pressure=e.chamber_pressure_bar * 1e5,
            expansion_ratio=e.expansion_ratio,
            throat_area=e.throat_area_cm2 * 1e-4,
            ambient_pressure=e.ambient_pressure_bar * 1e5,
            eta_cstar=e.eta_cstar,
            eta_cf=e.eta_cf,
            nozzle_half_angle_deg=e.alpha_deg,
        )

    def run_engine(self) -> None:
        try:
            inputs = self.engine_inputs()
        except ValueError as error:
            self.engine_failed.emit(str(error))
            return
        geometry = (self.data.engine.contraction_ratio, self.data.engine.characteristic_length)
        self.engine_started.emit()
        self.tasks.submit(
            "engine",
            lambda ctx: compute_engine(inputs, *geometry, ctx),
            on_result=self._engine_done,
            on_error=self.engine_failed.emit,
        )

    def _engine_done(self, run: EngineRun) -> None:
        self.run = run
        self.result = run.result
        self.engine_elapsed = run.elapsed
        self.results["engine"] = run.result
        self.data.last_isp_vacuum = run.result.vacuum.isp
        self.data.last_thrust = run.result.vacuum.thrust
        self.engine_result.emit(run.result)

    # vehicle

    def rocket(self) -> Rocket:
        return self.project.build_rocket()

    def snapshot(self) -> dict[str, Any]:
        """Engine inputs and headline results, for the comparison table."""
        if self.result is None:
            raise ValueError("Run the engine first")
        r = self.result
        return {
            "inputs": asdict(r.inputs),
            "chamber_temperature": r.chamber_temperature,
            "isp_vacuum": r.vacuum.isp,
            "isp_sea_level": r.sea_level.isp if r.sea_level else None,
            "thrust_vacuum": r.vacuum.thrust,
            "c_star": r.vacuum.c_star,
            "gamma": r.gamma,
            "molecular_weight": r.molecular_weight,
        }
