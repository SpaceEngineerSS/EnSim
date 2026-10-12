"""
Reading and writing ``.ensim`` project files.

A project is JSON with five sections: ``meta``, ``engine``, ``rocket``,
``recovery``, ``environment`` and an optional ``results`` cache. Files written
by EnSim 1.x (flat engine keys) and 3.x load unchanged; missing fields take
their defaults and unknown fields are ignored with a log message, so newer
files still open in older versions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from ensim.core.propellants import LEGACY_NAMES
from ensim.core.recovery import DeployTrigger, Parachute
from ensim.core.rocket import BodyTube, EngineMount, Fin, FinSet, NoseCone, NoseShape, Rocket

logger = logging.getLogger(__name__)

FORMAT_VERSION = "4.0"
FILE_EXTENSION = ".ensim"


class ProjectFileError(Exception):
    """A project file could not be read or written. The message is user-facing."""


@dataclass
class EngineData:
    """Engine inputs."""

    fuel: str = "LH2"
    oxidizer: str = "LOX"
    of_ratio: float = 6.0
    chamber_pressure_bar: float = 68.0
    throat_area_cm2: float = 100.0
    expansion_ratio: float = 50.0
    ambient_pressure_bar: float = 0.0
    eta_cstar: float = 0.95
    eta_cf: float = 0.95
    alpha_deg: float = 15.0


@dataclass
class RocketData:
    """Airframe and propellant load. Defaults match :func:`create_default_rocket`."""

    name: str = "My Rocket"
    nose_shape: str = "ogive"
    nose_length: float = 0.45
    nose_diameter: float = 0.12
    nose_mass: float = 0.7
    nose_material: str = "fiberglass"
    body_length: float = 1.8
    body_diameter: float = 0.12
    body_mass: float = 2.5
    body_material: str = "fiberglass"
    axial_drag_coefficient: float = 0.45
    fin_count: int = 4
    fin_root_chord: float = 0.26
    fin_tip_chord: float = 0.10
    fin_span: float = 0.16
    fin_sweep_angle: float = 35.0
    fin_mass: float = 0.6
    fin_material: str = "plywood"
    engine_mass_dry: float = 2.0
    tank_length: float = 0.45
    fuel_mass: float = 0.8
    oxidizer_mass: float = 2.4


@dataclass
class RecoveryData:
    """Parachutes."""

    dual_deploy: bool = False
    main_diameter: float = 1.0
    main_cd: float = 1.5
    main_deploy_altitude: float = 200.0
    drogue_diameter: float = 0.3
    drogue_cd: float = 1.2


@dataclass
class EnvironmentData:
    """Launch site and weather."""

    wind_speed: float = 0.0
    wind_direction: float = 0.0
    rail_length: float = 1.5
    launch_altitude: float = 0.0
    launch_angle: float = 85.0


@dataclass
class ProjectData:
    """Everything stored in one project file."""

    version: str = FORMAT_VERSION
    created: str = ""
    modified: str = ""
    engine: EngineData = field(default_factory=EngineData)
    rocket: RocketData = field(default_factory=RocketData)
    recovery: RecoveryData = field(default_factory=RecoveryData)
    environment: EnvironmentData = field(default_factory=EnvironmentData)
    last_isp_vacuum: float | None = None
    last_thrust: float | None = None
    last_apogee: float | None = None


T = TypeVar("T")


def _ambient_from_label(label: str) -> float:
    """3.x stored the ambient choice as text such as 'Sea Level (1.01 bar)'."""
    match = re.search(r"\(([\d.]+)\s*bar\)", label)
    return float(match.group(1)) if match else 0.0


def _major_version(text: str) -> int:
    match = re.match(r"\s*(\d+)", text)
    return int(match.group(1)) if match else 1


def _section(cls: type[T], values: Any, name: str) -> T:
    """Build a section dataclass, ignoring unknown keys and keeping defaults for missing ones."""
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise ProjectFileError(f"Section '{name}' must be an object")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        logger.info("Ignoring unknown %s fields: %s", name, ", ".join(unknown))
    try:
        return cls(**{key: values[key] for key in values if key in known})
    except TypeError as error:
        raise ProjectFileError(f"Section '{name}' is invalid: {error}") from error


class ProjectManager:
    """Holds the open project and its file path."""

    FILE_EXTENSION = FILE_EXTENSION

    def __init__(self) -> None:
        self.current_path: Path | None = None
        self.data = ProjectData()
        self._modified = False

    @property
    def is_modified(self) -> bool:
        return self._modified

    @property
    def project_name(self) -> str:
        return self.current_path.stem if self.current_path else "Untitled"

    def mark_modified(self) -> None:
        self._modified = True

    def new_project(self) -> ProjectData:
        now = datetime.now().isoformat(timespec="seconds")
        self.current_path = None
        self.data = ProjectData(created=now, modified=now)
        self._modified = False
        return self.data

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "version": FORMAT_VERSION,
                "created": self.data.created,
                "modified": self.data.modified,
            },
            "engine": asdict(self.data.engine),
            "rocket": asdict(self.data.rocket),
            "recovery": asdict(self.data.recovery),
            "environment": asdict(self.data.environment),
            "results": {
                "isp_vacuum": self.data.last_isp_vacuum,
                "thrust": self.data.last_thrust,
                "apogee": self.data.last_apogee,
            },
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Write the project and return the path written.

        Raises:
            ProjectFileError: No path is known or the file cannot be written.
        """
        target = Path(path) if path is not None else self.current_path
        if target is None:
            raise ProjectFileError("Choose a file name before saving")
        if not target.suffix:
            target = target.with_suffix(FILE_EXTENSION)

        self.data.modified = datetime.now().isoformat(timespec="seconds")
        if not self.data.created:
            self.data.created = self.data.modified
        try:
            target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError as error:
            raise ProjectFileError(
                f"Could not write {target}: {error.strerror or error}"
            ) from error
        self.current_path = target
        self.data.version = FORMAT_VERSION
        self._modified = False
        return target

    def load(self, path: Path | str) -> ProjectData:
        """Read a project file.

        Raises:
            ProjectFileError: The file cannot be read or is not a valid project.
        """
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ProjectFileError(f"Could not read {path}: {error.strerror or error}") from error
        except json.JSONDecodeError as error:
            raise ProjectFileError(f"{path.name} is not a valid project file") from error
        if not isinstance(raw, dict):
            raise ProjectFileError(f"{path.name} is not a valid project file")

        meta = raw.get("meta") or {}
        data = ProjectData(
            version=str(meta.get("version", "1.0")),
            created=str(meta.get("created", "")),
            modified=str(meta.get("modified", "")),
        )
        if "engine" not in raw and "fuel" in raw:
            # EnSim 1.x wrote the engine inputs at the top level
            data.engine = _section(EngineData, raw, "engine")
            data.last_isp_vacuum = raw.get("isp_vacuum")
            data.last_thrust = raw.get("thrust")
        else:
            engine = dict(raw.get("engine") or {})
            if "ambient" in engine and "ambient_pressure_bar" not in engine:
                engine["ambient_pressure_bar"] = _ambient_from_label(str(engine.pop("ambient")))
            data.engine = _section(EngineData, engine, "engine")
            data.rocket = _section(RocketData, raw.get("rocket"), "rocket")
            data.recovery = _section(RecoveryData, raw.get("recovery"), "recovery")
            data.environment = _section(EnvironmentData, raw.get("environment"), "environment")
            results = raw.get("results") or {}
            data.last_isp_vacuum = results.get("isp_vacuum")
            data.last_thrust = results.get("thrust")
            data.last_apogee = results.get("apogee")

        if _major_version(data.version) < 4:
            # 3.x and older treated every reactant as a 298 K gas under these names
            data.engine.fuel = LEGACY_NAMES.get(data.engine.fuel, data.engine.fuel)
            data.engine.oxidizer = LEGACY_NAMES.get(data.engine.oxidizer, data.engine.oxidizer)

        self.data = data
        self.current_path = path
        self._modified = False
        return data

    def update_inputs(
        self,
        fuel: str,
        oxidizer: str,
        of_ratio: float,
        chamber_pressure_bar: float,
        throat_area_cm2: float,
        expansion_ratio: float,
        ambient_pressure_bar: float,
    ) -> None:
        """Set the main engine inputs."""
        engine = self.data.engine
        engine.fuel = fuel
        engine.oxidizer = oxidizer
        engine.of_ratio = of_ratio
        engine.chamber_pressure_bar = chamber_pressure_bar
        engine.throat_area_cm2 = throat_area_cm2
        engine.expansion_ratio = expansion_ratio
        engine.ambient_pressure_bar = ambient_pressure_bar
        self._modified = True

    def update_from_ui(
        self,
        engine_params: dict,
        rocket_params: dict,
        recovery_params: dict,
        environment_params: dict,
    ) -> None:
        """Copy matching keys from the UI dictionaries into the project."""
        for section, values in (
            (self.data.engine, engine_params),
            (self.data.rocket, rocket_params),
            (self.data.recovery, recovery_params),
            (self.data.environment, environment_params),
        ):
            names = {f.name for f in fields(section)}
            for key, value in values.items():
                if key in names:
                    setattr(section, key, value)
        self._modified = True

    def build_rocket(self) -> Rocket:
        """Rocket described by the project."""
        rd = self.data.rocket
        try:
            nose_shape = NoseShape(rd.nose_shape.lower())
        except ValueError as error:
            choices = ", ".join(shape.value for shape in NoseShape)
            raise ProjectFileError(
                f"Unknown nose shape '{rd.nose_shape}'. Use one of: {choices}"
            ) from error
        return Rocket(
            name=rd.name,
            axial_drag_coefficient=rd.axial_drag_coefficient,
            nose=NoseCone(
                shape=nose_shape,
                length=rd.nose_length,
                diameter=rd.nose_diameter,
                mass=rd.nose_mass,
                material=rd.nose_material,
            ),
            body=BodyTube(
                length=rd.body_length,
                diameter=rd.body_diameter,
                mass=rd.body_mass,
                material=rd.body_material,
            ),
            fins=FinSet(
                fin=Fin(
                    root_chord=rd.fin_root_chord,
                    tip_chord=rd.fin_tip_chord,
                    span=rd.fin_span,
                    sweep_angle=rd.fin_sweep_angle,
                ),
                count=rd.fin_count,
                mass=rd.fin_mass,
                material=rd.fin_material,
            ),
            engine=EngineMount(
                engine_mass_dry=rd.engine_mass_dry,
                fuel_mass=rd.fuel_mass,
                oxidizer_mass=rd.oxidizer_mass,
                tank_length=rd.tank_length,
            ),
        )

    def build_recovery(self) -> tuple[Parachute, Parachute | None]:
        """Main parachute and, with dual deployment, the drogue."""
        rec = self.data.recovery
        main = Parachute(
            name="Main",
            diameter=rec.main_diameter,
            cd=rec.main_cd,
            deploy_trigger=DeployTrigger.AT_ALTITUDE
            if rec.dual_deploy
            else DeployTrigger.AT_APOGEE,
            deploy_altitude=rec.main_deploy_altitude,
        )
        drogue = None
        if rec.dual_deploy:
            drogue = Parachute(
                name="Drogue",
                diameter=rec.drogue_diameter,
                cd=rec.drogue_cd,
                deploy_trigger=DeployTrigger.AT_APOGEE,
            )
        return main, drogue
