"""
Full Project Persistence Manager.

Saves and loads complete EnSim projects including:
- Engine parameters
- Rocket configuration
- Recovery system
- Environment settings
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.core.recovery import DeployTrigger, Parachute
from src.core.rocket import BodyTube, EngineMount, Fin, FinSet, NoseCone, NoseShape, Rocket


@dataclass
class EngineData:
    """Engine simulation parameters."""
    fuel: str = "H2"
    oxidizer: str = "O2"
    of_ratio: float = 8.0
    chamber_pressure_bar: float = 68.0
    throat_area_cm2: float = 100.0
    expansion_ratio: float = 50.0
    ambient: str = "Vacuum (0 bar)"
    eta_cstar: float = 0.95
    eta_cf: float = 0.95
    alpha_deg: float = 15.0


@dataclass
class RocketData:
    """Rocket geometry data."""
    name: str = "My Rocket"
    # Nose
    nose_shape: str = "ogive"
    nose_length: float = 0.25
    nose_diameter: float = 0.1
    nose_mass: float = 0.3
    nose_material: str = "fiberglass"
    # Body
    body_length: float = 1.0
    body_diameter: float = 0.1
    body_mass: float = 1.0
    body_material: str = "cardboard"
    # Fins
    fin_count: int = 4
    fin_root_chord: float = 0.12
    fin_tip_chord: float = 0.06
    fin_span: float = 0.06
    fin_sweep_angle: float = 30.0
    fin_mass: float = 0.2
    fin_material: str = "plywood"
    # Engine
    fuel_mass: float = 5.0
    oxidizer_mass: float = 25.0


@dataclass
class RecoveryData:
    """Recovery system data."""
    dual_deploy: bool = False
    main_diameter: float = 1.0
    main_cd: float = 1.5
    main_deploy_altitude: float = 200.0
    drogue_diameter: float = 0.3
    drogue_cd: float = 1.2


@dataclass
class EnvironmentData:
    """Launch environment data."""
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    rail_length: float = 1.5
    launch_altitude: float = 0.0


@dataclass
class ProjectData:
    """Complete project data structure."""
    # Meta
    version: str = "3.0"
    created: str = ""
    modified: str = ""

    # Components
    engine: EngineData = field(default_factory=EngineData)
    rocket: RocketData = field(default_factory=RocketData)
    recovery: RecoveryData = field(default_factory=RecoveryData)
    environment: EnvironmentData = field(default_factory=EnvironmentData)

    # Cached results (optional)
    last_isp_vacuum: float | None = None
    last_thrust: float | None = None
    last_apogee: float | None = None


class ProjectManagerV3:
    """
    Full project persistence manager (Version 3).

    Saves/loads complete rocket projects including vehicle,
    recovery, and environment configuration.
    """

    FILE_EXTENSION = ".ensim"
    FILE_FILTER = "EnSim Project (*.ensim);;All Files (*)"

    def __init__(self):
        self.current_path: Path | None = None
        self.data = ProjectData()
        self._modified = False

    @property
    def is_modified(self) -> bool:
        return self._modified

    @property
    def project_name(self) -> str:
        if self.current_path:
            return self.current_path.stem
        return "Untitled"

    def new_project(self) -> ProjectData:
        """Create a new blank project."""
        self.current_path = None
        self.data = ProjectData(
            created=datetime.now().isoformat(),
            modified=datetime.now().isoformat()
        )
        self._modified = False
        return self.data

    def save(self, path: Path | None = None) -> bool:
        """Save project to .ensim file."""
        if path:
            self.current_path = Path(path)

        if not self.current_path:
            return False

        if not self.current_path.suffix:
            self.current_path = self.current_path.with_suffix(self.FILE_EXTENSION)

        self.data.modified = datetime.now().isoformat()

        try:
            # Convert to nested dict
            project_dict = {
                "meta": {
                    "version": self.data.version,
                    "created": self.data.created,
                    "modified": self.data.modified
                },
                "engine": asdict(self.data.engine),
                "rocket": asdict(self.data.rocket),
                "recovery": asdict(self.data.recovery),
                "environment": asdict(self.data.environment),
                "results": {
                    "isp_vacuum": self.data.last_isp_vacuum,
                    "thrust": self.data.last_thrust,
                    "apogee": self.data.last_apogee
                }
            }

            with open(self.current_path, 'w', encoding='utf-8') as f:
                json.dump(project_dict, f, indent=2)

            self._modified = False
            return True

        except Exception as e:
            print(f"Error saving project: {e}")
            return False

    def load(self, path: Path) -> ProjectData | None:
        """Load project from .ensim file."""
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            # Parse nested structure
            self.data = ProjectData(
                version=data.get("meta", {}).get("version", "1.0"),
                created=data.get("meta", {}).get("created", ""),
                modified=data.get("meta", {}).get("modified", "")
            )

            # Engine
            if "engine" in data:
                self.data.engine = EngineData(**data["engine"])

            # Rocket
            if "rocket" in data:
                self.data.rocket = RocketData(**data["rocket"])

            # Recovery
            if "recovery" in data:
                self.data.recovery = RecoveryData(**data["recovery"])

            # Environment
            if "environment" in data:
                self.data.environment = EnvironmentData(**data["environment"])

            # Results
            if "results" in data:
                self.data.last_isp_vacuum = data["results"].get("isp_vacuum")
                self.data.last_thrust = data["results"].get("thrust")
                self.data.last_apogee = data["results"].get("apogee")

            self.current_path = Path(path)
            self._modified = False
            return self.data

        except Exception as e:
            print(f"Error loading project: {e}")
            return None

    def update_from_ui(
        self,
        engine_params: dict,
        rocket_params: dict,
        recovery_params: dict,
        environment_params: dict
    ):
        """Update project data from UI widgets."""
        # Engine
        for key, value in engine_params.items():
            if hasattr(self.data.engine, key):
                setattr(self.data.engine, key, value)

        # Rocket
        for key, value in rocket_params.items():
            if hasattr(self.data.rocket, key):
                setattr(self.data.rocket, key, value)

        # Recovery
        for key, value in recovery_params.items():
            if hasattr(self.data.recovery, key):
                setattr(self.data.recovery, key, value)

        # Environment
        for key, value in environment_params.items():
            if hasattr(self.data.environment, key):
                setattr(self.data.environment, key, value)

        self._modified = True

    def build_rocket(self) -> Rocket:
        """Construct Rocket object from saved data."""
        rd = self.data.rocket

        # Map shape name to enum
        shape_map = {
            "ogive": NoseShape.OGIVE,
            "conical": NoseShape.CONICAL,
            "elliptical": NoseShape.ELLIPTICAL,
            "parabolic": NoseShape.PARABOLIC
        }
        nose_shape = shape_map.get(rd.nose_shape.lower(), NoseShape.OGIVE)

        return Rocket(
            name=rd.name,
            nose=NoseCone(
                shape=nose_shape,
                length=rd.nose_length,
                diameter=rd.nose_diameter,
                mass=rd.nose_mass
            ),
            body=BodyTube(
                length=rd.body_length,
                diameter=rd.body_diameter,
                mass=rd.body_mass
            ),
            fins=FinSet(
                fin=Fin(
                    root_chord=rd.fin_root_chord,
                    tip_chord=rd.fin_tip_chord,
                    span=rd.fin_span,
                    sweep_angle=rd.fin_sweep_angle
                ),
                count=rd.fin_count,
                mass=rd.fin_mass
            ),
            engine=EngineMount(
                engine_mass_dry=0.5,
                fuel_mass=rd.fuel_mass,
                oxidizer_mass=rd.oxidizer_mass,
                tank_length=0.5
            )
        )

    def build_recovery(self) -> tuple:
        """Construct recovery system from saved data."""
        rec = self.data.recovery

        main = Parachute(
            name="Main",
            diameter=rec.main_diameter,
            cd=rec.main_cd,
            deploy_trigger=DeployTrigger.AT_ALTITUDE if rec.dual_deploy else DeployTrigger.AT_APOGEE,
            deploy_altitude=rec.main_deploy_altitude
        )

        drogue = None
        if rec.dual_deploy:
            drogue = Parachute(
                name="Drogue",
                diameter=rec.drogue_diameter,
                cd=rec.drogue_cd,
                deploy_trigger=DeployTrigger.AT_APOGEE
            )

        return main, drogue

    def mark_modified(self):
        """Mark project as modified."""
        self._modified = True
