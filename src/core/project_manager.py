"""Project save/load manager for EnSim."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ProjectData:
    """Data structure for saved project."""
    # Metadata
    version: str = "1.0.0"
    created: str = ""
    modified: str = ""

    # Propellants
    fuel: str = "H2"
    oxidizer: str = "O2"
    of_ratio: float = 8.0

    # Chamber
    chamber_pressure_bar: float = 68.0
    throat_area_cm2: float = 100.0

    # Nozzle
    expansion_ratio: float = 50.0
    ambient: str = "Vacuum (0 bar)"

    # Results (optional, saved if simulation was run)
    temperature: float | None = None
    isp_vacuum: float | None = None
    isp_sea_level: float | None = None
    thrust: float | None = None
    c_star: float | None = None
    gamma: float | None = None
    mean_mw: float | None = None


class ProjectManager:
    """
    Manages project save/load operations.

    Projects are stored as .ensim files (JSON format).
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
        """
        Save project to file.

        Args:
            path: File path. If None, uses current_path.

        Returns:
            True if saved successfully.
        """
        if path:
            self.current_path = Path(path)

        if not self.current_path:
            return False

        # Ensure extension
        if not self.current_path.suffix:
            self.current_path = self.current_path.with_suffix(self.FILE_EXTENSION)

        # Update modified time
        self.data.modified = datetime.now().isoformat()

        try:
            with open(self.current_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.data), f, indent=2)
            self._modified = False
            return True
        except Exception as e:
            print(f"Error saving project: {e}")
            return False

    def load(self, path: Path) -> ProjectData | None:
        """
        Load project from file.

        Args:
            path: Path to .ensim file.

        Returns:
            ProjectData if loaded successfully, None otherwise.
        """
        try:
            with open(path, encoding='utf-8') as f:
                data_dict = json.load(f)

            self.data = ProjectData(**data_dict)
            self.current_path = Path(path)
            self._modified = False
            return self.data

        except Exception as e:
            print(f"Error loading project: {e}")
            return None

    def update_inputs(
        self,
        fuel: str,
        oxidizer: str,
        of_ratio: float,
        chamber_pressure_bar: float,
        throat_area_cm2: float,
        expansion_ratio: float,
        ambient: str
    ):
        """Update input parameters."""
        self.data.fuel = fuel
        self.data.oxidizer = oxidizer
        self.data.of_ratio = of_ratio
        self.data.chamber_pressure_bar = chamber_pressure_bar
        self.data.throat_area_cm2 = throat_area_cm2
        self.data.expansion_ratio = expansion_ratio
        self.data.ambient = ambient
        self._modified = True

    def update_results(
        self,
        temperature: float,
        isp_vacuum: float,
        isp_sea_level: float,
        thrust: float,
        c_star: float,
        gamma: float,
        mean_mw: float
    ):
        """Update simulation results."""
        self.data.temperature = temperature
        self.data.isp_vacuum = isp_vacuum
        self.data.isp_sea_level = isp_sea_level
        self.data.thrust = thrust
        self.data.c_star = c_star
        self.data.gamma = gamma
        self.data.mean_mw = mean_mw
        self._modified = True

    def mark_modified(self):
        """Mark project as modified."""
        self._modified = True
