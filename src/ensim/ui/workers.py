"""Worker threads for non-blocking calculations."""

from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

from ensim.core.engine import EngineInputs, analyze_engine
from ensim.core.flight_6dof import simulate_flight_6dof
from ensim.core.monte_carlo import DispersionConfig, run_monte_carlo
from ensim.core.rocket import Rocket


@dataclass
class SimulationParams:
    """Parameters for a simulation run."""

    fuel: str = "H2"
    oxidizer: str = "O2"
    fuel_moles: float = 2.0
    oxidizer_moles: float = 1.0
    of_ratio_mass: float | None = None
    chamber_pressure_bar: float = 10.0
    expansion_ratio: float = 50.0
    ambient_pressure_bar: float = 0.0  # 0 = vacuum
    throat_area_cm2: float | None = 100.0  # cm²
    eta_cstar: float = 1.0  # Combustion efficiency
    eta_cf: float = 1.0  # Nozzle efficiency
    alpha_deg: float = 15.0  # Nozzle half-angle (degrees)

    @property
    def of_ratio(self) -> float:
        """Calculate O/F mass ratio."""
        return self.of_ratio_mass if self.of_ratio_mass is not None else 0.0


@dataclass
class SimulationResult:
    """Complete simulation results."""

    # Combustion
    temperature: float
    gamma: float
    mean_mw: float
    species_fractions: dict

    # Performance
    isp_vacuum: float
    isp_sea_level: float
    isp_operating: float
    c_star: float
    exit_velocity: float
    thrust_vacuum: float
    thrust_operating: float
    exit_mach: float
    exit_pressure: float
    operating_ambient_pressure: float
    mass_flow_rate: float | None

    # Status
    converged: bool
    iterations: int


class CalculationWorker(QThread):
    """
    Worker thread for running simulations.

    Signals:
        log(str): Progress messages
        finished(SimulationResult): Emitted on success
        error(str): Emitted on failure
    """

    log = pyqtSignal(str)
    finished = pyqtSignal(object)  # SimulationResult
    error = pyqtSignal(str)

    def __init__(self, params: SimulationParams, parent=None):
        super().__init__(parent)
        self.params = params
        self._species_db = None

    def run(self):
        """Run ``analyze_engine`` off the GUI thread."""
        try:
            params = self.params
            if params.of_ratio_mass is None:
                raise ValueError("A mass O/F ratio is required")
            if params.chamber_pressure_bar < 2.0:
                raise ValueError(
                    f"Chamber pressure too low ({params.chamber_pressure_bar} bar). "
                    "At least 2 bar is required for the sea-level choked-flow result."
                )
            self.log.emit(
                f"Solving {params.fuel}/{params.oxidizer} at {params.chamber_pressure_bar:g} bar"
            )
            result = analyze_engine(
                EngineInputs(
                    fuel=params.fuel,
                    oxidizer=params.oxidizer,
                    of_ratio=params.of_ratio_mass,
                    chamber_pressure=params.chamber_pressure_bar * 1e5,
                    expansion_ratio=params.expansion_ratio,
                    throat_area=params.throat_area_cm2 * 1e-4 if params.throat_area_cm2 else None,
                    ambient_pressure=params.ambient_pressure_bar * 1e5,
                    eta_cstar=params.eta_cstar,
                    eta_cf=params.eta_cf,
                    nozzle_half_angle_deg=params.alpha_deg,
                )
            )
            sea_level = result.sea_level or result.operating
            self.finished.emit(
                SimulationResult(
                    temperature=result.chamber_temperature,
                    gamma=result.gamma,
                    mean_mw=result.molecular_weight,
                    species_fractions={k: v for k, v in result.composition.items() if v > 1e-3},
                    isp_vacuum=result.vacuum.isp,
                    isp_sea_level=sea_level.isp,
                    isp_operating=result.operating.isp,
                    c_star=result.vacuum.c_star,
                    exit_velocity=result.vacuum.exit_velocity,
                    thrust_vacuum=result.vacuum.thrust,
                    thrust_operating=result.operating.thrust,
                    exit_mach=result.vacuum.exit_mach,
                    exit_pressure=result.vacuum.exit_pressure,
                    operating_ambient_pressure=result.inputs.ambient_pressure,
                    mass_flow_rate=result.vacuum.mass_flow_rate,
                    converged=result.equilibrium.converged,
                    iterations=result.equilibrium.iterations,
                )
            )
        except KeyError as e:
            self.error.emit(str(e.args[0]) if e.args else str(e))
        except Exception as e:
            self.error.emit(str(e))


# 6-DOF Flight Simulation Worker
@dataclass
class FlightParams:
    """Parameters for 6-DOF flight simulation."""

    # Required
    thrust_vac: float = 10000.0  # Vacuum thrust (N)
    isp_vac: float = 250.0  # Vacuum ISP (s)
    burn_time: float = 10.0  # Burn time (s)

    # Rocket configuration
    fuel_mass: float = 5.0  # kg
    oxidizer_mass: float = 15.0  # kg

    # Launch parameters
    launch_angle_deg: float = 85.0  # From horizontal
    launch_azimuth_deg: float = 0.0  # North = 0

    # Simulation parameters
    dt: float = 0.01  # Time step
    max_time: float = 300.0  # Max simulation time
    exit_area: float = 0.01  # Nozzle exit area (m²)

    # Adaptive parameters
    use_adaptive: bool = True
    output_dt: float = 0.01  # Fixed output rate
    rtol: float = 1e-6
    atol: float = 1e-6

    # Perturbation parameters (Monte Carlo)
    throttle: float = 1.0
    cd_factor: float = 1.0
    fin_misalignment_deg: float = 0.0


class FlightSimulationWorker(QThread):
    """
    Worker thread for 6-DOF flight simulation.

    Signals:
        log(str): Progress messages
        progress(int): Progress percentage (0-100)
        finished(FlightResult6DOF): Emitted on success
        error(str): Emitted on failure
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # FlightResult6DOF
    error = pyqtSignal(str)

    def __init__(self, params: FlightParams, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        """Execute 6-DOF flight simulation in background thread."""
        try:
            # Step 1: Create rocket
            self.log.emit("Creating rocket configuration...")
            self.progress.emit(10)

            rocket = Rocket()
            rocket.engine.fuel_mass = self.params.fuel_mass
            rocket.engine.oxidizer_mass = self.params.oxidizer_mass
            rocket.engine.thrust_vac = self.params.thrust_vac
            rocket.engine.isp_vac = self.params.isp_vac

            # Step 2: Run simulation
            self.log.emit(f"Running 6-DOF simulation (adaptive={self.params.use_adaptive})...")
            self.progress.emit(30)

            result = simulate_flight_6dof(
                rocket=rocket,
                thrust_vac=self.params.thrust_vac,
                isp_vac=self.params.isp_vac,
                burn_time=self.params.burn_time,
                exit_area=self.params.exit_area,
                dt=self.params.dt,
                max_time=self.params.max_time,
                launch_angle_deg=self.params.launch_angle_deg,
                launch_azimuth_deg=self.params.launch_azimuth_deg,
                use_adaptive=self.params.use_adaptive,
                output_dt=self.params.output_dt,
                rtol=self.params.rtol,
                atol=self.params.atol,
                throttle=self.params.throttle,
                cd_factor=self.params.cd_factor,
                fin_misalignment_deg=self.params.fin_misalignment_deg,
            )

            self.progress.emit(90)
            self.log.emit(f"Simulation complete! Apogee: {result.apogee_altitude:.1f}m")
            self.progress.emit(100)

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# Monte Carlo Dispersion Worker
@dataclass
class MonteCarloParams:
    """Parameters for Monte Carlo dispersion analysis."""

    # Base flight params
    rocket: Rocket | None = None
    thrust_vac: float = 10000.0
    isp_vac: float = 250.0
    burn_time: float = 10.0
    fuel_mass: float = 5.0
    oxidizer_mass: float = 15.0

    # Monte Carlo config
    num_simulations: int = 100
    thrust_sigma: float = 0.02  # 2% thrust variation
    isp_sigma: float = 0.01  # 1% ISP variation
    cd_sigma: float = 0.05  # 5% drag variation
    wind_speed_mean: float = 3.0  # m/s
    wind_speed_sigma: float = 2.0  # m/s
    seed: int | None = None  # Reproducibility


class MonteCarloWorker(QThread):
    """
    Worker thread for Monte Carlo dispersion analysis.

    Signals:
        log(str): Progress messages
        progress(int): Progress percentage (0-100)
        finished(DispersionResult): Emitted on success
        error(str): Emitted on failure
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # DispersionResult
    error = pyqtSignal(str)

    def __init__(self, params: MonteCarloParams, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        """Execute Monte Carlo analysis in background thread."""
        try:
            # Step 1: Create rocket
            self.log.emit("Creating rocket configuration...")
            self.progress.emit(5)

            rocket = self.params.rocket or Rocket()
            if self.params.rocket is None:
                rocket.engine.fuel_mass = self.params.fuel_mass
                rocket.engine.oxidizer_mass = self.params.oxidizer_mass

            # Step 2: Configure dispersion
            self.log.emit(f"Starting Monte Carlo ({self.params.num_simulations} simulations)...")
            self.progress.emit(10)

            config = DispersionConfig(
                num_simulations=self.params.num_simulations,
                thrust_sigma=self.params.thrust_sigma,
                isp_sigma=self.params.isp_sigma,
                cd_sigma=self.params.cd_sigma,
                wind_speed_mean=self.params.wind_speed_mean,
                wind_speed_sigma=self.params.wind_speed_sigma,
                seed=self.params.seed,
            )

            # Step 3: Run Monte Carlo
            result = run_monte_carlo(
                rocket=rocket,
                thrust_vac=self.params.thrust_vac,
                isp_vac=self.params.isp_vac,
                burn_time=self.params.burn_time,
                config=config,
                dt=0.05,  # Faster for MC
                max_time=120.0,
            )

            self.progress.emit(95)
            self.log.emit(f"Monte Carlo complete! CEP: {result.cep_radius:.1f}m")
            self.progress.emit(100)

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
