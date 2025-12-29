"""Worker threads for non-blocking calculations."""

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.chemistry import CombustionProblem
from src.core.propulsion import NozzleConditions, calculate_performance, PerformanceResult
from src.utils.nasa_parser import create_sample_database
from src.core.rocket import Rocket
from src.core.flight_6dof import simulate_flight_6dof, FlightResult6DOF
from src.core.monte_carlo import run_monte_carlo, DispersionConfig, DispersionResult


@dataclass
class SimulationParams:
    """Parameters for a simulation run."""
    fuel: str = "H2"
    oxidizer: str = "O2"
    fuel_moles: float = 2.0
    oxidizer_moles: float = 1.0
    chamber_pressure_bar: float = 10.0
    expansion_ratio: float = 50.0
    ambient_pressure_bar: float = 0.0  # 0 = vacuum
    throat_area_cm2: Optional[float] = 100.0  # cm²
    eta_cstar: float = 1.0  # Combustion efficiency
    eta_cf: float = 1.0  # Nozzle efficiency
    alpha_deg: float = 15.0  # Nozzle half-angle (degrees)
    
    @property
    def of_ratio(self) -> float:
        """Calculate O/F mass ratio."""
        # Approximate MW for common propellants
        mw = {"H2": 2.016, "CH4": 16.04, "O2": 32.0, "N2O": 44.01}
        fuel_mass = self.fuel_moles * mw.get(self.fuel, 16.0)
        ox_mass = self.oxidizer_moles * mw.get(self.oxidizer, 32.0)
        return ox_mass / fuel_mass if fuel_mass > 0 else 0


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
    c_star: float
    exit_velocity: float
    thrust: float
    exit_mach: float
    mass_flow_rate: Optional[float]
    
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
        """Execute the simulation in background thread."""
        try:
            # Step 1: Load database
            self.log.emit("Loading thermodynamic database...")
            self._species_db = create_sample_database()
            
            # Step 2: Setup combustion problem
            self.log.emit(f"Setting up {self.params.fuel}/{self.params.oxidizer} combustion...")
            problem = CombustionProblem(self._species_db)
            problem.add_fuel(self.params.fuel, moles=self.params.fuel_moles)
            problem.add_oxidizer(self.params.oxidizer, moles=self.params.oxidizer_moles)
            
            # Step 3: Solve equilibrium
            P_chamber = self.params.chamber_pressure_bar * 1e5  # bar to Pa
            
            # HOTFIX: Pressure guard - prevent solver crash at extreme low pressure
            if self.params.chamber_pressure_bar < 0.5:
                raise ValueError(f"Chamber pressure too low ({self.params.chamber_pressure_bar} bar). Minimum: 0.5 bar.")
            
            self.log.emit(f"Solving chemical equilibrium at {self.params.chamber_pressure_bar:.1f} bar...")
            
            eq_result = problem.solve(
                pressure=P_chamber,
                initial_temp_guess=3000.0,
                max_iterations=50,
                tolerance=1e-5
            )
            
            self.log.emit(f"  T = {eq_result.temperature:.1f} K, γ = {eq_result.gamma:.3f}")
            
            # Step 4: Calculate vacuum performance
            self.log.emit(f"Calculating nozzle performance (ε = {self.params.expansion_ratio:.0f})...")
            
            throat_area = None
            if self.params.throat_area_cm2:
                throat_area = self.params.throat_area_cm2 * 1e-4  # cm² to m²
            
            nozzle_vac = NozzleConditions(
                area_ratio=self.params.expansion_ratio,
                chamber_pressure=P_chamber,
                ambient_pressure=0.0,
                throat_area=throat_area
            )
            
            perf_vac = calculate_performance(
                T_chamber=eq_result.temperature,
                P_chamber=P_chamber,
                gamma=eq_result.gamma,
                mean_molecular_weight=eq_result.mean_molecular_weight,
                nozzle=nozzle_vac,
                eta_cstar=self.params.eta_cstar,
                eta_cf=self.params.eta_cf,
                alpha_deg=self.params.alpha_deg
            )
            
            self.log.emit(f"  Vacuum Isp = {perf_vac.isp:.1f} s")
            
            # Step 5: Calculate sea-level performance
            self.log.emit("Calculating sea-level performance...")
            
            nozzle_sl = NozzleConditions(
                area_ratio=min(self.params.expansion_ratio, 15.0),  # Limit for SL
                chamber_pressure=P_chamber,
                ambient_pressure=101325.0,
                throat_area=throat_area
            )
            
            perf_sl = calculate_performance(
                T_chamber=eq_result.temperature,
                P_chamber=P_chamber,
                gamma=eq_result.gamma,
                mean_molecular_weight=eq_result.mean_molecular_weight,
                nozzle=nozzle_sl,
                eta_cstar=self.params.eta_cstar,
                eta_cf=self.params.eta_cf,
                alpha_deg=self.params.alpha_deg
            )
            
            self.log.emit(f"  Sea Level Isp = {perf_sl.isp:.1f} s")
            
            # Build species fractions dict
            species_fractions = {}
            for name, frac in zip(eq_result.species_names, eq_result.mole_fractions):
                if frac > 0.001:
                    species_fractions[name] = frac
            
            # Build result
            result = SimulationResult(
                temperature=eq_result.temperature,
                gamma=eq_result.gamma,
                mean_mw=eq_result.mean_molecular_weight,
                species_fractions=species_fractions,
                isp_vacuum=perf_vac.isp,
                isp_sea_level=perf_sl.isp,
                c_star=perf_vac.c_star,
                exit_velocity=perf_vac.exit_velocity,
                thrust=perf_vac.thrust,
                exit_mach=perf_vac.exit_mach,
                mass_flow_rate=perf_vac.mass_flow_rate,
                converged=eq_result.converged,
                iterations=eq_result.iterations
            )
            
            self.log.emit("Simulation complete!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# 6-DOF Flight Simulation Worker
# =============================================================================

@dataclass
class FlightParams:
    """Parameters for 6-DOF flight simulation."""
    # Required
    thrust_vac: float = 10000.0      # Vacuum thrust (N)
    isp_vac: float = 250.0           # Vacuum ISP (s)
    burn_time: float = 10.0          # Burn time (s)
    
    # Rocket configuration
    fuel_mass: float = 5.0           # kg
    oxidizer_mass: float = 15.0      # kg
    
    # Launch parameters
    launch_angle_deg: float = 85.0   # From horizontal
    launch_azimuth_deg: float = 0.0  # North = 0
    
    # Simulation parameters
    dt: float = 0.01                 # Time step
    max_time: float = 300.0          # Max simulation time
    exit_area: float = 0.01          # Nozzle exit area (m²)
    
    # Adaptive parameters
    use_adaptive: bool = True
    output_dt: float = 0.01          # Fixed output rate
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
                fin_misalignment_deg=self.params.fin_misalignment_deg
            )
            
            self.progress.emit(90)
            self.log.emit(f"Simulation complete! Apogee: {result.apogee_altitude:.1f}m")
            self.progress.emit(100)
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# Monte Carlo Dispersion Worker
# =============================================================================

@dataclass
class MonteCarloParams:
    """Parameters for Monte Carlo dispersion analysis."""
    # Base flight params
    thrust_vac: float = 10000.0
    isp_vac: float = 250.0
    burn_time: float = 10.0
    fuel_mass: float = 5.0
    oxidizer_mass: float = 15.0
    
    # Monte Carlo config
    num_simulations: int = 100
    thrust_sigma: float = 0.02       # 2% thrust variation
    isp_sigma: float = 0.01          # 1% ISP variation
    cd_sigma: float = 0.05           # 5% drag variation
    wind_speed_mean: float = 3.0     # m/s
    wind_speed_sigma: float = 2.0    # m/s
    seed: Optional[int] = None       # Reproducibility


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
            
            rocket = Rocket()
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
                seed=self.params.seed
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
                verbose=False
            )
            
            self.progress.emit(95)
            self.log.emit(f"Monte Carlo complete! CEP: {result.cep_radius:.1f}m")
            self.progress.emit(100)
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
