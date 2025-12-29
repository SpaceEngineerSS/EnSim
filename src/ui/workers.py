"""Worker threads for non-blocking calculations."""

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.chemistry import CombustionProblem
from src.core.propulsion import NozzleConditions, calculate_performance, PerformanceResult
from src.utils.nasa_parser import create_sample_database


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
