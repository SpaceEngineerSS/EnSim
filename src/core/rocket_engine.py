"""
Rocket Engine Real-Time Model.

Provides dynamic thrust and ISP calculations based on altitude,
chamber conditions, and flow regime. Integrates with the 6-DOF
flight simulator for realistic propulsion modeling.

Features:
    - Altitude-corrected thrust using ambient pressure
    - Flow separation detection (Summerfield criterion)
    - Thrust loss factor for over-expanded operation
    - Pre-computed lookup table interpolation for combustion properties

References:
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed.
    - NASA RP-1311 (Gordon & McBride)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numba import jit

from .constants import GAS_CONSTANT, G0
from .propulsion import (
    calculate_c_star,
    calculate_thrust_coefficient,
    calculate_exit_conditions,
    solve_mach_from_area_ratio_supersonic,
)


# =============================================================================
# Flow Status Enumeration and Dataclass
# =============================================================================

class FlowRegime(Enum):
    """Nozzle flow regime classification."""
    ATTACHED = "attached"           # Normal operation
    OVEREXPANDED = "overexpanded"   # Pe < Pa but flow still attached
    SEPARATED = "separated"         # Flow separation at nozzle wall
    UNDEREXPANDED = "underexpanded" # Pe > Pa, expansion waves at exit


@dataclass
class FlowStatus:
    """
    Nozzle flow status with thrust correction factors.
    
    Attributes:
        regime: Current flow regime
        Pe_Pa_ratio: Exit pressure to ambient pressure ratio
        thrust_loss_factor: Multiplicative factor for thrust (0.0-1.0)
        side_load_warning: True if side loads are expected (separation)
        message: Human-readable status message
    """
    regime: FlowRegime
    Pe_Pa_ratio: float
    thrust_loss_factor: float
    side_load_warning: bool
    message: str


@jit(nopython=True, cache=True)
def calculate_flow_status_jit(
    P_exit: float,
    P_ambient: float
) -> Tuple[int, float, float, bool]:
    """
    JIT-compiled flow status calculation.
    
    Returns:
        Tuple of (regime_code, Pe_Pa_ratio, thrust_loss_factor, side_load_warning)
        regime_code: 0=attached, 1=overexpanded, 2=separated, 3=underexpanded
    """
    # Vacuum case
    if P_ambient <= 1.0:  # Essentially vacuum
        return 0, 1e6, 1.0, False  # Attached, no loss
    
    Pe_Pa = P_exit / P_ambient
    
    # Underexpanded: Pe > Pa
    if Pe_Pa >= 1.0:
        return 3, Pe_Pa, 1.0, False  # No thrust loss
    
    # Summerfield separation criterion: Pe/Pa < 0.4
    SUMMERFIELD_THRESHOLD = 0.4
    
    if Pe_Pa < SUMMERFIELD_THRESHOLD:
        # Flow separation - significant thrust loss
        # Thrust loss increases as Pe/Pa decreases below threshold
        # Empirical: loss_factor ~ (Pe/Pa) / 0.4 clamped to [0.5, 1.0]
        loss_factor = Pe_Pa / SUMMERFIELD_THRESHOLD
        if loss_factor < 0.5:
            loss_factor = 0.5  # Minimum 50% thrust retained
        return 2, Pe_Pa, loss_factor, True  # Separated with side loads
    
    elif Pe_Pa < 0.6:
        # Approaching separation - mild overexpansion
        # Small thrust loss (5-15%)
        loss_factor = 0.9 + 0.1 * (Pe_Pa - 0.4) / 0.2
        return 1, Pe_Pa, loss_factor, False  # Overexpanded
    
    else:
        # Normal overexpanded operation (0.6 <= Pe/Pa < 1.0)
        # No significant thrust loss
        return 0, Pe_Pa, 1.0, False  # Attached


def calculate_flow_status(P_exit: float, P_ambient: float) -> FlowStatus:
    """
    Calculate nozzle flow status with physical thrust loss factor.
    
    This function returns a FlowStatus that AFFECTS PHYSICS, not just warnings.
    The thrust_loss_factor should be applied to calculated thrust.
    
    Args:
        P_exit: Nozzle exit pressure (Pa)
        P_ambient: Ambient pressure (Pa)
        
    Returns:
        FlowStatus with regime, thrust_loss_factor, and warnings
    """
    regime_code, Pe_Pa, loss_factor, side_load = calculate_flow_status_jit(
        P_exit, P_ambient
    )
    
    regime_map = {
        0: FlowRegime.ATTACHED,
        1: FlowRegime.OVEREXPANDED,
        2: FlowRegime.SEPARATED,
        3: FlowRegime.UNDEREXPANDED,
    }
    regime = regime_map[regime_code]
    
    # Generate message
    if regime == FlowRegime.ATTACHED:
        msg = f"Flow attached: Pe/Pa = {Pe_Pa:.3f}"
    elif regime == FlowRegime.OVEREXPANDED:
        msg = f"Mildly overexpanded: Pe/Pa = {Pe_Pa:.3f}, thrust factor = {loss_factor:.2f}"
    elif regime == FlowRegime.SEPARATED:
        msg = f"FLOW SEPARATION: Pe/Pa = {Pe_Pa:.3f}, thrust factor = {loss_factor:.2f}"
    else:  # UNDEREXPANDED
        msg = f"Underexpanded: Pe/Pa = {Pe_Pa:.3f}"
    
    return FlowStatus(
        regime=regime,
        Pe_Pa_ratio=Pe_Pa,
        thrust_loss_factor=loss_factor,
        side_load_warning=side_load,
        message=msg,
    )


# =============================================================================
# Engine State Dataclass
# =============================================================================

@dataclass
class EngineState:
    """
    Current engine state for a single time step.
    
    This is the output of RocketEngine.update_state() and contains
    all information needed by the 6-DOF integrator.
    """
    thrust: float              # Actual thrust after all corrections (N)
    isp: float                 # Actual specific impulse (s)
    mdot: float                # Mass flow rate (kg/s)
    c_star: float              # Characteristic velocity (m/s)
    c_f: float                 # Thrust coefficient
    P_exit: float              # Nozzle exit pressure (Pa)
    T_exit: float              # Nozzle exit temperature (K)
    flow_status: FlowStatus    # Flow regime and loss factors
    is_firing: bool            # True if engine is producing thrust


# =============================================================================
# RocketEngine Class
# =============================================================================

class RocketEngine:
    """
    Real-time rocket engine model with altitude correction.
    
    This class encapsulates engine geometry and thermodynamic properties,
    providing an update_state() method that returns current performance
    based on ambient conditions and throttle setting.
    
    Key Features:
        - Altitude-corrected thrust: F = mdot * Ve + (Pe - Pa) * Ae
        - Flow separation detection using Summerfield criterion
        - thrust_loss_factor applied to physics (not just warning)
        - Compatible with pre-computed lookup tables for fast interpolation
    
    Example:
        engine = RocketEngine(
            Pc_design=5e6,  # 5 MPa
            At=0.05,        # 0.05 m² throat
            epsilon=50.0,   # Expansion ratio 50:1
            T_chamber=3400, # K
            gamma=1.2,
            M_mol=22.0      # g/mol
        )
        
        # In simulation loop:
        state = engine.update_state(P_ambient=10000.0, throttle=1.0)
        F_thrust = state.thrust  # Corrected thrust
    """
    
    def __init__(
        self,
        Pc_design: float,
        At: float,
        epsilon: float,
        T_chamber: float,
        gamma: float,
        M_mol: float,
        eta_cstar: float = 0.95,
        eta_cf: float = 0.95,
    ):
        """
        Initialize rocket engine.
        
        Args:
            Pc_design: Design chamber pressure (Pa)
            At: Throat area (m²)
            epsilon: Nozzle expansion ratio (Ae/At)
            T_chamber: Chamber temperature (K)
            gamma: Ratio of specific heats (Cp/Cv)
            M_mol: Mean molecular weight of combustion products (g/mol)
            eta_cstar: Combustion efficiency (default 0.95)
            eta_cf: Nozzle efficiency (default 0.95)
        """
        self.Pc_design = Pc_design
        self.At = At
        self.epsilon = epsilon
        self.Ae = At * epsilon
        self.T_chamber = T_chamber
        self.gamma = gamma
        self.M_mol = M_mol
        self.eta_cstar = eta_cstar
        self.eta_cf = eta_cf
        
        # Derived properties
        self.R_specific = GAS_CONSTANT / (M_mol / 1000.0)  # J/(kg·K)
        
        # Calculate nominal exit Mach
        self.M_exit = solve_mach_from_area_ratio_supersonic(epsilon, gamma)
        
        # Calculate ideal C*
        self._c_star_ideal = calculate_c_star(gamma, self.R_specific, T_chamber)
        self.c_star = self._c_star_ideal * eta_cstar
        
        # Calculate nominal mdot at design Pc
        self._mdot_design = Pc_design * At / self.c_star
        
        # Burn time tracking
        self.burn_time: float = 0.0
        self._time_started: float = 0.0
        self._is_started: bool = False
    
    def set_burn_time(self, burn_time: float) -> None:
        """Set total burn time for the engine."""
        self.burn_time = burn_time
    
    def start(self, t: float) -> None:
        """Mark engine start time."""
        self._time_started = t
        self._is_started = True
    
    def is_firing_at(self, t: float) -> bool:
        """Check if engine is firing at time t."""
        if not self._is_started:
            return False
        elapsed = t - self._time_started
        return 0 <= elapsed < self.burn_time
    
    def update_state(
        self,
        P_ambient: float,
        throttle: float = 1.0,
        Pc_current: Optional[float] = None,
        t: Optional[float] = None,
    ) -> EngineState:
        """
        Update engine state for current conditions.
        
        This is the main interface for the 6-DOF simulator. It calculates
        thrust with altitude correction and applies flow separation losses.
        
        Args:
            P_ambient: Ambient pressure (Pa)
            throttle: Throttle setting 0.0-1.0 (default 1.0)
            Pc_current: Current chamber pressure (Pa), if None uses design
            t: Current simulation time (s), for burn time check
            
        Returns:
            EngineState with thrust, ISP, flow status, etc.
        """
        # Check if engine is firing
        is_firing = True
        if t is not None and self.burn_time > 0:
            is_firing = self.is_firing_at(t)
        
        if not is_firing or throttle <= 0:
            return EngineState(
                thrust=0.0,
                isp=0.0,
                mdot=0.0,
                c_star=self.c_star,
                c_f=0.0,
                P_exit=P_ambient,
                T_exit=self.T_chamber,
                flow_status=FlowStatus(
                    FlowRegime.ATTACHED, 1.0, 1.0, False, "Engine off"
                ),
                is_firing=False,
            )
        
        # Chamber pressure (with throttle)
        Pc = Pc_current if Pc_current is not None else self.Pc_design
        Pc = Pc * throttle
        
        # Exit conditions from isentropic relations
        T_exit, P_exit, _ = calculate_exit_conditions(
            self.gamma, self.T_chamber, Pc, self.M_exit
        )
        
        # Flow status with thrust loss factor
        flow_status = calculate_flow_status(P_exit, P_ambient)
        
        # Mass flow rate (varies with Pc)
        mdot = Pc * self.At / self.c_star
        
        # Thrust coefficient
        pressure_ratio = P_exit / Pc
        ambient_ratio = P_ambient / Pc
        Cf_ideal = calculate_thrust_coefficient(
            self.gamma, pressure_ratio, self.epsilon, ambient_ratio
        )
        
        # Apply efficiencies AND flow separation loss
        Cf = Cf_ideal * self.eta_cf * flow_status.thrust_loss_factor
        
        # Thrust
        thrust = Cf * Pc * self.At
        
        # Specific impulse
        isp = Cf * self.c_star / G0 if G0 > 0 else 0.0
        
        return EngineState(
            thrust=thrust,
            isp=isp,
            mdot=mdot,
            c_star=self.c_star,
            c_f=Cf,
            P_exit=P_exit,
            T_exit=T_exit,
            flow_status=flow_status,
            is_firing=True,
        )
    
    def get_vacuum_performance(self) -> Tuple[float, float, float]:
        """
        Get vacuum performance (for reference).
        
        Returns:
            Tuple of (thrust_vac, Isp_vac, mdot)
        """
        state = self.update_state(P_ambient=0.0, throttle=1.0)
        return state.thrust, state.isp, state.mdot
    
    def get_sea_level_performance(self) -> Tuple[float, float, float]:
        """
        Get sea level performance (for reference).
        
        Returns:
            Tuple of (thrust_sl, Isp_sl, mdot)
        """
        P_sl = 101325.0  # Pa
        state = self.update_state(P_ambient=P_sl, throttle=1.0)
        return state.thrust, state.isp, state.mdot
    
    def __repr__(self) -> str:
        F_vac, Isp_vac, mdot = self.get_vacuum_performance()
        return (f"RocketEngine(Pc={self.Pc_design/1e6:.1f}MPa, "
                f"ε={self.epsilon:.1f}, "
                f"F_vac={F_vac/1000:.1f}kN, "
                f"Isp_vac={Isp_vac:.0f}s)")
