"""
Recovery Systems Module.

Implements parachute physics, descent calculations,
and landing drift estimation.

References:
- OpenRocket Technical Documentation
- NAR Safety Code descent rate requirements
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numba import jit


class DeployTrigger(Enum):
    """Parachute deployment trigger types."""
    AT_APOGEE = "apogee"
    AT_ALTITUDE = "altitude"


class FlightPhase(Enum):
    """Rocket flight phases for state machine."""
    PAD = "pad"
    ASCENT = "ascent"
    COAST = "coast"
    DROGUE_DESCENT = "drogue_descent"
    MAIN_DESCENT = "main_descent"
    GROUND = "ground"


@dataclass
class Parachute:
    """
    Parachute configuration.

    Typical Cd values:
    - Round (hemispherical): 1.5
    - Cross/cruciform: 1.0
    - Elliptical: 1.8
    """
    name: str = "Main"
    diameter: float = 1.0  # m
    cd: float = 1.5  # Drag coefficient
    deploy_trigger: DeployTrigger = DeployTrigger.AT_APOGEE
    deploy_altitude: float = 300.0  # m (for altitude trigger)

    @property
    def area(self) -> float:
        """Parachute canopy area (m²)."""
        return np.pi * (self.diameter / 2) ** 2

    @property
    def cda(self) -> float:
        """Effective drag area Cd × A (m²)."""
        return self.cd * self.area

    def get_descent_rate(self, mass: float, rho: float = 1.225) -> float:
        """
        Calculate terminal descent velocity.

        V = sqrt(2mg / (ρ × Cd × A))

        Args:
            mass: Total suspended mass (kg)
            rho: Air density (kg/m³)

        Returns:
            Descent rate (m/s), positive downward
        """
        if self.cda <= 0:
            return 100.0  # Very fast, no chute

        g = 9.80665
        v_descent = np.sqrt(2 * mass * g / (rho * self.cda))
        return v_descent

    def is_safe_descent(self, mass: float, rho: float = 1.225) -> tuple[float, bool]:
        """
        Check if descent rate is safe.

        NAR Safety Code: < 6.1 m/s (20 ft/s) for competition
        Hobby standard: < 9.1 m/s (30 ft/s)

        Returns:
            Tuple of (descent_rate, is_safe)
        """
        v = self.get_descent_rate(mass, rho)
        is_safe = v < 6.1
        return v, is_safe


@dataclass
class RecoverySystem:
    """
    Complete recovery system configuration.

    Supports single deployment (main only) or dual deployment
    (drogue at apogee + main at altitude).
    """
    main_chute: Parachute = field(default_factory=Parachute)
    drogue_chute: Parachute | None = None
    dual_deploy: bool = False

    def get_active_chute(self, altitude: float, at_apogee: bool) -> Parachute | None:
        """
        Get the currently active parachute based on flight state.

        Args:
            altitude: Current altitude (m)
            at_apogee: True if at or past apogee

        Returns:
            Active parachute or None if not deployed
        """
        if not at_apogee:
            return None

        if self.dual_deploy and self.drogue_chute is not None:
            # Dual deployment: drogue first, then main
            if altitude > self.main_chute.deploy_altitude:
                return self.drogue_chute
            else:
                return self.main_chute
        else:
            # Single deployment
            if self.main_chute.deploy_trigger == DeployTrigger.AT_APOGEE or altitude <= self.main_chute.deploy_altitude:
                return self.main_chute

        return None


@jit(nopython=True, cache=True)
def calculate_descent_velocity(
    mass: float,
    rho: float,
    chute_cda: float,
    current_velocity: float,
    dt: float
) -> float:
    """
    Calculate velocity during parachute descent.

    Uses quasi-steady approximation with gradual deployment.

    Args:
        mass: Total mass (kg)
        rho: Air density (kg/m³)
        chute_cda: Parachute Cd×A (m²)
        current_velocity: Current vertical velocity (m/s, negative = down)
        dt: Time step (s)

    Returns:
        New velocity (m/s)
    """
    g = 9.80665

    # Terminal velocity under chute
    v_terminal = -np.sqrt(2 * mass * g / (rho * chute_cda))

    # Exponential approach to terminal (simulates opening shock)
    tau = 0.5  # Time constant for opening
    alpha = 1 - np.exp(-dt / tau)

    new_velocity = current_velocity + alpha * (v_terminal - current_velocity)

    return new_velocity


def calculate_drift(
    descent_time: float,
    wind_speed: float,
    wind_direction: float = 0.0
) -> tuple[float, float]:
    """
    Calculate landing drift from launch point.

    Drift = Wind_speed × Descent_time

    Args:
        descent_time: Time from apogee to ground (s)
        wind_speed: Average wind speed (m/s)
        wind_direction: Wind direction in degrees from North

    Returns:
        Tuple of (drift_distance, drift_direction)
    """
    drift_distance = wind_speed * descent_time
    return drift_distance, wind_direction


def estimate_descent(
    apogee: float,
    mass: float,
    chute: Parachute,
    rho_avg: float = 1.1
) -> dict:
    """
    Estimate descent parameters.

    Args:
        apogee: Maximum altitude (m)
        mass: Rocket mass (kg)
        chute: Parachute configuration
        rho_avg: Average air density during descent

    Returns:
        Dict with descent_rate, descent_time, kinetic_energy
    """
    v_descent = chute.get_descent_rate(mass, rho_avg)
    descent_time = apogee / v_descent if v_descent > 0 else 0
    kinetic_energy = 0.5 * mass * v_descent ** 2

    return {
        'descent_rate': v_descent,
        'descent_time': descent_time,
        'kinetic_energy': kinetic_energy,
        'is_safe': v_descent < 6.1
    }


def size_parachute(
    mass: float,
    target_descent_rate: float = 5.0,
    cd: float = 1.5,
    rho: float = 1.225
) -> float:
    """
    Calculate required parachute diameter for target descent rate.

    D = 2 × sqrt(2mg / (ρ × Cd × π × V²))

    Args:
        mass: Suspended mass (kg)
        target_descent_rate: Desired descent velocity (m/s)
        cd: Parachute drag coefficient
        rho: Air density (kg/m³)

    Returns:
        Required diameter (m)
    """
    g = 9.80665
    area = 2 * mass * g / (rho * cd * target_descent_rate ** 2)
    diameter = 2 * np.sqrt(area / np.pi)
    return diameter
