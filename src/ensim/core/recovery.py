"""Parachute sizing and steady descent estimates.

The 6-DOF simulation applies parachute drag itself; these helpers answer the
quick design questions (how big, how fast, how long).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .constants import G0


class DeployTrigger(Enum):
    """Parachute deployment trigger types."""

    AT_APOGEE = "apogee"
    AT_ALTITUDE = "altitude"


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
        if self.cda <= 0.0 or mass <= 0.0 or rho <= 0.0:
            raise ValueError("Mass, density and parachute drag area must be positive")

        v_descent = np.sqrt(2 * mass * G0 / (rho * self.cda))
        return v_descent


def estimate_descent(apogee: float, mass: float, chute: Parachute, rho_avg: float = 1.1) -> dict:
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
    if apogee < 0.0:
        raise ValueError("Apogee cannot be negative")
    v_descent = chute.get_descent_rate(mass, rho_avg)
    descent_time = apogee / v_descent
    kinetic_energy = 0.5 * mass * v_descent**2

    return {
        "descent_rate": v_descent,
        "descent_time": descent_time,
        "kinetic_energy": kinetic_energy,
    }


def size_parachute(
    mass: float, target_descent_rate: float = 5.0, cd: float = 1.5, rho: float = 1.225
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
    if mass <= 0.0 or target_descent_rate <= 0.0 or cd <= 0.0 or rho <= 0.0:
        raise ValueError("Sizing inputs must be positive")
    area = 2 * mass * G0 / (rho * cd * target_descent_rate**2)
    diameter = 2 * np.sqrt(area / np.pi)
    return diameter
