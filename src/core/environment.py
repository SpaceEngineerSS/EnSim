"""
Environmental Physics Module.

Implements wind effects, launch rail physics, and atmospheric
turbulence for realistic flight simulation.

References:
- OpenRocket Technical Documentation
- Mandell, Caporaso, Bengen "Topics in Advanced Model Rocketry"
"""

import numpy as np
from numba import jit
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WindProfile:
    """
    Wind model with altitude-dependent speed.
    
    Uses power law profile: V(h) = V_ref × (h/h_ref)^α
    """
    speed_ground: float = 0.0  # m/s at ground level
    direction: float = 0.0  # degrees from North (0 = from North)
    reference_height: float = 10.0  # m (standard measurement height)
    power_exponent: float = 0.143  # Typical for open terrain
    turbulence_intensity: float = 0.1  # Standard deviation / mean
    
    def get_wind_speed(self, altitude: float) -> float:
        """
        Get wind speed at altitude using power law.
        
        V(z) = V_ref × (z / z_ref)^α
        """
        if altitude <= 0:
            return 0.0
        if altitude < self.reference_height:
            # Linear interpolation near ground
            return self.speed_ground * (altitude / self.reference_height)
        
        return self.speed_ground * (altitude / self.reference_height) ** self.power_exponent
    
    def get_wind_vector(self, altitude: float) -> Tuple[float, float]:
        """
        Get wind velocity vector (Vx, Vy) at altitude.
        
        Returns components in launch frame:
        - Vx: East component
        - Vy: North component
        """
        speed = self.get_wind_speed(altitude)
        direction_rad = np.radians(self.direction)
        
        # Wind direction is where it comes FROM
        # Velocity is opposite (where it's going TO)
        Vx = -speed * np.sin(direction_rad)
        Vy = -speed * np.cos(direction_rad)
        
        return Vx, Vy


@dataclass
class LaunchRail:
    """
    Launch rail configuration.
    
    Constraints rocket motion until it clears the rail.
    """
    length: float = 1.5  # m (rail length)
    angle: float = 0.0  # degrees from vertical (launch angle)
    azimuth: float = 0.0  # degrees from North
    friction_coeff: float = 0.05  # Rail button friction
    
    def is_on_rail(self, altitude: float, rail_position: float) -> bool:
        """Check if rocket is still on rail."""
        # Rail position is distance along rail
        return rail_position < self.length
    
    def get_rail_direction(self) -> Tuple[float, float, float]:
        """
        Get unit vector along rail direction.
        
        Returns (dx, dy, dz) where z is up.
        """
        angle_rad = np.radians(self.angle)
        azimuth_rad = np.radians(self.azimuth)
        
        dz = np.cos(angle_rad)  # Vertical component
        horizontal = np.sin(angle_rad)
        dx = horizontal * np.sin(azimuth_rad)  # East
        dy = horizontal * np.cos(azimuth_rad)  # North
        
        return dx, dy, dz
    
    def get_rail_exit_velocity(self, thrust: float, mass: float, 
                               drag: float = 0.0) -> float:
        """
        Calculate velocity at rail exit.
        
        Uses energy method: v² = 2 × (F_net/m) × L
        Accounts for friction and drag.
        """
        g = 9.80665
        
        # Net force along rail (approximate, ignoring angle for simplicity)
        angle_rad = np.radians(self.angle)
        F_gravity = mass * g * np.cos(angle_rad)  # Component along rail
        F_friction = self.friction_coeff * mass * g * np.sin(angle_rad)
        
        F_net = thrust - F_gravity - F_friction - drag
        
        if F_net <= 0:
            return 0.0
        
        # v² = 2aL where a = F_net/m
        v_exit = np.sqrt(2 * (F_net / mass) * self.length)
        return v_exit


@dataclass
class LaunchConditions:
    """Complete launch environment configuration."""
    altitude_asl: float = 0.0  # Launch site altitude (m ASL)
    wind: WindProfile = None
    rail: LaunchRail = None
    temperature_offset: float = 0.0  # K (deviation from ISA)
    
    def __post_init__(self):
        if self.wind is None:
            self.wind = WindProfile()
        if self.rail is None:
            self.rail = LaunchRail()


@jit(nopython=True, cache=True)
def calculate_angle_of_attack(
    velocity_x: float,
    velocity_y: float, 
    velocity_z: float,
    wind_x: float,
    wind_y: float,
    pitch: float,
    yaw: float
) -> float:
    """
    Calculate angle of attack due to wind and rocket orientation.
    
    α = arctan(V_relative_normal / V_relative_axial)
    
    Args:
        velocity_x, y, z: Rocket velocity in ground frame
        wind_x, y: Wind velocity components
        pitch, yaw: Rocket orientation (radians)
        
    Returns:
        Angle of attack in radians
    """
    # Relative velocity (rocket velocity - wind)
    rel_x = velocity_x - wind_x
    rel_y = velocity_y - wind_y
    rel_z = velocity_z  # Assume no vertical wind
    
    # Rocket axis direction (unit vector)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    # Axis pointing up and slightly inclined
    ax_x = sp * sy
    ax_y = sp * cy
    ax_z = cp
    
    # Axial velocity component
    V_axial = rel_x * ax_x + rel_y * ax_y + rel_z * ax_z
    
    # Total relative speed
    V_total = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
    
    if V_total < 0.1:
        return 0.0
    
    # Normal velocity component
    V_normal = np.sqrt(V_total**2 - V_axial**2)
    
    # Angle of attack
    alpha = np.arctan2(V_normal, abs(V_axial))
    
    return alpha


@jit(nopython=True, cache=True)
def calculate_normal_force(
    rho: float,
    velocity: float,
    cn_alpha: float,
    alpha: float,
    reference_area: float
) -> float:
    """
    Calculate normal (lift-like) force due to angle of attack.
    
    F_N = 0.5 × ρ × V² × CN_α × α × A_ref
    
    This force causes "weathercocking" - rocket turning into wind.
    
    Args:
        rho: Air density (kg/m³)
        velocity: Relative airspeed (m/s)
        cn_alpha: Normal force coefficient slope (per radian)
        alpha: Angle of attack (radians)
        reference_area: Reference area (m²)
        
    Returns:
        Normal force (N)
    """
    F_normal = 0.5 * rho * velocity**2 * cn_alpha * alpha * reference_area
    return F_normal


@jit(nopython=True, cache=True)
def calculate_weathercock_moment(
    normal_force: float,
    cp_position: float,
    cg_position: float
) -> float:
    """
    Calculate restoring moment due to normal force.
    
    M = F_N × (X_cp - X_cg)
    
    Positive moment rotates nose into wind (stable).
    
    Args:
        normal_force: Normal force (N)
        cp_position: CP distance from nose (m)
        cg_position: CG distance from nose (m)
        
    Returns:
        Moment (N·m)
    """
    moment_arm = cp_position - cg_position
    return normal_force * moment_arm


def get_turbulent_wind(wind: WindProfile, altitude: float, time: float) -> Tuple[float, float]:
    """
    Get instantaneous wind with turbulence.
    
    Uses simple sinusoidal gust model for deterministic behavior.
    """
    base_vx, base_vy = wind.get_wind_vector(altitude)
    
    if wind.turbulence_intensity <= 0:
        return base_vx, base_vy
    
    # Simple gust model (deterministic for reproducibility)
    gust_amplitude = wind.speed_ground * wind.turbulence_intensity
    gust_freq = 0.5  # Hz
    
    gust = gust_amplitude * np.sin(2 * np.pi * gust_freq * time)
    
    return base_vx + gust, base_vy


def create_standard_conditions(
    wind_speed: float = 0.0,
    wind_direction: float = 0.0,
    rail_length: float = 1.5,
    launch_altitude: float = 0.0
) -> LaunchConditions:
    """Create standard launch conditions."""
    return LaunchConditions(
        altitude_asl=launch_altitude,
        wind=WindProfile(
            speed_ground=wind_speed,
            direction=wind_direction
        ),
        rail=LaunchRail(length=rail_length)
    )
