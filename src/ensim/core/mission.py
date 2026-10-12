"""
Atmosphere and steady altitude-performance analysis.

Implements the U.S. Standard Atmosphere 1976 and engine altitude sweeps.
"""

from dataclasses import dataclass

import numpy as np

from ensim.utils.numba_support import jit

from .constants import EARTH_RADIUS_US76, G0

# U.S. Standard Atmosphere 1976, layers 0-7. Base altitudes are geopotential (m').
H_LAYERS = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
T_LAYERS = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
L_LAYERS = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])
P_LAYERS = np.array(
    [101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420, 0.3733836]
)

R_AIR = 8314.32 / 28.9644
"""Specific gas constant of air as defined by US76, J/(kg K)."""
GAMMA_AIR = 1.4

ATMOSPHERE_MODEL_TOP = 86_000.0
"""Geometric altitude where the US76 lower-atmosphere tables end, m.

Above it the temperature is held at the 86 km value and pressure decays
isothermally. That keeps density and sound speed continuous for trajectories
that leave the sensible atmosphere, but it is not the 1976 thermosphere model.
"""


@dataclass
class AtmosphereState:
    """Atmospheric conditions at an altitude."""

    altitude: float  # m, geometric
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m³
    speed_of_sound: float  # m/s
    viscosity: float  # Pa·s
    within_model: bool = True  # False above ATMOSPHERE_MODEL_TOP


@dataclass
class MissionPoint:
    """Performance at a single mission altitude."""

    altitude: float  # m
    atmosphere: AtmosphereState
    thrust: float  # N
    isp: float  # s
    exit_pressure: float  # Pa
    pressure_ratio: float  # Pe/Pa
    optimal_expansion: bool  # True if Pe ≈ Pa


@dataclass
class MissionProfile:
    """Complete mission performance envelope."""

    altitudes: np.ndarray  # m
    thrust: np.ndarray  # N
    isp: np.ndarray  # s
    pressure_ratio: np.ndarray  # Pe/Pa

    # Key altitudes
    optimal_altitude: float  # Where Pe = Pa
    max_thrust_altitude: float
    max_isp_altitude: float


@jit(nopython=True, cache=True)
def geopotential_altitude(z: float) -> float:
    """Convert geometric altitude z (m) to US76 geopotential altitude (m')."""
    return EARTH_RADIUS_US76 * z / (EARTH_RADIUS_US76 + z)


@jit(nopython=True, cache=True)
def atmosphere_properties(z: float) -> tuple[float, float, float, float]:
    """Temperature, pressure, density and speed of sound at geometric altitude z.

    Negative altitudes are clamped to sea level. Returns (T [K], P [Pa],
    rho [kg/m³], a [m/s]). Numba-compiled so the flight kernel can call it.
    """
    if z < 0.0:
        z = 0.0
    h = geopotential_altitude(z)
    layer = len(H_LAYERS) - 1
    for i in range(len(H_LAYERS) - 1):
        if h < H_LAYERS[i + 1]:
            layer = i
            break
    dh = h - H_LAYERS[layer]
    t_base = T_LAYERS[layer]
    lapse = L_LAYERS[layer]
    if lapse != 0.0:
        temperature = t_base + lapse * dh
        pressure = P_LAYERS[layer] * (t_base / temperature) ** (G0 / (R_AIR * lapse))
    else:
        temperature = t_base
        pressure = P_LAYERS[layer] * np.exp(-G0 * dh / (R_AIR * t_base))
    density = pressure / (R_AIR * temperature)
    sound_speed = np.sqrt(GAMMA_AIR * R_AIR * temperature)
    return temperature, pressure, density, sound_speed


def get_atmosphere(altitude: float) -> AtmosphereState:
    """Atmospheric state at a geometric altitude (m), US Standard Atmosphere 1976."""
    if not np.isfinite(altitude) or altitude < 0.0:
        raise ValueError("Atmospheric altitude must be finite and nonnegative")
    T, P, rho, a = atmosphere_properties(float(altitude))
    # Sutherland's law with the US76 constants
    mu = 1.458e-6 * T**1.5 / (T + 110.4)
    return AtmosphereState(
        altitude=altitude,
        temperature=T,
        pressure=P,
        density=rho,
        speed_of_sound=a,
        viscosity=mu,
        within_model=altitude <= ATMOSPHERE_MODEL_TOP,
    )


def calculate_altitude_performance(
    altitude: float,
    T_chamber: float,
    P_chamber: float,
    gamma: float,
    mean_mw: float,
    expansion_ratio: float,
    throat_area: float,
    eta_cstar: float = 1.0,
    eta_cf: float = 1.0,
    alpha_deg: float = 15.0,
) -> MissionPoint:
    """
    Calculate engine performance at a specific altitude.

    Args:
        altitude: Flight altitude (m)
        T_chamber: Chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        gamma: Ratio of specific heats
        mean_mw: Mean molecular weight (g/mol)
        expansion_ratio: Nozzle area ratio
        throat_area: Throat area (m²)
        eta_cstar, eta_cf: Efficiency factors
        alpha_deg: Nozzle half-angle (degrees)

    Returns:
        MissionPoint with performance at this altitude
    """
    from ensim.core.propulsion import NozzleConditions, calculate_performance

    atm = get_atmosphere(altitude)

    nozzle = NozzleConditions(
        area_ratio=expansion_ratio,
        chamber_pressure=P_chamber,
        ambient_pressure=atm.pressure,
        throat_area=throat_area,
    )

    perf = calculate_performance(
        T_chamber=T_chamber,
        P_chamber=P_chamber,
        gamma=gamma,
        mean_molecular_weight=mean_mw,
        nozzle=nozzle,
        eta_cstar=eta_cstar,
        eta_cf=eta_cf,
        alpha_deg=alpha_deg,
    )

    # Check if optimally expanded
    Pe_Pa = perf.exit_pressure / atm.pressure if atm.pressure > 0 else float("inf")
    optimal = 0.9 < Pe_Pa < 1.1

    return MissionPoint(
        altitude=altitude,
        atmosphere=atm,
        thrust=perf.thrust,
        isp=perf.isp,
        exit_pressure=perf.exit_pressure,
        pressure_ratio=Pe_Pa,
        optimal_expansion=optimal,
    )


def analyze_altitude_sweep(
    T_chamber: float,
    P_chamber: float,
    gamma: float,
    mean_mw: float,
    expansion_ratio: float,
    throat_area: float,
    eta_cstar: float = 1.0,
    eta_cf: float = 1.0,
    alpha_deg: float = 15.0,
    max_altitude: float = 100000.0,
    step_size: float = 1000.0,
) -> MissionProfile:
    """
    Evaluate steady engine performance over a prescribed altitude grid.

    Args:
        Engine and frozen-gas nozzle inputs
        max_altitude: Maximum altitude in the sweep (m)
        step_size: Altitude step (m)

    Returns:
        MissionProfile containing steady performance at each altitude
    """
    if not np.isfinite(max_altitude) or max_altitude <= 0.0:
        raise ValueError("Maximum sweep altitude must be positive and finite")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("Altitude step must be positive and finite")
    altitudes = np.arange(0, max_altitude + step_size, step_size)
    n = len(altitudes)

    thrust = np.zeros(n)
    isp = np.zeros(n)
    pressure_ratio = np.zeros(n)
    optimal_alt = None

    for i, alt in enumerate(altitudes):
        point = calculate_altitude_performance(
            altitude=alt,
            T_chamber=T_chamber,
            P_chamber=P_chamber,
            gamma=gamma,
            mean_mw=mean_mw,
            expansion_ratio=expansion_ratio,
            throat_area=throat_area,
            eta_cstar=eta_cstar,
            eta_cf=eta_cf,
            alpha_deg=alpha_deg,
        )

        thrust[i] = point.thrust
        isp[i] = point.isp
        pressure_ratio[i] = point.pressure_ratio

        if point.optimal_expansion and optimal_alt is None:
            optimal_alt = alt

    finite_pressure_ratio = np.maximum(pressure_ratio, np.finfo(float).tiny)
    matched_expansion_altitude = altitudes[np.argmin(np.abs(np.log(finite_pressure_ratio)))]
    max_thrust_alt = altitudes[np.argmax(thrust)]
    max_isp_alt = altitudes[np.argmax(isp)]

    return MissionProfile(
        altitudes=altitudes,
        thrust=thrust,
        isp=isp,
        pressure_ratio=pressure_ratio,
        optimal_altitude=optimal_alt if optimal_alt is not None else matched_expansion_altitude,
        max_thrust_altitude=max_thrust_alt,
        max_isp_altitude=max_isp_alt,
    )
