"""
Mission Analysis Module - Atmosphere Model and Flight Envelope.

Implements US Standard Atmosphere 1976 and trajectory performance analysis.
"""

from dataclasses import dataclass

import numpy as np
from numba import jit

# US Standard Atmosphere 1976 Constants
# Geopotential altitude layers (m)
H_LAYERS = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
# Base temperatures (K)
T_LAYERS = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
# Lapse rates (K/m)
L_LAYERS = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])
# Base pressures (Pa)
P_LAYERS = np.array([101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642, 0.3734])

# Physical constants
G0 = 9.80665  # m/s² standard gravity
R_AIR = 287.058  # J/(kg·K) gas constant for air
GAMMA_AIR = 1.4  # ratio of specific heats for air


@dataclass
class AtmosphereState:
    """Atmospheric conditions at an altitude."""
    altitude: float  # m
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m³
    speed_of_sound: float  # m/s
    viscosity: float  # Pa·s


@dataclass
class MissionPoint:
    """Performance at a single mission altitude."""
    altitude: float  # m
    atmosphere: AtmosphereState
    thrust: float  # N
    isp: float  # s
    exit_pressure: float  # Pa
    flow_separated: bool
    separation_margin: float  # Pe/Pa ratio
    optimal_expansion: bool  # True if Pe ≈ Pa


@dataclass
class MissionProfile:
    """Complete mission performance envelope."""
    altitudes: np.ndarray  # m
    thrust: np.ndarray  # N
    isp: np.ndarray  # s
    pressure_ratio: np.ndarray  # Pe/Pa
    flow_status: list[str]  # 'attached', 'warning', 'separated'

    # Key altitudes
    optimal_altitude: float  # Where Pe = Pa
    separation_altitude: float | None  # Where separation begins
    max_thrust_altitude: float
    max_isp_altitude: float


@jit(nopython=True, cache=True)
def _atmosphere_layer(h: float) -> int:
    """Find atmosphere layer index for given altitude."""
    for i in range(len(H_LAYERS) - 1):
        if h < H_LAYERS[i + 1]:
            return i
    return len(H_LAYERS) - 2


@jit(nopython=True, cache=True)
def _atmosphere_pressure(h: float) -> tuple[float, float]:
    """
    Calculate pressure and temperature at altitude using US Std Atm 1976.

    Returns: (pressure, temperature)
    """
    if h < 0:
        h = 0
    if h > 84852:
        h = 84852

    layer = _atmosphere_layer(h)

    h_base = H_LAYERS[layer]
    T_base = T_LAYERS[layer]
    L = L_LAYERS[layer]
    P_base = P_LAYERS[layer]

    dh = h - h_base

    if abs(L) > 1e-10:
        # Non-isothermal layer
        T = T_base + L * dh
        P = P_base * (T / T_base) ** (-G0 / (R_AIR * L))
    else:
        # Isothermal layer
        T = T_base
        P = P_base * np.exp(-G0 * dh / (R_AIR * T))

    return P, T


def get_atmosphere(altitude: float) -> AtmosphereState:
    """
    Get complete atmospheric state at altitude.

    Args:
        altitude: Geometric altitude (m)

    Returns:
        AtmosphereState object
    """
    P, T = _atmosphere_pressure(altitude)

    rho = P / (R_AIR * T)
    a = np.sqrt(GAMMA_AIR * R_AIR * T)

    # Sutherland's law for viscosity
    mu = 1.458e-6 * T**1.5 / (T + 110.4)

    return AtmosphereState(
        altitude=altitude,
        temperature=T,
        pressure=P,
        density=rho,
        speed_of_sound=a,
        viscosity=mu
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
    alpha_deg: float = 15.0
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
    from src.core.propulsion import NozzleConditions, calculate_performance, check_flow_separation

    atm = get_atmosphere(altitude)

    nozzle = NozzleConditions(
        area_ratio=expansion_ratio,
        chamber_pressure=P_chamber,
        ambient_pressure=atm.pressure,
        throat_area=throat_area
    )

    perf = calculate_performance(
        T_chamber=T_chamber,
        P_chamber=P_chamber,
        gamma=gamma,
        mean_molecular_weight=mean_mw,
        nozzle=nozzle,
        eta_cstar=eta_cstar,
        eta_cf=eta_cf,
        alpha_deg=alpha_deg
    )

    # Check flow separation
    separated, margin, _ = check_flow_separation(perf.exit_pressure, atm.pressure)

    # Check if optimally expanded
    Pe_Pa = perf.exit_pressure / atm.pressure if atm.pressure > 0 else float('inf')
    optimal = 0.9 < Pe_Pa < 1.1

    return MissionPoint(
        altitude=altitude,
        atmosphere=atm,
        thrust=perf.thrust,
        isp=perf.isp,
        exit_pressure=perf.exit_pressure,
        flow_separated=separated,
        separation_margin=Pe_Pa,
        optimal_expansion=optimal
    )


def simulate_ascent(
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
    step_size: float = 1000.0
) -> MissionProfile:
    """
    Simulate engine performance throughout ascent.

    Args:
        Various engine parameters...
        max_altitude: Maximum altitude to simulate (m)
        step_size: Altitude step (m)

    Returns:
        MissionProfile with full trajectory performance
    """
    altitudes = np.arange(0, max_altitude + step_size, step_size)
    n = len(altitudes)

    thrust = np.zeros(n)
    isp = np.zeros(n)
    pressure_ratio = np.zeros(n)
    flow_status = []

    separation_alt = None
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
            alpha_deg=alpha_deg
        )

        thrust[i] = point.thrust
        isp[i] = point.isp
        pressure_ratio[i] = point.separation_margin

        if point.flow_separated:
            flow_status.append('separated')
            if separation_alt is None:
                separation_alt = alt
        elif point.separation_margin < 0.6:
            flow_status.append('warning')
        else:
            flow_status.append('attached')

        if point.optimal_expansion and optimal_alt is None:
            optimal_alt = alt

    # Find peak values
    max_thrust_alt = altitudes[np.argmax(thrust)]
    max_isp_alt = altitudes[np.argmax(isp)]

    return MissionProfile(
        altitudes=altitudes,
        thrust=thrust,
        isp=isp,
        pressure_ratio=pressure_ratio,
        flow_status=flow_status,
        optimal_altitude=optimal_alt if optimal_alt else max_altitude,
        separation_altitude=separation_alt,
        max_thrust_altitude=max_thrust_alt,
        max_isp_altitude=max_isp_alt
    )


def get_atmosphere_table(max_alt: float = 100000, step: float = 5000) -> dict:
    """
    Generate atmosphere property table.

    Returns dict with arrays of altitude, P, T, rho, a.
    """
    alts = np.arange(0, max_alt + step, step)
    n = len(alts)

    P = np.zeros(n)
    T = np.zeros(n)
    rho = np.zeros(n)
    a = np.zeros(n)

    for i, alt in enumerate(alts):
        atm = get_atmosphere(alt)
        P[i] = atm.pressure
        T[i] = atm.temperature
        rho[i] = atm.density
        a[i] = atm.speed_of_sound

    return {
        'altitude': alts,
        'pressure': P,
        'temperature': T,
        'density': rho,
        'speed_of_sound': a
    }
