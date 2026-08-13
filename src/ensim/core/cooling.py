"""
Regenerative cooling thermal analysis module.

Provides detailed thermal modeling for rocket engine cooling:
- Regenerative cooling channel design
- Heat transfer calculations (gas-side and coolant-side)
- Wall temperature prediction
- Thermal stress analysis

References:
    - Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant Rocket Engines"
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed., Ch. 8
    - Bartz correlation for gas-side heat transfer
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numba import jit
from numpy.typing import NDArray


class CoolingType(Enum):
    """Engine cooling methods."""

    REGENERATIVE = auto()  # Coolant flows through wall channels
    ABLATIVE = auto()  # Sacrificial liner
    FILM_COOLING = auto()  # Fuel film along wall
    RADIATION = auto()  # Radiation-cooled (high-temp materials)
    TRANSPIRATION = auto()  # Porous wall with coolant injection


class CoolantType(Enum):
    """Common propellant coolants."""

    RP1 = auto()  # Kerosene (RP-1)
    LH2 = auto()  # Liquid hydrogen
    LOX = auto()  # Liquid oxygen
    LCH4 = auto()  # Liquid methane
    N2H4 = auto()  # Hydrazine
    MMH = auto()  # Monomethylhydrazine
    WATER = auto()  # Water (for testing)


@dataclass
class CoolantProperties:
    """
    Thermophysical properties of coolants.

    Properties are typically at saturation conditions.

    Attributes:
        density: Density (kg/m³)
        specific_heat: Specific heat (J/(kg·K))
        thermal_conductivity: Thermal conductivity (W/(m·K))
        dynamic_viscosity: Dynamic viscosity (Pa·s)
        boiling_point: Boiling point at 1 atm (K)
        critical_temp: Critical temperature (K)
        critical_pressure: Critical pressure (Pa)
    """

    density: float
    specific_heat: float
    thermal_conductivity: float
    dynamic_viscosity: float
    boiling_point: float
    critical_temp: float
    critical_pressure: float
    reference_inlet_temperature: float = 293.15


# Coolant property database (at typical operating conditions)
COOLANT_DATABASE: dict[CoolantType, CoolantProperties] = {
    CoolantType.RP1: CoolantProperties(
        density=820.0,
        specific_heat=2010.0,
        thermal_conductivity=0.13,
        dynamic_viscosity=1.5e-3,
        boiling_point=489.0,
        critical_temp=658.0,
        critical_pressure=2.1e6,
        reference_inlet_temperature=293.15,
    ),
    CoolantType.LH2: CoolantProperties(
        density=70.8,
        specific_heat=9680.0,
        thermal_conductivity=0.1,
        dynamic_viscosity=1.3e-5,
        boiling_point=20.3,
        critical_temp=33.2,
        critical_pressure=1.3e6,
        reference_inlet_temperature=20.0,
    ),
    CoolantType.LOX: CoolantProperties(
        density=1141.0,
        specific_heat=920.0,
        thermal_conductivity=0.15,
        dynamic_viscosity=1.9e-4,
        boiling_point=90.2,
        critical_temp=154.6,
        critical_pressure=5.04e6,
        reference_inlet_temperature=90.0,
    ),
    CoolantType.LCH4: CoolantProperties(
        density=422.0,
        specific_heat=3480.0,
        thermal_conductivity=0.19,
        dynamic_viscosity=1.1e-4,
        boiling_point=111.7,
        critical_temp=190.6,
        critical_pressure=4.6e6,
        reference_inlet_temperature=110.0,
    ),
    CoolantType.WATER: CoolantProperties(
        density=998.0,
        specific_heat=4186.0,
        thermal_conductivity=0.6,
        dynamic_viscosity=1.0e-3,
        boiling_point=373.15,
        critical_temp=647.1,
        critical_pressure=22.06e6,
        reference_inlet_temperature=293.15,
    ),
}

COOLING_MATERIALS: dict[str, tuple[float, float]] = {
    "Inconel 718": (11.4, 1533.0),
    "OFHC Copper": (385.0, 1356.0),
    "GRCop-84": (300.0, 1356.0),
    "Haynes 230": (8.9, 1573.0),
    "Monel 400": (21.8, 1573.0),
}


@dataclass
class CoolingChannel:
    """
    Definition of a cooling channel.

    Attributes:
        width: Channel width (m)
        height: Channel height (m)
        wall_thickness: Inner wall thickness (m)
        land_width: Width between channels (m)
        num_channels: Number of channels around circumference
        length: Total channel length (m)
    """

    width: float
    height: float
    wall_thickness: float
    land_width: float
    num_channels: int
    length: float

    @property
    def hydraulic_diameter(self) -> float:
        """Calculate hydraulic diameter Dh = 4A/P."""
        area = self.width * self.height
        perimeter = 2 * (self.width + self.height)
        return 4 * area / perimeter if perimeter > 0 else 0.0

    @property
    def flow_area(self) -> float:
        """Total flow area of all channels."""
        return self.width * self.height * self.num_channels

    @property
    def surface_area(self) -> float:
        """Total heat transfer surface area."""
        # Area of inner surface exposed to hot gas
        perimeter = 2 * (self.width + self.height)
        return perimeter * self.length * self.num_channels


@dataclass
class ThermalAnalysisResult:
    """Results of thermal analysis at a station."""

    axial_position: float  # Position along nozzle (m)
    wall_temp_gas_side: float  # Gas-side wall temperature (K)
    wall_temp_coolant_side: float  # Coolant-side wall temperature (K)
    coolant_temp: float  # Bulk coolant temperature (K)
    heat_flux: float  # Heat flux (W/m²)
    coolant_velocity: float  # Coolant velocity (m/s)
    coolant_pressure: float  # Coolant pressure (Pa)
    margin_to_melting: float  # Difference from nominal wall melting point (K)
    margin_to_critical_temperature: float  # Difference from coolant critical T (K)


@dataclass
class ThermalProfileResult:
    """Results of nozzle thermal profile calculation."""

    x_position: NDArray[np.float64]  # Axial positions (m)
    heat_flux: NDArray[np.float64]  # Heat flux at each station (W/m²)
    wall_temp_gas: NDArray[np.float64]  # Gas-side wall temp (K)
    wall_temp_coolant: NDArray[np.float64]  # Coolant-side wall temp (K)
    max_wall_temp: float  # Maximum wall temperature (K)
    max_heat_flux: float  # Maximum heat flux (W/m²)
    within_material_limit: bool
    critical_x: float  # Position of max temp (m)


@dataclass
class CoolingSystemDesign:
    """Complete regenerative cooling system design."""

    channels: CoolingChannel
    coolant: CoolantType
    coolant_inlet_temp: float  # K
    coolant_inlet_pressure: float  # Pa
    coolant_mass_flow: float  # kg/s
    wall_material: str  # Material name
    wall_thermal_conductivity: float  # W/(m·K)
    wall_melting_point: float  # K
    coolant_flow_from_exit: bool = True


# =============================================================================
# Heat Transfer Correlations
# =============================================================================


@jit(nopython=True, cache=True)
def bartz_heat_transfer_coefficient(
    D_throat: float,
    P_chamber: float,
    c_star: float,
    T_chamber: float,
    gamma: float,
    Pr: float,
    mu_ref: float,
    area_ratio: float,
    local_diameter: float,
    throat_radius_of_curvature: float | None = None,
    wall_temperature: float | None = None,
    molecular_weight: float = 22.0,
    supersonic: bool = True,
) -> float:
    """
    Calculate gas-side heat transfer coefficient using Bartz correlation.

    The Bartz equation is the standard correlation for rocket nozzle
    heat transfer, based on pipe flow correlations with modifications
    for acceleration and compressibility effects.

    h_g = (0.026/D_t^0.2)(μ^0.2 Cp/Pr^0.6)(Pc/c*)^0.8
          × (D_t/r_c)^0.1 × (A_t/A)^0.9 × σ

    Args:
        D_throat: Throat diameter (m)
        P_chamber: Chamber pressure (Pa)
        c_star: Characteristic velocity (m/s)
        T_chamber: Chamber temperature (K)
        gamma: Specific heat ratio
        Pr: Prandtl number
        mu_ref: Reference dynamic viscosity (Pa·s)
        area_ratio: Local area ratio (A/At)
        local_diameter: Local nozzle diameter (m)
        throat_radius_of_curvature: Throat contour radius (m); defaults to D_t
        wall_temperature: Gas-side wall temperature for the property correction (K)
        molecular_weight: Gas molecular weight (kg/kmol)
        supersonic: Select the supersonic area-Mach branch

    Returns:
        Gas-side heat transfer coefficient (W/(m²·K))

    Reference:
        Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of
        Rocket Nozzle Convective Heat Transfer Coefficients"
        Jet Propulsion, Vol. 27, No. 1, pp. 49-51.
    """
    if (
        D_throat <= 0.0
        or local_diameter <= 0.0
        or P_chamber <= 0.0
        or c_star <= 0.0
        or T_chamber <= 0.0
        or not 1.0 < gamma < 2.0
        or Pr <= 0.0
        or mu_ref <= 0.0
        or molecular_weight <= 0.0
    ):
        return 0.0

    geometric_area_ratio = (local_diameter / D_throat) ** 2
    if abs(geometric_area_ratio - area_ratio) / max(area_ratio, 1.0) > 1e-6:
        area_ratio = geometric_area_ratio
    area_ratio = max(1.0, area_ratio)

    R_spec = 8314.46261815324 / molecular_weight
    Cp = gamma * R_spec / (gamma - 1.0)
    M_local = _mach_from_area_ratio(area_ratio, gamma, supersonic)
    temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * M_local**2
    wall_ratio = (
        wall_temperature / T_chamber
        if wall_temperature is not None and wall_temperature > 0.0
        else 0.3
    )
    sigma = (0.5 * wall_ratio * temperature_factor + 0.5) ** -0.68 * temperature_factor**-0.12
    curvature = throat_radius_of_curvature if throat_radius_of_curvature is not None else D_throat
    if curvature <= 0.0:
        return 0.0

    return (
        0.026
        / D_throat**0.2
        * (mu_ref**0.2 * Cp / Pr**0.6)
        * (P_chamber / c_star) ** 0.8
        * (D_throat / curvature) ** 0.1
        * area_ratio**-0.9
        * sigma
    )


@jit(nopython=True, cache=True)
def _mach_from_area_ratio(area_ratio: float, gamma: float, supersonic: bool) -> float:
    """Solve the calorically-perfect isentropic area-Mach relation."""
    if area_ratio <= 1.0 + 1e-12:
        return 1.0

    def area_function(mach: float) -> float:
        factor = 2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
        exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return factor**exponent / mach

    lower = 1.0 + 1e-10 if supersonic else 1e-8
    upper = 50.0 if supersonic else 1.0 - 1e-10
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        value = area_function(midpoint)
        if supersonic:
            if value < area_ratio:
                lower = midpoint
            else:
                upper = midpoint
        elif value > area_ratio:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


@jit(nopython=True, cache=True)
def coolant_heat_transfer_coefficient(
    reynolds: float,
    prandtl: float,
    conductivity: float,
    hydraulic_diameter: float,
) -> float:
    """Return a single-phase internal-flow coefficient using Gnielinski."""
    if reynolds <= 0.0 or prandtl <= 0.0 or conductivity <= 0.0 or hydraulic_diameter <= 0.0:
        return 0.0
    laminar_nusselt = 3.66

    def gnielinski(reynolds_number: float) -> float:
        friction = (0.79 * np.log(reynolds_number) - 1.64) ** -2
        return (
            (friction / 8.0)
            * (reynolds_number - 1000.0)
            * prandtl
            / (1.0 + 12.7 * np.sqrt(friction / 8.0) * (prandtl ** (2.0 / 3.0) - 1.0))
        )

    if reynolds <= 2300.0:
        nusselt = laminar_nusselt
    elif reynolds < 3000.0:
        fraction = (reynolds - 2300.0) / 700.0
        nusselt = (1.0 - fraction) * laminar_nusselt + fraction * gnielinski(3000.0)
    else:
        nusselt = gnielinski(reynolds)
    return nusselt * conductivity / hydraulic_diameter


@jit(nopython=True, cache=True)
def smooth_channel_darcy_friction_factor(reynolds: float) -> float:
    """Return the Darcy friction factor for a hydraulically smooth channel."""
    if reynolds <= 0.0:
        return 0.0
    if reynolds < 2300.0:
        return 64.0 / reynolds
    return (-1.8 * np.log10(6.9 / reynolds)) ** -2


@jit(nopython=True, cache=True)
def calculate_wall_temperatures(
    q_flux: float,
    h_gas: float,
    h_coolant: float,
    T_gas: float,
    T_coolant: float,
    wall_thickness: float,
    k_wall: float,
) -> tuple[float, float]:
    """
    Calculate wall temperatures for given heat flux.

    Uses 1D steady-state conduction through wall.

    Args:
        q_flux: Heat flux (W/m²)
        h_gas: Gas-side heat transfer coefficient (W/(m²·K))
        h_coolant: Coolant-side coefficient (W/(m²·K))
        T_gas: Gas recovery temperature (K)
        T_coolant: Bulk coolant temperature (K)
        wall_thickness: Wall thickness (m)
        k_wall: Wall thermal conductivity (W/(m·K))

    Returns:
        Tuple of (T_wall_gas_side, T_wall_coolant_side) in K
    """
    # Total thermal resistance
    R_gas = 1.0 / h_gas if h_gas > 0 else 1e6
    R_wall = wall_thickness / k_wall if k_wall > 0 else 0.0
    # Heat flux from overall temperature difference
    # q = (T_gas - T_coolant) / R_total
    # Given q, find wall temps

    # Gas-side wall temp
    T_wall_gas = T_gas - q_flux * R_gas

    # Coolant-side wall temp
    T_wall_coolant = T_wall_gas - q_flux * R_wall

    return T_wall_gas, T_wall_coolant


# =============================================================================
# Cooling System Analysis
# =============================================================================


def analyze_cooling_system(
    design: CoolingSystemDesign,
    nozzle_profile: list[tuple[float, float]],  # List of (x, diameter)
    chamber_conditions: dict,
    num_stations: int = 50,
) -> list[ThermalAnalysisResult]:
    """
    Perform thermal analysis along the nozzle.

    Calculates wall temperatures, heat fluxes, and coolant conditions
    at multiple stations along the nozzle.

    Args:
        design: Cooling system design parameters
        nozzle_profile: List of (axial_position, diameter) tuples
        chamber_conditions: Dict with T_chamber, P_chamber, gamma, c_star
        num_stations: Number of analysis stations

    Returns:
        List of ThermalAnalysisResult at each station
    """
    if num_stations < 2:
        raise ValueError("At least two thermal stations are required")
    if len(nozzle_profile) < 2:
        raise ValueError("Nozzle profile requires at least two points")
    channel_values = (
        design.channels.width,
        design.channels.height,
        design.channels.wall_thickness,
        design.channels.land_width,
        design.channels.length,
    )
    if any(value <= 0.0 or not np.isfinite(value) for value in channel_values):
        raise ValueError("All cooling-channel dimensions must be finite and positive")
    if design.channels.num_channels <= 0:
        raise ValueError("Channel count must be positive")
    if design.coolant_mass_flow <= 0.0 or not np.isfinite(design.coolant_mass_flow):
        raise ValueError("Coolant mass flow must be finite and positive")
    if design.coolant_inlet_temp <= 0.0 or not np.isfinite(design.coolant_inlet_temp):
        raise ValueError("Coolant inlet temperature must be finite and positive")
    if design.coolant_inlet_pressure <= 0.0 or not np.isfinite(design.coolant_inlet_pressure):
        raise ValueError("Coolant inlet pressure must be finite and positive")
    if design.channels.hydraulic_diameter <= 0.0:
        raise ValueError("Channel hydraulic diameter must be positive")
    if design.wall_thermal_conductivity <= 0.0:
        raise ValueError("Wall thermal conductivity must be positive")
    if design.wall_melting_point <= 0.0:
        raise ValueError("Wall melting point must be positive")
    profile = sorted(nozzle_profile)
    if any(
        not np.isfinite(position) or not np.isfinite(diameter) or diameter <= 0.0
        for position, diameter in profile
    ):
        raise ValueError("Nozzle-profile values must be finite and diameters positive")
    if any(right[0] <= left[0] for left, right in zip(profile, profile[1:], strict=False)):
        raise ValueError("Nozzle-profile axial positions must be unique")

    results = []

    # Get coolant properties
    coolant_props = COOLANT_DATABASE.get(design.coolant)
    if coolant_props is None:
        raise ValueError(f"Unknown coolant: {design.coolant}")

    required_conditions = {"T_chamber", "P_chamber", "gamma", "c_star", "molecular_weight"}
    missing = sorted(required_conditions.difference(chamber_conditions))
    if missing:
        raise ValueError(f"Missing chamber conditions: {', '.join(missing)}")
    T_chamber = float(chamber_conditions["T_chamber"])
    P_chamber = float(chamber_conditions["P_chamber"])
    gamma = float(chamber_conditions["gamma"])
    c_star = float(chamber_conditions["c_star"])
    gas_prandtl = chamber_conditions.get("prandtl", 0.7)
    gas_viscosity = chamber_conditions.get("gas_viscosity", 5e-5)
    molecular_weight = float(chamber_conditions["molecular_weight"])
    throat_radius = chamber_conditions.get("throat_radius_of_curvature")
    if (
        T_chamber <= 0.0
        or P_chamber <= 0.0
        or c_star <= 0.0
        or molecular_weight <= 0.0
        or not 1.0 < gamma < 2.0
        or gas_prandtl <= 0.0
        or gas_viscosity <= 0.0
    ):
        raise ValueError("Chamber and gas-property inputs are outside the model domain")

    # Find throat location (minimum diameter)
    min_dia = min(d for _, d in profile)
    D_throat = min_dia
    throat_x = min(profile, key=lambda point: point[1])[0]

    # Coolant state tracking
    T_coolant = design.coolant_inlet_temp
    P_coolant = design.coolant_inlet_pressure
    m_dot = design.coolant_mass_flow

    # Coolant velocity
    v_coolant = m_dot / (coolant_props.density * design.channels.flow_area)

    # Coolant Reynolds and Prandtl numbers
    Re_coolant = (
        coolant_props.density
        * v_coolant
        * design.channels.hydraulic_diameter
        / coolant_props.dynamic_viscosity
    )
    Pr_coolant = (
        coolant_props.dynamic_viscosity
        * coolant_props.specific_heat
        / coolant_props.thermal_conductivity
    )

    # Coolant-side heat transfer coefficient
    h_coolant = coolant_heat_transfer_coefficient(
        Re_coolant,
        Pr_coolant,
        coolant_props.thermal_conductivity,
        design.channels.hydraulic_diameter,
    )

    # Analyze each station
    x_positions = np.linspace(profile[0][0], profile[-1][0], num_stations)
    if design.coolant_flow_from_exit:
        x_positions = x_positions[::-1]

    for i, x in enumerate(x_positions):
        # Interpolate local diameter from profile
        local_dia = np.interp(x, [p[0] for p in profile], [p[1] for p in profile])

        area_ratio = (local_dia / D_throat) ** 2

        is_supersonic = x >= throat_x
        local_mach = _mach_from_area_ratio(area_ratio, gamma, is_supersonic)
        temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * local_mach**2
        recovery_factor = gas_prandtl ** (1.0 / 3.0)
        T_static = T_chamber / temperature_factor
        T_recovery = T_static * (1.0 + recovery_factor * 0.5 * (gamma - 1.0) * local_mach**2)

        T_wall_estimate = 0.3 * T_chamber
        for _ in range(5):
            h_gas = bartz_heat_transfer_coefficient(
                D_throat,
                P_chamber,
                c_star,
                T_chamber,
                gamma,
                Pr=gas_prandtl,
                mu_ref=gas_viscosity,
                area_ratio=area_ratio,
                local_diameter=local_dia,
                throat_radius_of_curvature=throat_radius,
                wall_temperature=T_wall_estimate,
                molecular_weight=molecular_weight,
                supersonic=is_supersonic,
            )
            R_total = (
                1.0 / h_gas
                + design.channels.wall_thickness / design.wall_thermal_conductivity
                + 1.0 / h_coolant
            )
            q_flux = (T_recovery - T_coolant) / R_total
            T_wall_estimate = T_recovery - q_flux / h_gas

        # Calculate wall temperatures
        T_wall_gas, T_wall_coolant = calculate_wall_temperatures(
            q_flux,
            h_gas,
            h_coolant,
            T_recovery,
            T_coolant,
            design.channels.wall_thickness,
            design.wall_thermal_conductivity,
        )

        # Update coolant temperature
        # Calculate margins
        margin_to_melting = design.wall_melting_point - T_wall_gas
        margin_to_critical_temperature = coolant_props.critical_temp - T_coolant

        results.append(
            ThermalAnalysisResult(
                axial_position=x,
                wall_temp_gas_side=T_wall_gas,
                wall_temp_coolant_side=T_wall_coolant,
                coolant_temp=T_coolant,
                heat_flux=q_flux,
                coolant_velocity=v_coolant,
                coolant_pressure=P_coolant,
                margin_to_melting=margin_to_melting,
                margin_to_critical_temperature=margin_to_critical_temperature,
            )
        )

        if i < len(x_positions) - 1:
            dx = abs(x_positions[i + 1] - x)
            circumference = np.pi * local_dia
            q_total = q_flux * circumference * dx
            T_coolant += q_total / (m_dot * coolant_props.specific_heat)
            friction = smooth_channel_darcy_friction_factor(Re_coolant)
            P_coolant -= (
                friction
                * dx
                / design.channels.hydraulic_diameter
                * 0.5
                * coolant_props.density
                * v_coolant**2
            )
            if P_coolant <= 0.0:
                raise ValueError("Predicted coolant pressure became non-positive")

    return sorted(results, key=lambda station: station.axial_position)


def calculate_thermal_profile(
    T_chamber: float,
    P_chamber: float,
    c_star: float,
    gamma: float,
    throat_diameter: float,
    expansion_ratio: float,
    wall_thickness: float,
    wall_conductivity: float,
    coolant_temp: float,
    coolant_htc: float,
    material_limit: float,
    num_stations: int = 50,
    gas_prandtl: float = 0.7,
    gas_viscosity: float = 5e-5,
    molecular_weight: float = 22.0,
    throat_radius_of_curvature: float | None = None,
    contraction_ratio: float = 3.0,
    convergent_half_angle_deg: float = 30.0,
    divergent_half_angle_deg: float = 15.0,
) -> ThermalProfileResult:
    """
    Calculate thermal profile along a nozzle contour.

    Uses Bartz correlation for gas-side heat transfer and 1D conduction
    through the wall to determine temperature distribution.

    Args:
        T_chamber: Chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        c_star: Characteristic velocity (m/s)
        gamma: Specific heat ratio
        throat_diameter: Throat diameter (m)
        expansion_ratio: Nozzle expansion ratio (Ae/At)
        wall_thickness: Wall thickness (m)
        wall_conductivity: Wall thermal conductivity (W/(m·K))
        coolant_temp: Coolant temperature (K)
        coolant_htc: Coolant-side heat transfer coefficient (W/(m²·K))
        material_limit: Maximum allowable wall temperature (K)
        num_stations: Number of analysis stations
        contraction_ratio: Chamber-to-throat area ratio
        convergent_half_angle_deg: Conical convergent half-angle
        divergent_half_angle_deg: Conical divergent half-angle

    Returns:
        ThermalProfileResult with temperature and heat flux arrays
    """
    if (
        T_chamber <= 0.0
        or P_chamber <= 0.0
        or c_star <= 0.0
        or throat_diameter <= 0.0
        or expansion_ratio < 1.0
        or wall_thickness <= 0.0
        or wall_conductivity <= 0.0
        or coolant_temp <= 0.0
        or coolant_htc <= 0.0
        or material_limit <= 0.0
        or contraction_ratio <= 1.0
        or not 0.0 < convergent_half_angle_deg < 90.0
        or not 0.0 < divergent_half_angle_deg < 90.0
    ):
        raise ValueError("Thermal-profile inputs must be physically positive")
    if not 1.0 < gamma < 2.0 or not gas_prandtl > 0.0:
        raise ValueError("Gamma and gas Prandtl number are outside valid bounds")
    if num_stations < 3:
        raise ValueError("At least three thermal stations are required")

    # Nozzle geometry
    # Assume conical nozzle with 15° half-angle
    D_throat = throat_diameter
    D_exit = D_throat * np.sqrt(expansion_ratio)

    # Convergent section (contraction ratio ~3)
    D_chamber = D_throat * np.sqrt(contraction_ratio)
    L_conv = (D_chamber - D_throat) / (2 * np.tan(np.radians(convergent_half_angle_deg)))

    # Divergent section
    L_div = (D_exit - D_throat) / (2 * np.tan(np.radians(divergent_half_angle_deg)))

    # Total length with throat at x=0
    x_positions = np.linspace(-L_conv, L_div, num_stations)

    # Calculate diameter at each position
    diameters = np.zeros(num_stations)
    for i, x in enumerate(x_positions):
        if x < 0:
            # Convergent section
            diameters[i] = D_throat + 2 * abs(x) * np.tan(np.radians(convergent_half_angle_deg))
        else:
            # Divergent section
            diameters[i] = D_throat + 2 * x * np.tan(np.radians(divergent_half_angle_deg))

    # Area ratios
    area_ratios = (diameters / D_throat) ** 2

    # Calculate heat flux and wall temperatures at each station
    heat_flux = np.zeros(num_stations)
    wall_temp_gas = np.zeros(num_stations)
    wall_temp_coolant = np.zeros(num_stations)

    for i in range(num_stations):
        is_supersonic = x_positions[i] >= 0.0
        local_mach = _mach_from_area_ratio(area_ratios[i], gamma, is_supersonic)
        temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * local_mach**2
        recovery_factor = gas_prandtl ** (1.0 / 3.0)
        recovery_temperature = (
            T_chamber
            / temperature_factor
            * (1.0 + recovery_factor * 0.5 * (gamma - 1.0) * local_mach**2)
        )
        wall_estimate = 0.3 * T_chamber
        for _ in range(5):
            h_gas = bartz_heat_transfer_coefficient(
                D_throat=D_throat,
                P_chamber=P_chamber,
                c_star=c_star,
                T_chamber=T_chamber,
                gamma=gamma,
                Pr=gas_prandtl,
                mu_ref=gas_viscosity,
                area_ratio=area_ratios[i],
                local_diameter=diameters[i],
                throat_radius_of_curvature=throat_radius_of_curvature,
                wall_temperature=wall_estimate,
                molecular_weight=molecular_weight,
                supersonic=is_supersonic,
            )
            R_gas = 1.0 / h_gas
            R_wall = wall_thickness / wall_conductivity
            R_coolant = 1.0 / coolant_htc
            R_total = R_gas + R_wall + R_coolant
            q = (recovery_temperature - coolant_temp) / R_total
            wall_estimate = recovery_temperature - q * R_gas
        heat_flux[i] = q

        # Wall temperatures
        T_wall_gas = recovery_temperature - q * R_gas
        T_wall_cool = T_wall_gas - q * R_wall

        wall_temp_gas[i] = T_wall_gas
        wall_temp_coolant[i] = T_wall_cool

    # Find maximum values
    max_temp = np.max(wall_temp_gas)
    max_flux = np.max(heat_flux)
    max_temp_idx = np.argmax(wall_temp_gas)
    critical_x = x_positions[max_temp_idx]

    # Safety check
    within_material_limit = max_temp < material_limit

    return ThermalProfileResult(
        x_position=x_positions,
        heat_flux=heat_flux,
        wall_temp_gas=wall_temp_gas,
        wall_temp_coolant=wall_temp_coolant,
        max_wall_temp=max_temp,
        max_heat_flux=max_flux,
        within_material_limit=within_material_limit,
        critical_x=critical_x,
    )
