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

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from numba import jit
from numpy.typing import NDArray


class CoolingType(Enum):
    """Engine cooling methods."""
    REGENERATIVE = auto()      # Coolant flows through wall channels
    ABLATIVE = auto()          # Sacrificial liner
    FILM_COOLING = auto()      # Fuel film along wall
    RADIATION = auto()         # Radiation-cooled (high-temp materials)
    TRANSPIRATION = auto()     # Porous wall with coolant injection


class CoolantType(Enum):
    """Common propellant coolants."""
    RP1 = auto()              # Kerosene (RP-1)
    LH2 = auto()              # Liquid hydrogen
    LOX = auto()              # Liquid oxygen
    LCH4 = auto()             # Liquid methane
    N2H4 = auto()             # Hydrazine
    MMH = auto()              # Monomethylhydrazine
    WATER = auto()            # Water (for testing)


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


# Coolant property database (at typical operating conditions)
COOLANT_DATABASE: dict[CoolantType, CoolantProperties] = {
    CoolantType.RP1: CoolantProperties(
        density=820.0,
        specific_heat=2010.0,
        thermal_conductivity=0.13,
        dynamic_viscosity=1.5e-3,
        boiling_point=489.0,
        critical_temp=658.0,
        critical_pressure=2.1e6
    ),
    CoolantType.LH2: CoolantProperties(
        density=70.8,
        specific_heat=9680.0,
        thermal_conductivity=0.1,
        dynamic_viscosity=1.3e-5,
        boiling_point=20.3,
        critical_temp=33.2,
        critical_pressure=1.3e6
    ),
    CoolantType.LOX: CoolantProperties(
        density=1141.0,
        specific_heat=920.0,
        thermal_conductivity=0.15,
        dynamic_viscosity=1.9e-4,
        boiling_point=90.2,
        critical_temp=154.6,
        critical_pressure=5.04e6
    ),
    CoolantType.LCH4: CoolantProperties(
        density=422.0,
        specific_heat=3480.0,
        thermal_conductivity=0.19,
        dynamic_viscosity=1.1e-4,
        boiling_point=111.7,
        critical_temp=190.6,
        critical_pressure=4.6e6
    ),
    CoolantType.WATER: CoolantProperties(
        density=998.0,
        specific_heat=4186.0,
        thermal_conductivity=0.6,
        dynamic_viscosity=1.0e-3,
        boiling_point=373.15,
        critical_temp=647.1,
        critical_pressure=22.06e6
    ),
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
    axial_position: float          # Position along nozzle (m)
    wall_temp_gas_side: float      # Gas-side wall temperature (K)
    wall_temp_coolant_side: float  # Coolant-side wall temperature (K)
    coolant_temp: float            # Bulk coolant temperature (K)
    heat_flux: float               # Heat flux (W/m²)
    coolant_velocity: float        # Coolant velocity (m/s)
    coolant_pressure: float        # Coolant pressure (Pa)
    margin_to_melting: float       # Temperature margin to wall melting (K)
    margin_to_boiling: float       # Coolant margin to boiling (K)


@dataclass
class ThermalProfileResult:
    """Results of nozzle thermal profile calculation."""
    x_position: NDArray[np.float64]      # Axial positions (m)
    heat_flux: NDArray[np.float64]       # Heat flux at each station (W/m²)
    wall_temp_gas: NDArray[np.float64]   # Gas-side wall temp (K)
    wall_temp_coolant: NDArray[np.float64]  # Coolant-side wall temp (K)
    max_wall_temp: float                 # Maximum wall temperature (K)
    max_heat_flux: float                 # Maximum heat flux (W/m²)
    is_safe: bool                        # True if within material limits
    critical_x: float                    # Position of max temp (m)


@dataclass
class CoolingSystemDesign:
    """Complete regenerative cooling system design."""
    channels: CoolingChannel
    coolant: CoolantType
    coolant_inlet_temp: float      # K
    coolant_inlet_pressure: float  # Pa
    coolant_mass_flow: float       # kg/s
    wall_material: str             # Material name
    wall_thermal_conductivity: float  # W/(m·K)
    wall_melting_point: float      # K


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
    local_diameter: float
) -> float:
    """
    Calculate gas-side heat transfer coefficient using Bartz correlation.

    The Bartz equation is the standard correlation for rocket nozzle
    heat transfer, based on pipe flow correlations with modifications
    for acceleration and compressibility effects.

    h_g = (0.026/D_t^0.2) × (μ^0.2 × Cp / Pr^0.6) × (ρ_c × c*)^0.8 × (A_t/A)^0.9 × σ

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

    Returns:
        Gas-side heat transfer coefficient (W/(m²·K))

    Reference:
        Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of
        Rocket Nozzle Convective Heat Transfer Coefficients"
        Jet Propulsion, Vol. 27, No. 1, pp. 49-51.
    """
    if D_throat <= 0 or c_star <= 0:
        return 0.0

    # Estimate Cp from gamma (for combustion gases)
    # Cp = gamma * R / (gamma - 1) / MW
    R_spec = 8314.46 / 22.0  # Assume MW ~22 g/mol
    Cp = gamma * R_spec / (gamma - 1.0)

    # Reference density (chamber conditions)
    rho_c = P_chamber / (R_spec * T_chamber)

    # Sigma correction factor for acceleration effects
    # σ = 1 / [0.5(Tw/Tc)(1 + (γ-1)/2 M²) + 0.5]^0.68 × [(1 + (γ-1)/2 M²)]^0.12
    # Simplified: assume Tw/Tc ~ 0.3 for cooled walls, M varies along nozzle

    # Local Mach number estimate from area ratio
    # Using approximation: M ≈ sqrt((2/(γ-1))*(area_ratio^((γ-1)/γ) - 1))
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    if area_ratio >= 1.0:
        # Supersonic estimate
        M_local = np.sqrt(2.0 / gm1 * (area_ratio ** (gm1 / gamma) - 1.0))
        M_local = max(1.0, min(10.0, M_local))  # Clamp
    else:
        M_local = 0.5  # Subsonic

    Tw_Tc = 0.3  # Typical for cooled wall
    temp_ratio = Tw_Tc * (1.0 + gm1 / 2.0 * M_local**2) + 0.5
    sigma = 1.0 / (temp_ratio ** 0.68 * (1.0 + gm1 / 2.0 * M_local**2) ** 0.12)

    # Bartz correlation
    h_g = (0.026 / D_throat**0.2) * \
          (mu_ref**0.2 * Cp / Pr**0.6) * \
          (rho_c * c_star)**0.8 * \
          (1.0 / area_ratio)**0.9 * \
          sigma

    return h_g


@jit(nopython=True, cache=True)
def dittus_boelter_coefficient(
    Re: float,
    Pr: float,
    k: float,
    D_h: float,
    heating: bool = True
) -> float:
    """
    Calculate coolant-side heat transfer coefficient using Dittus-Boelter.

    Nu = 0.023 × Re^0.8 × Pr^n
    where n = 0.4 for heating (coolant), n = 0.3 for cooling

    Args:
        Re: Reynolds number
        Pr: Prandtl number
        k: Thermal conductivity (W/(m·K))
        D_h: Hydraulic diameter (m)
        heating: True if fluid is being heated (default for coolant)

    Returns:
        Heat transfer coefficient (W/(m²·K))
    """
    if Re < 2300:
        # Laminar: Nu = 3.66 for constant wall temp
        Nu = 3.66
    else:
        # Turbulent: Dittus-Boelter
        n = 0.4 if heating else 0.3
        Nu = 0.023 * Re**0.8 * Pr**n

    return Nu * k / D_h if D_h > 0 else 0.0


@jit(nopython=True, cache=True)
def calculate_wall_temperatures(
    q_flux: float,
    h_gas: float,
    h_coolant: float,
    T_gas: float,
    T_coolant: float,
    wall_thickness: float,
    k_wall: float
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
    R_coolant = 1.0 / h_coolant if h_coolant > 0 else 1e6

    R_total = R_gas + R_wall + R_coolant

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
    num_stations: int = 50
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
    results = []

    # Get coolant properties
    coolant_props = COOLANT_DATABASE.get(design.coolant)
    if coolant_props is None:
        raise ValueError(f"Unknown coolant: {design.coolant}")

    # Unpack chamber conditions
    T_chamber = chamber_conditions.get('T_chamber', 3500.0)
    P_chamber = chamber_conditions.get('P_chamber', 7e6)
    gamma = chamber_conditions.get('gamma', 1.2)
    c_star = chamber_conditions.get('c_star', 1800.0)

    # Find throat location (minimum diameter)
    min_dia = min(d for _, d in nozzle_profile)
    D_throat = min_dia

    # Coolant state tracking
    T_coolant = design.coolant_inlet_temp
    P_coolant = design.coolant_inlet_pressure
    m_dot = design.coolant_mass_flow

    # Coolant velocity
    v_coolant = m_dot / (coolant_props.density * design.channels.flow_area)

    # Coolant Reynolds and Prandtl numbers
    Re_coolant = (coolant_props.density * v_coolant *
                  design.channels.hydraulic_diameter / coolant_props.dynamic_viscosity)
    Pr_coolant = (coolant_props.dynamic_viscosity * coolant_props.specific_heat /
                  coolant_props.thermal_conductivity)

    # Coolant-side heat transfer coefficient
    h_coolant = dittus_boelter_coefficient(
        Re_coolant, Pr_coolant,
        coolant_props.thermal_conductivity,
        design.channels.hydraulic_diameter
    )

    # Analyze each station
    x_positions = np.linspace(0, design.channels.length, num_stations)

    for i, x in enumerate(x_positions):
        # Interpolate local diameter from profile
        local_dia = np.interp(x, [p[0] for p in nozzle_profile],
                              [p[1] for p in nozzle_profile])

        area_ratio = (local_dia / D_throat) ** 2

        # Gas-side heat transfer coefficient (Bartz)
        # Reference viscosity estimate
        mu_ref = 5e-5  # Typical for hot combustion gases

        h_gas = bartz_heat_transfer_coefficient(
            D_throat, P_chamber, c_star, T_chamber, gamma,
            Pr=0.7,  # Typical for combustion products
            mu_ref=mu_ref,
            area_ratio=area_ratio,
            local_diameter=local_dia
        )

        # Recovery temperature
        Pr_gas = 0.7
        r = Pr_gas ** 0.33  # Recovery factor
        T_recovery = T_chamber  # Simplified; actual depends on Mach

        # Initial heat flux estimate
        R_total = 1.0/h_gas + design.channels.wall_thickness/design.wall_thermal_conductivity + 1.0/h_coolant
        q_flux = (T_recovery - T_coolant) / R_total

        # Calculate wall temperatures
        T_wall_gas, T_wall_coolant = calculate_wall_temperatures(
            q_flux, h_gas, h_coolant,
            T_recovery, T_coolant,
            design.channels.wall_thickness,
            design.wall_thermal_conductivity
        )

        # Update coolant temperature
        if i > 0:
            dx = x - x_positions[i-1]
            # Heat absorbed by coolant
            circumference = np.pi * local_dia
            q_total = q_flux * circumference * dx
            dT_coolant = q_total / (m_dot * coolant_props.specific_heat)
            T_coolant += dT_coolant

            # Pressure drop (friction)
            f = 0.046 * Re_coolant ** (-0.2)  # Turbulent friction factor
            dp = f * (dx / design.channels.hydraulic_diameter) * \
                 0.5 * coolant_props.density * v_coolant**2
            P_coolant -= dp

        # Calculate margins
        margin_to_melting = design.wall_melting_point - T_wall_gas
        margin_to_boiling = coolant_props.boiling_point - T_coolant

        results.append(ThermalAnalysisResult(
            axial_position=x,
            wall_temp_gas_side=T_wall_gas,
            wall_temp_coolant_side=T_wall_coolant,
            coolant_temp=T_coolant,
            heat_flux=q_flux,
            coolant_velocity=v_coolant,
            coolant_pressure=P_coolant,
            margin_to_melting=margin_to_melting,
            margin_to_boiling=margin_to_boiling
        ))

    return results


def design_cooling_channels(
    thrust: float,
    chamber_pressure: float,
    chamber_temp: float,
    coolant: CoolantType,
    coolant_mass_flow_fraction: float = 1.0,
    wall_material: str = "Inconel 718",
    safety_factor: float = 1.5
) -> CoolingSystemDesign:
    """
    Preliminary design of regenerative cooling channels.

    Uses empirical correlations and design guidelines to size
    the cooling system for given engine parameters.

    Args:
        thrust: Engine thrust (N)
        chamber_pressure: Chamber pressure (Pa)
        chamber_temp: Chamber temperature (K)
        coolant: Coolant type
        coolant_mass_flow_fraction: Fraction of fuel used for cooling
        wall_material: Wall material name
        safety_factor: Safety factor on wall temperature

    Returns:
        CoolingSystemDesign with preliminary channel sizing
    """
    from src.core.propulsion import GAS_CONSTANT

    # Material properties database
    material_data = {
        "Inconel 718": {"k": 11.4, "T_melt": 1533.0},
        "OFHC Copper": {"k": 385.0, "T_melt": 1356.0},
        "GRCop-84": {"k": 300.0, "T_melt": 1356.0},
        "Haynes 230": {"k": 8.9, "T_melt": 1573.0},
        "Monel 400": {"k": 21.8, "T_melt": 1573.0},
    }

    mat = material_data.get(wall_material, material_data["Inconel 718"])
    coolant_props = COOLANT_DATABASE.get(coolant, COOLANT_DATABASE[CoolantType.RP1])

    # Estimate engine dimensions
    Cf = 1.8  # Typical thrust coefficient
    A_throat = thrust / (chamber_pressure * Cf)
    D_throat = 2 * np.sqrt(A_throat / np.pi)

    # Chamber dimensions (L* ~ 1m for typical engines)
    L_star = 1.0  # Characteristic length
    A_chamber = A_throat * 3.0  # Contraction ratio ~3
    L_chamber = L_star * A_throat / A_chamber

    # Nozzle length (conical, 15° half-angle)
    expansion_ratio = 40.0  # Typical for vacuum engine
    D_exit = D_throat * np.sqrt(expansion_ratio)
    L_nozzle = (D_exit - D_throat) / (2 * np.tan(np.radians(15)))

    total_length = L_chamber + L_nozzle

    # Mass flow estimate
    G0 = 9.80665
    Isp_est = 320  # Typical Isp
    m_dot_total = thrust / (Isp_est * G0)
    m_dot_coolant = m_dot_total * coolant_mass_flow_fraction

    # Estimate required heat transfer area
    # Peak heat flux at throat: q ~ 10-50 MW/m²
    q_peak = 20e6  # W/m²

    # Channel sizing for manageable temperature rise
    dT_coolant_allowed = coolant_props.boiling_point - 200.0 - 250.0  # Stay below boiling
    if dT_coolant_allowed < 50:
        dT_coolant_allowed = 100.0

    # Total heat to be absorbed (rough estimate)
    circumference_avg = np.pi * (D_throat + D_exit) / 2
    Q_total = q_peak * 0.3 * circumference_avg * total_length  # 30% of peak avg

    # Required coolant flow
    m_dot_required = Q_total / (coolant_props.specific_heat * dT_coolant_allowed)
    m_dot_coolant = max(m_dot_coolant, m_dot_required)

    # Channel dimensions
    # Target velocity: 15-30 m/s for good heat transfer
    v_target = 20.0
    A_flow = m_dot_coolant / (coolant_props.density * v_target)

    # Number of channels (typical: 100-200 for medium engines)
    num_channels = int(np.pi * D_throat * 1000)  # ~1 channel per mm circumference
    num_channels = max(50, min(500, num_channels))

    A_channel = A_flow / num_channels
    channel_height = 0.003  # 3 mm typical
    channel_width = A_channel / channel_height

    # Clamp to reasonable dimensions
    channel_width = max(0.001, min(0.01, channel_width))
    channel_height = max(0.002, min(0.008, channel_height))

    # Wall thickness (structural + thermal considerations)
    # t_min = P * r / σ_allow for pressure containment
    sigma_allow = 200e6  # Pa (yield/SF)
    t_pressure = chamber_pressure * D_throat / 2 / sigma_allow
    t_thermal = 0.001  # 1 mm minimum for thermal resistance
    wall_thickness = max(t_pressure, t_thermal, 0.0015)

    channels = CoolingChannel(
        width=channel_width,
        height=channel_height,
        wall_thickness=wall_thickness,
        land_width=channel_width * 0.5,  # 50% land width
        num_channels=num_channels,
        length=total_length
    )

    return CoolingSystemDesign(
        channels=channels,
        coolant=coolant,
        coolant_inlet_temp=coolant_props.boiling_point - 50,  # Subcooled
        coolant_inlet_pressure=chamber_pressure * 1.3,  # 30% margin
        coolant_mass_flow=m_dot_coolant,
        wall_material=wall_material,
        wall_thermal_conductivity=mat["k"],
        wall_melting_point=mat["T_melt"] / safety_factor
    )


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
    num_stations: int = 50
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

    Returns:
        ThermalProfileResult with temperature and heat flux arrays
    """
    # Nozzle geometry
    # Assume conical nozzle with 15° half-angle
    D_throat = throat_diameter
    D_exit = D_throat * np.sqrt(expansion_ratio)
    
    # Convergent section (contraction ratio ~3)
    D_chamber = D_throat * np.sqrt(3.0)
    L_conv = (D_chamber - D_throat) / (2 * np.tan(np.radians(30)))
    
    # Divergent section
    L_div = (D_exit - D_throat) / (2 * np.tan(np.radians(15)))
    
    # Total length with throat at x=0
    x_positions = np.linspace(-L_conv, L_div, num_stations)
    
    # Calculate diameter at each position
    diameters = np.zeros(num_stations)
    for i, x in enumerate(x_positions):
        if x < 0:
            # Convergent section
            diameters[i] = D_throat + 2 * abs(x) * np.tan(np.radians(30))
        else:
            # Divergent section
            diameters[i] = D_throat + 2 * x * np.tan(np.radians(15))
    
    # Area ratios
    area_ratios = (diameters / D_throat) ** 2
    
    # Reference viscosity for combustion gases
    mu_ref = 5e-5  # Pa·s
    
    # Calculate heat flux and wall temperatures at each station
    heat_flux = np.zeros(num_stations)
    wall_temp_gas = np.zeros(num_stations)
    wall_temp_coolant = np.zeros(num_stations)
    
    for i in range(num_stations):
        # Gas-side heat transfer coefficient (Bartz)
        h_gas = bartz_heat_transfer_coefficient(
            D_throat=D_throat,
            P_chamber=P_chamber,
            c_star=c_star,
            T_chamber=T_chamber,
            gamma=gamma,
            Pr=0.7,
            mu_ref=mu_ref,
            area_ratio=area_ratios[i],
            local_diameter=diameters[i]
        )
        
        # Total thermal resistance
        R_gas = 1.0 / h_gas if h_gas > 0 else 1e6
        R_wall = wall_thickness / wall_conductivity
        R_coolant = 1.0 / coolant_htc if coolant_htc > 0 else 1e6
        R_total = R_gas + R_wall + R_coolant
        
        # Heat flux from gas to coolant
        q = (T_chamber - coolant_temp) / R_total
        heat_flux[i] = q
        
        # Wall temperatures
        T_wall_gas = T_chamber - q * R_gas
        T_wall_cool = T_wall_gas - q * R_wall
        
        wall_temp_gas[i] = T_wall_gas
        wall_temp_coolant[i] = T_wall_cool
    
    # Find maximum values
    max_temp = np.max(wall_temp_gas)
    max_flux = np.max(heat_flux)
    max_temp_idx = np.argmax(wall_temp_gas)
    critical_x = x_positions[max_temp_idx]
    
    # Safety check
    is_safe = max_temp < material_limit
    
    return ThermalProfileResult(
        x_position=x_positions,
        heat_flux=heat_flux,
        wall_temp_gas=wall_temp_gas,
        wall_temp_coolant=wall_temp_coolant,
        max_wall_temp=max_temp,
        max_heat_flux=max_flux,
        is_safe=is_safe,
        critical_x=critical_x
    )
