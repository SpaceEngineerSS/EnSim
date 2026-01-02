"""
Combustion Instability Analysis Module.

Analyzes combustion chambers for acoustic instabilities including:
- Longitudinal (axial) modes
- Transverse (tangential and radial) modes
- Low-frequency chugging

References:
    1. Harrje & Reardon, "Liquid Propellant Rocket Combustion
       Instability", NASA SP-194, 1972.
    2. Yang & Anderson, "Liquid Rocket Engine Combustion
       Instability", AIAA Progress in Astronautics, 1995.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ModeType(Enum):
    """Type of acoustic mode."""
    LONGITUDINAL = "longitudinal"
    TANGENTIAL = "tangential"
    RADIAL = "radial"
    COMBINED = "combined"


@dataclass
class AcousticMode:
    """An acoustic mode of the combustion chamber."""
    mode_type: ModeType
    mode_indices: tuple[int, ...]  # (n,) for longitudinal, (m, n) for transverse
    frequency: float  # Hz
    wavelength: float  # m
    description: str

    @property
    def name(self) -> str:
        """Standard mode name (e.g., 1L, 1T, 2R)."""
        if self.mode_type == ModeType.LONGITUDINAL:
            return f"{self.mode_indices[0]}L"
        elif self.mode_type == ModeType.TANGENTIAL:
            return f"{self.mode_indices[0]}T"
        elif self.mode_type == ModeType.RADIAL:
            return f"{self.mode_indices[0]}R"
        else:
            return f"{self.mode_indices[0]}T{self.mode_indices[1]}R"


@dataclass
class StabilityMargin:
    """Stability margin analysis result."""
    mode: AcousticMode
    driving_gain: float  # Gain from combustion response
    damping_loss: float  # Loss from acoustic damping
    margin: float  # damping - driving (positive = stable)
    is_stable: bool


@dataclass
class InstabilityResult:
    """Complete combustion instability analysis."""
    chamber_length: float  # m
    chamber_diameter: float  # m
    speed_of_sound: float  # m/s

    longitudinal_modes: list[AcousticMode] = field(default_factory=list)
    transverse_modes: list[AcousticMode] = field(default_factory=list)
    all_modes: list[AcousticMode] = field(default_factory=list)

    chugging_frequency: float = 0.0  # Hz
    buzz_frequency_range: tuple[float, float] = (0.0, 0.0)  # Hz

    stability_margins: list[StabilityMargin] = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        """Check if all modes are stable."""
        if not self.stability_margins:
            return True  # Assume stable if not analyzed
        return all(m.is_stable for m in self.stability_margins)

    @property
    def most_critical_mode(self) -> StabilityMargin | None:
        """Find the mode with smallest stability margin."""
        if not self.stability_margins:
            return None
        return min(self.stability_margins, key=lambda m: m.margin)

    def get_modes_in_range(
        self,
        f_min: float,
        f_max: float
    ) -> list[AcousticMode]:
        """Get all modes within a frequency range."""
        return [m for m in self.all_modes if f_min <= m.frequency <= f_max]


# =============================================================================
# Bessel Function Zeros for Transverse Modes
# =============================================================================

# Zeros of J'_m(x) = 0 (derivative of Bessel function of first kind)
# These determine transverse mode frequencies in cylindrical chambers
# Format: BESSEL_ZEROS[m][n] = n-th zero of J'_m

BESSEL_ZEROS = {
    0: [0.0, 3.8317, 7.0156, 10.1735, 13.3237],  # Radial modes
    1: [1.8412, 5.3314, 8.5363, 11.7060, 14.8636],  # 1st tangential
    2: [3.0542, 6.7061, 9.9695, 13.1704, 16.3475],  # 2nd tangential
    3: [4.2012, 8.0152, 11.3459, 14.5858, 17.7887],  # 3rd tangential
    4: [5.3175, 9.2824, 12.6819, 15.9641, 19.1960],  # 4th tangential
}

# Mode descriptions
MODE_DESCRIPTIONS = {
    (0, 1): "1st Radial (1R) - Breathing mode",
    (0, 2): "2nd Radial (2R)",
    (1, 0): "1st Tangential (1T) - Spinning mode, most common instability",
    (1, 1): "1st Combined (1T1R)",
    (2, 0): "2nd Tangential (2T)",
    (2, 1): "2nd Combined (2T1R)",
    (3, 0): "3rd Tangential (3T)",
}


# =============================================================================
# Core Analysis Functions
# =============================================================================

def calculate_speed_of_sound(
    gamma: float,
    T_chamber: float,  # K
    mean_mw: float  # g/mol
) -> float:
    """
    Calculate speed of sound in combustion chamber.

    c = sqrt(γ * R * T)

    Args:
        gamma: Ratio of specific heats
        T_chamber: Chamber temperature (K)
        mean_mw: Mean molecular weight (g/mol)

    Returns:
        Speed of sound (m/s)
    """
    R = 8314.46 / mean_mw  # J/(kg·K)
    return np.sqrt(gamma * R * T_chamber)


def calculate_longitudinal_modes(
    chamber_length: float,  # m
    speed_of_sound: float,  # m/s
    n_modes: int = 5
) -> list[AcousticMode]:
    """
    Calculate longitudinal (axial) acoustic modes.

    For a chamber with closed injector end and open nozzle end,
    the fundamental frequency is approximately:

    f_n = n * c / (2L)  for n = 1, 2, 3, ...

    In reality, the nozzle provides an impedance boundary that
    modifies these frequencies slightly.

    Args:
        chamber_length: Chamber length (m)
        speed_of_sound: Speed of sound (m/s)
        n_modes: Number of modes to calculate

    Returns:
        List of longitudinal acoustic modes
    """
    modes = []

    for n in range(1, n_modes + 1):
        frequency = n * speed_of_sound / (2 * chamber_length)
        wavelength = 2 * chamber_length / n

        mode = AcousticMode(
            mode_type=ModeType.LONGITUDINAL,
            mode_indices=(n,),
            frequency=frequency,
            wavelength=wavelength,
            description=f"{n}L - {n}{'st' if n==1 else 'nd' if n==2 else 'rd' if n==3 else 'th'} Longitudinal"
        )
        modes.append(mode)

    return modes


def calculate_transverse_modes(
    chamber_diameter: float,  # m
    speed_of_sound: float,  # m/s
    max_m: int = 3,  # Maximum tangential order
    max_n: int = 2   # Maximum radial order
) -> list[AcousticMode]:
    """
    Calculate transverse (tangential and radial) acoustic modes.

    For a cylindrical chamber:
    f_mn = α_mn * c / (π * D)

    where α_mn is the (m,n)-th zero of J'_m(x).

    - m = 0: Pure radial modes (breathing)
    - m > 0, n = 0: Pure tangential modes (spinning/standing)
    - m > 0, n > 0: Combined tangential-radial modes

    Args:
        chamber_diameter: Chamber diameter (m)
        speed_of_sound: Speed of sound (m/s)
        max_m: Maximum tangential mode order
        max_n: Maximum radial mode order

    Returns:
        List of transverse acoustic modes
    """
    modes = []
    chamber_diameter / 2

    for m in range(max_m + 1):
        for n in range(max_n + 1):
            # Skip (0,0) - no mode
            if m == 0 and n == 0:
                continue

            # Get Bessel zero
            if m > 4 or n >= len(BESSEL_ZEROS[m]):
                continue

            alpha_mn = BESSEL_ZEROS[m][n]
            if alpha_mn == 0:
                continue

            # Calculate frequency
            frequency = alpha_mn * speed_of_sound / (np.pi * chamber_diameter)
            wavelength = np.pi * chamber_diameter / alpha_mn

            # Determine mode type
            if m == 0:
                mode_type = ModeType.RADIAL
                description = MODE_DESCRIPTIONS.get((m, n), f"{n}R - Radial mode")
            elif n == 0:
                mode_type = ModeType.TANGENTIAL
                description = MODE_DESCRIPTIONS.get((m, n), f"{m}T - Tangential mode")
            else:
                mode_type = ModeType.COMBINED
                description = MODE_DESCRIPTIONS.get((m, n), f"{m}T{n}R - Combined mode")

            mode = AcousticMode(
                mode_type=mode_type,
                mode_indices=(m, n),
                frequency=frequency,
                wavelength=wavelength,
                description=description
            )
            modes.append(mode)

    # Sort by frequency
    modes.sort(key=lambda m: m.frequency)

    return modes


def calculate_chugging_frequency(
    injection_pressure_drop: float,  # Pa
    chamber_pressure: float,  # Pa
    propellant_density: float,  # kg/m³
    feed_line_length: float,  # m
    feed_line_area: float  # m²
) -> float:
    """
    Estimate low-frequency chugging frequency.

    Chugging occurs when feed system hydraulics couple with
    chamber pressure oscillations. Typical range: 50-500 Hz.

    Simplified estimate based on Helmholtz resonator model:
    f_chug ≈ (1/2π) * sqrt(ΔP / (ρ * L * A))

    Args:
        injection_pressure_drop: Injector ΔP (Pa)
        chamber_pressure: Chamber pressure (Pa)
        propellant_density: Propellant density (kg/m³)
        feed_line_length: Feed line length (m)
        feed_line_area: Feed line cross-section area (m²)

    Returns:
        Estimated chugging frequency (Hz)
    """
    # Effective inertia of propellant in feed line
    inertia = propellant_density * feed_line_length

    # Spring constant from injection pressure drop
    k = injection_pressure_drop / feed_line_area

    # Natural frequency
    if inertia > 0:
        f_chug = (1 / (2 * np.pi)) * np.sqrt(k / inertia)
    else:
        f_chug = 100.0  # Default estimate

    return f_chug


def estimate_combustion_response(
    frequency: float,
    tau_c: float = 0.001  # Combustion time lag (s)
) -> float:
    """
    Estimate combustion response function magnitude.

    The n-τ model gives a rough estimate of how combustion
    responds to pressure oscillations:

    R(f) = n * exp(-2πf*τ) * (1 - exp(-2πf*τ))

    where n is the interaction index (~2-3 for typical propellants).

    Args:
        frequency: Oscillation frequency (Hz)
        tau_c: Characteristic combustion time lag (s)

    Returns:
        Response function magnitude
    """
    omega = 2 * np.pi * frequency
    x = omega * tau_c

    n = 2.5  # Typical interaction index

    response = n * np.exp(-x) * (1 - np.exp(-x)) if x > 0 else 0.0

    return response


def estimate_acoustic_damping(
    frequency: float,
    chamber_diameter: float,
    nozzle_throat_diameter: float,
    has_baffles: bool = False,
    has_acoustic_liner: bool = False
) -> float:
    """
    Estimate acoustic damping in the chamber.

    Sources of damping:
    1. Nozzle admittance (main energy sink)
    2. Viscous wall losses
    3. Baffles (if present)
    4. Acoustic liners (if present)

    Args:
        frequency: Mode frequency (Hz)
        chamber_diameter: Chamber diameter (m)
        nozzle_throat_diameter: Throat diameter (m)
        has_baffles: Whether baffles are installed
        has_acoustic_liner: Whether acoustic liner is present

    Returns:
        Damping coefficient (1/s)
    """
    # Nozzle damping (dominant term)
    throat_ratio = (nozzle_throat_diameter / chamber_diameter) ** 2
    nozzle_damping = 200 * throat_ratio  # Simplified model

    # Viscous damping (small contribution)
    viscous_damping = 0.01 * frequency

    # Baffle damping
    baffle_damping = 100 if has_baffles else 0

    # Acoustic liner damping
    liner_damping = 150 if has_acoustic_liner else 0

    return nozzle_damping + viscous_damping + baffle_damping + liner_damping


# =============================================================================
# High-Level Interface
# =============================================================================

def analyze_combustion_instability(
    chamber_length: float,  # m
    chamber_diameter: float,  # m
    gamma: float,
    T_chamber: float,  # K
    mean_mw: float,  # g/mol
    chamber_pressure: float = 1e6,  # Pa
    throat_diameter: float = None,  # m
    feed_line_length: float = 1.0,  # m
    injection_dp_ratio: float = 0.2,  # ΔP/Pc
    combustion_tau: float = 0.001,  # s
    has_baffles: bool = False,
    has_acoustic_liner: bool = False
) -> InstabilityResult:
    """
    Perform complete combustion instability analysis.

    Calculates all acoustic modes and estimates stability margins.

    Args:
        chamber_length: Combustion chamber length (m)
        chamber_diameter: Chamber diameter (m)
        gamma: Ratio of specific heats
        T_chamber: Chamber temperature (K)
        mean_mw: Mean molecular weight (g/mol)
        chamber_pressure: Chamber pressure (Pa)
        throat_diameter: Nozzle throat diameter (m), defaults to D/4
        feed_line_length: Propellant feed line length (m)
        injection_dp_ratio: Injector pressure drop ratio ΔP/Pc
        combustion_tau: Combustion time lag (s)
        has_baffles: Whether anti-oscillation baffles are present
        has_acoustic_liner: Whether acoustic damping liner is present

    Returns:
        Complete InstabilityResult with all modes and stability analysis
    """
    # Default throat diameter
    if throat_diameter is None:
        throat_diameter = chamber_diameter / 4

    # Calculate speed of sound
    c = calculate_speed_of_sound(gamma, T_chamber, mean_mw)

    # Calculate all modes
    longitudinal = calculate_longitudinal_modes(chamber_length, c)
    transverse = calculate_transverse_modes(chamber_diameter, c)
    all_modes = longitudinal + transverse
    all_modes.sort(key=lambda m: m.frequency)

    # Calculate chugging frequency
    injection_dp = injection_dp_ratio * chamber_pressure
    propellant_density = 1000.0  # Approximate for LOX
    feed_area = np.pi * (0.05) ** 2  # 5cm diameter line

    f_chug = calculate_chugging_frequency(
        injection_dp, chamber_pressure,
        propellant_density, feed_line_length, feed_area
    )

    # Buzz frequency range (intermediate instability)
    f_buzz = (200.0, 1000.0)

    # Analyze stability for each mode
    stability_margins = []

    for mode in all_modes:
        # Combustion driving
        driving = estimate_combustion_response(mode.frequency, combustion_tau)

        # Acoustic damping
        damping = estimate_acoustic_damping(
            mode.frequency, chamber_diameter, throat_diameter,
            has_baffles, has_acoustic_liner
        )

        # Normalize to comparable units
        driving_normalized = driving * 100
        damping_normalized = damping

        margin = damping_normalized - driving_normalized

        stability_margins.append(StabilityMargin(
            mode=mode,
            driving_gain=driving_normalized,
            damping_loss=damping_normalized,
            margin=margin,
            is_stable=(margin > 0)
        ))

    return InstabilityResult(
        chamber_length=chamber_length,
        chamber_diameter=chamber_diameter,
        speed_of_sound=c,
        longitudinal_modes=longitudinal,
        transverse_modes=transverse,
        all_modes=all_modes,
        chugging_frequency=f_chug,
        buzz_frequency_range=f_buzz,
        stability_margins=stability_margins
    )


def quick_stability_check(
    chamber_length: float,
    chamber_diameter: float,
    gamma: float = 1.2,
    T_chamber: float = 3500.0,
    mean_mw: float = 18.0
) -> str:
    """
    Quick stability assessment with recommendations.

    Returns:
        String summary of stability status
    """
    result = analyze_combustion_instability(
        chamber_length, chamber_diameter,
        gamma, T_chamber, mean_mw
    )

    lines = []
    lines.append(f"Chamber: L={chamber_length*1000:.0f}mm, D={chamber_diameter*1000:.0f}mm")
    lines.append(f"Speed of sound: {result.speed_of_sound:.0f} m/s")
    lines.append("")

    lines.append("Key Acoustic Modes:")
    for mode in result.all_modes[:5]:
        lines.append(f"  {mode.name}: {mode.frequency:.0f} Hz")

    lines.append("")
    if result.is_stable:
        lines.append("✓ All modes appear stable")
    else:
        critical = result.most_critical_mode
        if critical:
            lines.append(f"⚠️ UNSTABLE: {critical.mode.name} at {critical.mode.frequency:.0f} Hz")
            lines.append(f"   Margin: {critical.margin:.1f} (negative = unstable)")

    return "\n".join(lines)
