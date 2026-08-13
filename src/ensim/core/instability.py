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

    chugging_frequency: float | None = None  # Hz
    buzz_frequency_range: tuple[float, float] | None = None  # Hz

    stability_margins: list[StabilityMargin] = field(default_factory=list)
    assessment_level: str = "acoustic_modes_only"
    limitations: tuple[str, ...] = ()

    @property
    def is_stable(self) -> bool | None:
        """Return modal stability, or None when response data was not supplied."""
        if not self.stability_margins:
            return None
        return all(m.is_stable for m in self.stability_margins)

    @property
    def most_critical_mode(self) -> StabilityMargin | None:
        """Find the mode with smallest stability margin."""
        if not self.stability_margins:
            return None
        return min(self.stability_margins, key=lambda m: m.margin)

    def get_modes_in_range(self, f_min: float, f_max: float) -> list[AcousticMode]:
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
    mean_mw: float,  # g/mol
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
    if T_chamber <= 0.0 or mean_mw <= 0.0 or not 1.0 < gamma < 2.0:
        raise ValueError("Gas properties are outside physical bounds")
    R = 8314.46261815324 / mean_mw  # J/(kg·K)
    return np.sqrt(gamma * R * T_chamber)


def calculate_longitudinal_modes(
    chamber_length: float,  # m
    speed_of_sound: float,  # m/s
    n_modes: int = 5,
) -> list[AcousticMode]:
    """
    Calculate longitudinal (axial) acoustic modes.

    In the ideal uniform cylindrical-chamber screening model:

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
    if chamber_length <= 0.0 or speed_of_sound <= 0.0 or n_modes < 1:
        raise ValueError("Length, sound speed, and mode count must be positive")
    modes = []

    for n in range(1, n_modes + 1):
        frequency = n * speed_of_sound / (2 * chamber_length)
        wavelength = 2 * chamber_length / n

        mode = AcousticMode(
            mode_type=ModeType.LONGITUDINAL,
            mode_indices=(n,),
            frequency=frequency,
            wavelength=wavelength,
            description=f"{n}L - {n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} Longitudinal",
        )
        modes.append(mode)

    return modes


def calculate_transverse_modes(
    chamber_diameter: float,  # m
    speed_of_sound: float,  # m/s
    max_m: int = 3,  # Maximum tangential order
    max_n: int = 2,  # Maximum radial order
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
    if chamber_diameter <= 0.0 or speed_of_sound <= 0.0 or max_m < 0 or max_n < 0:
        raise ValueError("Diameter, sound speed, and mode orders are invalid")
    modes = []

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
                description=description,
            )
            modes.append(mode)

    # Sort by frequency
    modes.sort(key=lambda m: m.frequency)

    return modes


def calculate_chugging_frequency(
    propellant_density: float,  # kg/m³
    feed_line_length: float,  # m
    feed_line_area: float,  # m²
    system_compliance: float,  # m³/Pa
) -> float:
    """
    Estimate low-frequency chugging frequency.

    Chugging occurs when feed system hydraulics couple with
    chamber pressure oscillations. Typical range: 50-500 Hz.

    Lumped inertance-compliance model:
    f_chug = (1/2π) sqrt(A / (ρ L C))

    Args:
        propellant_density: Propellant density (kg/m³)
        feed_line_length: Feed line length (m)
        feed_line_area: Feed line cross-section area (m²)
        system_compliance: Combined feed/chamber volume compliance (m³/Pa)

    Returns:
        Estimated chugging frequency (Hz)
    """
    if (
        propellant_density <= 0.0
        or feed_line_length <= 0.0
        or feed_line_area <= 0.0
        or system_compliance <= 0.0
    ):
        raise ValueError("Feed-system properties and compliance must be positive")
    return (
        1.0
        / (2.0 * np.pi)
        * np.sqrt(feed_line_area / (propellant_density * feed_line_length * system_compliance))
    )


# =============================================================================
# High-Level Interface
# =============================================================================


def analyze_combustion_instability(
    chamber_length: float,  # m
    chamber_diameter: float,  # m
    gamma: float,
    T_chamber: float,  # K
    mean_mw: float,  # g/mol
    feed_line_length: float = 1.0,  # m
    feed_line_area: float | None = None,  # m²
    propellant_density: float | None = None,  # kg/m³
    feed_system_compliance: float | None = None,  # m³/Pa
    modal_rate_data: dict[str, tuple[float, float]] | None = None,
) -> InstabilityResult:
    """
    Perform complete combustion instability analysis.

    Calculates ideal cylindrical-chamber acoustic modes. Stability margins are
    only evaluated when measured or independently modelled modal driving and
    damping rates are supplied in consistent units of 1/s.

    Args:
        chamber_length: Combustion chamber length (m)
        chamber_diameter: Chamber diameter (m)
        gamma: Ratio of specific heats
        T_chamber: Chamber temperature (K)
        mean_mw: Mean molecular weight (g/mol)
        feed_line_length: Propellant feed line length (m)
        feed_line_area: Feed-line flow area (m²)
        propellant_density: Feed propellant density (kg/m³)
        feed_system_compliance: Lumped feed/chamber compliance (m³/Pa)
        modal_rate_data: Mapping from mode name to (driving, damping) rates (1/s)

    Returns:
        Complete InstabilityResult with all modes and stability analysis
    """
    if chamber_length <= 0.0 or chamber_diameter <= 0.0:
        raise ValueError("Chamber dimensions must be positive")
    if T_chamber <= 0.0 or mean_mw <= 0.0 or not 1.0 < gamma < 2.0:
        raise ValueError("Gas properties are outside physical bounds")

    # Calculate speed of sound
    c = calculate_speed_of_sound(gamma, T_chamber, mean_mw)

    # Calculate all modes
    longitudinal = calculate_longitudinal_modes(chamber_length, c)
    transverse = calculate_transverse_modes(chamber_diameter, c)
    all_modes = longitudinal + transverse
    all_modes.sort(key=lambda m: m.frequency)

    chug_inputs = (feed_line_area, propellant_density, feed_system_compliance)
    if all(value is not None for value in chug_inputs):
        f_chug = calculate_chugging_frequency(
            propellant_density,
            feed_line_length,
            feed_line_area,
            feed_system_compliance,
        )
    elif any(value is not None for value in chug_inputs):
        raise ValueError(
            "Feed-line area, propellant density, and compliance must be supplied together"
        )
    else:
        f_chug = None

    stability_margins = []
    if modal_rate_data is not None:
        modes_by_name = {mode.name: mode for mode in all_modes}
        unknown_modes = set(modal_rate_data) - set(modes_by_name)
        if unknown_modes:
            raise ValueError(f"Unknown acoustic modes: {sorted(unknown_modes)}")
        for mode_name, (driving, damping) in modal_rate_data.items():
            if driving < 0.0 or damping < 0.0:
                raise ValueError("Modal driving and damping rates must be nonnegative")
            margin = damping - driving
            stability_margins.append(
                StabilityMargin(
                    mode=modes_by_name[mode_name],
                    driving_gain=driving,
                    damping_loss=damping,
                    margin=margin,
                    is_stable=(margin > 0.0),
                )
            )

    return InstabilityResult(
        chamber_length=chamber_length,
        chamber_diameter=chamber_diameter,
        speed_of_sound=c,
        longitudinal_modes=longitudinal,
        transverse_modes=transverse,
        all_modes=all_modes,
        chugging_frequency=f_chug,
        buzz_frequency_range=None,
        stability_margins=stability_margins,
        assessment_level=(
            "modal_growth_rate_balance" if modal_rate_data is not None else "acoustic_modes_only"
        ),
        limitations=(
            "Ideal cylindrical chamber with uniform calorically-perfect gas properties",
            "No stability verdict without externally supplied modal driving and damping rates",
            "No injector coupling, nonlinear limit cycle, or spatial Rayleigh-index solution",
        ),
    )


def quick_stability_check(
    chamber_length: float,
    chamber_diameter: float,
    gamma: float = 1.2,
    T_chamber: float = 3500.0,
    mean_mw: float = 18.0,
) -> str:
    """
    Quick stability assessment with recommendations.

    Returns:
        String summary of stability status
    """
    result = analyze_combustion_instability(
        chamber_length, chamber_diameter, gamma, T_chamber, mean_mw
    )

    lines = []
    lines.append(f"Chamber: L={chamber_length * 1000:.0f}mm, D={chamber_diameter * 1000:.0f}mm")
    lines.append(f"Speed of sound: {result.speed_of_sound:.0f} m/s")
    lines.append("")

    lines.append("Key Acoustic Modes:")
    for mode in result.all_modes[:5]:
        lines.append(f"  {mode.name}: {mode.frequency:.0f} Hz")

    lines.append("")
    if result.is_stable is None:
        lines.append("Stability not assessed: modal driving and damping data are required.")
    elif result.is_stable:
        lines.append("All supplied modal growth-rate balances are stable.")
    else:
        critical = result.most_critical_mode
        if critical:
            lines.append(
                f"Unstable supplied balance: {critical.mode.name} at {critical.mode.frequency:.0f} Hz"
            )
            lines.append(f"Net damping rate: {critical.margin:.1f} 1/s")

    return "\n".join(lines)
