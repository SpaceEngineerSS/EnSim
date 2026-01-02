"""
Input Validation Framework for EnSim.

Comprehensive validation for all simulation inputs with
detailed error messages and warnings.
"""

from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Cannot proceed
    WARNING = "warning"  # Can proceed but unusual
    INFO = "info"        # Just informational


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    field: str
    message: str
    value: float | None = None
    valid_range: tuple[float, float] | None = None

    def __str__(self) -> str:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[self.severity.value]
        return f"{icon} {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def __str__(self) -> str:
        if self.is_valid and not self.warnings:
            return "✓ All inputs valid"

        lines = []
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)


# =============================================================================
# Physical Constants for Validation
# =============================================================================

# Pressure limits (bar)
PRESSURE_MIN = 0.01
PRESSURE_MAX_SAFE = 500.0
PRESSURE_MAX_ABSOLUTE = 1000.0

# Temperature limits (K)
TEMPERATURE_MIN = 200.0
TEMPERATURE_MAX_MATERIAL = 4500.0  # Typical ablative limit
TEMPERATURE_MAX_ABSOLUTE = 6000.0   # Physical limit for calculations

# O/F Ratio ranges by propellant
OF_RANGES = {
    'H2': (4.0, 8.0),     # H2/O2 stoichiometric ~8
    'CH4': (2.5, 4.5),    # Methane/O2
    'RP-1': (2.0, 3.5),   # Kerosene/O2
    'N2H4': (0.8, 1.5),   # Hydrazine/N2O4
    'MMH': (1.5, 3.0),    # MMH/N2O4
    'UDMH': (1.8, 3.0),   # UDMH/N2O4
}

# Expansion ratio limits
EXPANSION_RATIO_MIN = 1.01
EXPANSION_RATIO_MAX_VACUUM = 500.0
EXPANSION_RATIO_MAX_SEA_LEVEL = 50.0

# Efficiency limits
EFFICIENCY_MIN = 0.5
EFFICIENCY_MAX = 1.0
EFFICIENCY_TYPICAL_MIN = 0.85


# =============================================================================
# Validation Functions
# =============================================================================

def validate_chamber_pressure(pressure_bar: float) -> list[ValidationIssue]:
    """
    Validate combustion chamber pressure.

    Args:
        pressure_bar: Chamber pressure in bar

    Returns:
        List of validation issues
    """
    issues = []

    # Critical errors
    if pressure_bar <= 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Chamber Pressure",
            message=f"Pressure must be positive (got {pressure_bar:.2f} bar)",
            value=pressure_bar,
            valid_range=(PRESSURE_MIN, PRESSURE_MAX_ABSOLUTE)
        ))
        return issues

    if pressure_bar > PRESSURE_MAX_ABSOLUTE:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Chamber Pressure",
            message=f"Pressure exceeds calculation limits ({pressure_bar:.1f} > {PRESSURE_MAX_ABSOLUTE} bar)",
            value=pressure_bar,
            valid_range=(PRESSURE_MIN, PRESSURE_MAX_ABSOLUTE)
        ))
        return issues

    # Warnings
    if pressure_bar < 1.0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="Chamber Pressure",
            message=f"Very low pressure ({pressure_bar:.2f} bar) - typical engines use 20-300 bar",
            value=pressure_bar
        ))
    elif pressure_bar > PRESSURE_MAX_SAFE:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="Chamber Pressure",
            message=f"Extremely high pressure ({pressure_bar:.1f} bar) - exceeds most material limits",
            value=pressure_bar
        ))
    elif pressure_bar < 10.0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            field="Chamber Pressure",
            message=f"Low pressure ({pressure_bar:.1f} bar) - suitable for small thrusters",
            value=pressure_bar
        ))

    return issues


def validate_of_ratio(of_ratio: float, fuel_type: str = 'H2') -> list[ValidationIssue]:
    """
    Validate oxidizer-to-fuel mass ratio.

    Args:
        of_ratio: O/F mass ratio
        fuel_type: Fuel identifier for optimal range lookup

    Returns:
        List of validation issues
    """
    issues = []

    # Critical errors
    if of_ratio <= 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="O/F Ratio",
            message=f"O/F ratio must be positive (got {of_ratio:.2f})",
            value=of_ratio
        ))
        return issues

    if of_ratio > 100:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="O/F Ratio",
            message=f"O/F ratio unrealistically high ({of_ratio:.1f})",
            value=of_ratio
        ))
        return issues

    # Get optimal range for this fuel
    optimal_range = OF_RANGES.get(fuel_type, (1.0, 10.0))

    # Warnings
    if of_ratio < 0.5:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="O/F Ratio",
            message=f"Very fuel-rich mixture ({of_ratio:.2f}) - combustion may be incomplete",
            value=of_ratio,
            valid_range=optimal_range
        ))
    elif of_ratio > 20:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="O/F Ratio",
            message=f"Very oxidizer-rich mixture ({of_ratio:.1f}) - unusual for propulsion",
            value=of_ratio,
            valid_range=optimal_range
        ))
    elif of_ratio < optimal_range[0] or of_ratio > optimal_range[1]:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            field="O/F Ratio",
            message=f"O/F={of_ratio:.2f} outside optimal range {optimal_range} for {fuel_type}",
            value=of_ratio,
            valid_range=optimal_range
        ))

    return issues


def validate_expansion_ratio(
    expansion_ratio: float,
    ambient_pressure_bar: float = 0.0
) -> list[ValidationIssue]:
    """
    Validate nozzle expansion ratio.

    Args:
        expansion_ratio: Area ratio Ae/At
        ambient_pressure_bar: Ambient pressure for context

    Returns:
        List of validation issues
    """
    issues = []

    # Critical errors
    if expansion_ratio <= 1.0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Expansion Ratio",
            message=f"Expansion ratio must be > 1.0 (got {expansion_ratio:.2f})",
            value=expansion_ratio,
            valid_range=(EXPANSION_RATIO_MIN, EXPANSION_RATIO_MAX_VACUUM)
        ))
        return issues

    if expansion_ratio > EXPANSION_RATIO_MAX_VACUUM:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Expansion Ratio",
            message=f"Expansion ratio too large ({expansion_ratio:.1f} > {EXPANSION_RATIO_MAX_VACUUM})",
            value=expansion_ratio
        ))
        return issues

    # Context-aware warnings
    at_sea_level = ambient_pressure_bar > 0.5

    if at_sea_level and expansion_ratio > EXPANSION_RATIO_MAX_SEA_LEVEL:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="Expansion Ratio",
            message=f"High ε={expansion_ratio:.0f} at sea level - flow separation likely",
            value=expansion_ratio,
            valid_range=(EXPANSION_RATIO_MIN, EXPANSION_RATIO_MAX_SEA_LEVEL)
        ))

    if expansion_ratio > 200:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            field="Expansion Ratio",
            message=f"Very large expansion (ε={expansion_ratio:.0f}) - typical for vacuum-optimized stages",
            value=expansion_ratio
        ))

    return issues


def validate_throat_area(throat_area_cm2: float) -> list[ValidationIssue]:
    """
    Validate nozzle throat area.

    Args:
        throat_area_cm2: Throat area in cm²

    Returns:
        List of validation issues
    """
    issues = []

    if throat_area_cm2 <= 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Throat Area",
            message=f"Throat area must be positive (got {throat_area_cm2:.2f} cm²)",
            value=throat_area_cm2
        ))
        return issues

    if throat_area_cm2 < 0.01:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="Throat Area",
            message=f"Very small throat ({throat_area_cm2:.3f} cm²) - microthuster scale",
            value=throat_area_cm2
        ))
    elif throat_area_cm2 > 50000:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="Throat Area",
            message=f"Very large throat ({throat_area_cm2:.0f} cm²) - heavy-lift engine scale",
            value=throat_area_cm2
        ))

    return issues


def validate_efficiency(
    eta: float,
    efficiency_type: str = "C*"
) -> list[ValidationIssue]:
    """
    Validate efficiency factor.

    Args:
        eta: Efficiency value (0-1)
        efficiency_type: "C*" or "Cf" for context

    Returns:
        List of validation issues
    """
    issues = []

    if eta <= 0 or eta > 1.0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field=f"η_{efficiency_type}",
            message=f"Efficiency must be in (0, 1] (got {eta:.3f})",
            value=eta,
            valid_range=(EFFICIENCY_MIN, EFFICIENCY_MAX)
        ))
        return issues

    if eta < EFFICIENCY_MIN:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field=f"η_{efficiency_type}",
            message=f"Very low efficiency ({eta:.2f}) - check engine design",
            value=eta,
            valid_range=(EFFICIENCY_TYPICAL_MIN, EFFICIENCY_MAX)
        ))
    elif eta < EFFICIENCY_TYPICAL_MIN:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            field=f"η_{efficiency_type}",
            message=f"Below-typical efficiency ({eta:.2f}) - prototype/experimental range",
            value=eta
        ))

    return issues


def validate_all_inputs(
    pressure_bar: float,
    of_ratio: float,
    expansion_ratio: float,
    throat_area_cm2: float,
    eta_cstar: float,
    eta_cf: float,
    fuel_type: str = 'H2',
    ambient_pressure_bar: float = 0.0
) -> ValidationResult:
    """
    Validate all simulation inputs comprehensively.

    Args:
        pressure_bar: Chamber pressure (bar)
        of_ratio: O/F ratio
        expansion_ratio: Ae/At
        throat_area_cm2: Throat area (cm²)
        eta_cstar: Combustion efficiency
        eta_cf: Nozzle efficiency
        fuel_type: Fuel identifier
        ambient_pressure_bar: Ambient pressure (bar)

    Returns:
        ValidationResult with all issues
    """
    all_issues = []

    # Collect all issues
    all_issues.extend(validate_chamber_pressure(pressure_bar))
    all_issues.extend(validate_of_ratio(of_ratio, fuel_type))
    all_issues.extend(validate_expansion_ratio(expansion_ratio, ambient_pressure_bar))
    all_issues.extend(validate_throat_area(throat_area_cm2))
    all_issues.extend(validate_efficiency(eta_cstar, "C*"))
    all_issues.extend(validate_efficiency(eta_cf, "Cf"))

    # Determine overall validity (any ERROR = invalid)
    has_errors = any(i.severity == ValidationSeverity.ERROR for i in all_issues)

    return ValidationResult(
        is_valid=not has_errors,
        issues=all_issues
    )


def validate_temperature(temperature_k: float) -> list[ValidationIssue]:
    """
    Validate temperature value.

    Args:
        temperature_k: Temperature in Kelvin

    Returns:
        List of validation issues
    """
    issues = []

    if temperature_k <= 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Temperature",
            message=f"Temperature must be positive (got {temperature_k:.1f} K)",
            value=temperature_k
        ))
        return issues

    if temperature_k < TEMPERATURE_MIN:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Temperature",
            message=f"Temperature below valid range ({temperature_k:.1f} < {TEMPERATURE_MIN} K)",
            value=temperature_k,
            valid_range=(TEMPERATURE_MIN, TEMPERATURE_MAX_ABSOLUTE)
        ))

    if temperature_k > TEMPERATURE_MAX_ABSOLUTE:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="Temperature",
            message=f"Temperature exceeds calculation limits ({temperature_k:.0f} > {TEMPERATURE_MAX_ABSOLUTE} K)",
            value=temperature_k
        ))
    elif temperature_k > TEMPERATURE_MAX_MATERIAL:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="Temperature",
            message=f"Temperature exceeds typical material limits ({temperature_k:.0f} K)",
            value=temperature_k
        ))

    return issues
