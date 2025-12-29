"""
Unit tests for input validation module.

Tests all validation functions and edge cases.
"""

import pytest
from src.core.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_chamber_pressure,
    validate_of_ratio,
    validate_expansion_ratio,
    validate_throat_area,
    validate_efficiency,
    validate_all_inputs,
    validate_temperature,
)


class TestChamberPressureValidation:
    """Test chamber pressure validation."""
    
    def test_negative_pressure_is_error(self):
        """Negative pressure should be invalid."""
        issues = validate_chamber_pressure(-10.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
        assert "positive" in issues[0].message.lower()
    
    def test_zero_pressure_is_error(self):
        """Zero pressure should be invalid."""
        issues = validate_chamber_pressure(0.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_extreme_high_pressure_is_error(self):
        """Pressure above 1000 bar should be error."""
        issues = validate_chamber_pressure(1500.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_normal_pressure_no_issues(self):
        """Normal pressure should have no issues."""
        issues = validate_chamber_pressure(68.0)
        assert len(issues) == 0
    
    def test_low_pressure_warning(self):
        """Very low pressure should generate warning."""
        issues = validate_chamber_pressure(0.5)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
    
    def test_very_high_pressure_warning(self):
        """Very high (but valid) pressure should generate warning."""
        issues = validate_chamber_pressure(600.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING


class TestOFRatioValidation:
    """Test O/F ratio validation."""
    
    def test_negative_of_ratio_is_error(self):
        """Negative O/F ratio should be invalid."""
        issues = validate_of_ratio(-2.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_zero_of_ratio_is_error(self):
        """Zero O/F ratio should be invalid."""
        issues = validate_of_ratio(0.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_extreme_high_of_ratio_is_error(self):
        """O/F ratio above 100 should be error."""
        issues = validate_of_ratio(150.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_normal_h2_of_ratio_no_issues(self):
        """Normal H2/O2 O/F ratio should be OK."""
        issues = validate_of_ratio(6.0, fuel_type='H2')
        # Should be in optimal range - no errors/warnings
        assert not any(i.severity == ValidationSeverity.ERROR for i in issues)
    
    def test_out_of_range_generates_info(self):
        """O/F outside optimal range should generate info."""
        issues = validate_of_ratio(12.0, fuel_type='H2')
        # Should have info about being outside optimal range
        has_info = any(i.severity == ValidationSeverity.INFO for i in issues)
        has_warning = any(i.severity == ValidationSeverity.WARNING for i in issues)
        assert has_info or has_warning


class TestExpansionRatioValidation:
    """Test expansion ratio validation."""
    
    def test_ratio_below_one_is_error(self):
        """Expansion ratio <= 1 should be error."""
        issues = validate_expansion_ratio(0.5)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_ratio_equal_one_is_error(self):
        """Expansion ratio = 1 should be error."""
        issues = validate_expansion_ratio(1.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_very_large_ratio_is_error(self):
        """Expansion ratio > 500 should be error."""
        issues = validate_expansion_ratio(600.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_normal_vacuum_ratio_ok(self):
        """Normal vacuum expansion ratio should be OK."""
        issues = validate_expansion_ratio(50.0, ambient_pressure_bar=0.0)
        assert not any(i.severity == ValidationSeverity.ERROR for i in issues)
    
    def test_high_ratio_at_sea_level_warning(self):
        """High ratio at sea level should generate warning."""
        issues = validate_expansion_ratio(80.0, ambient_pressure_bar=1.0)
        assert len(issues) >= 1
        assert issues[0].severity == ValidationSeverity.WARNING


class TestEfficiencyValidation:
    """Test efficiency factor validation."""
    
    def test_negative_efficiency_is_error(self):
        """Negative efficiency should be error."""
        issues = validate_efficiency(-0.5)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_zero_efficiency_is_error(self):
        """Zero efficiency should be error."""
        issues = validate_efficiency(0.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_above_one_efficiency_is_error(self):
        """Efficiency > 1.0 should be error."""
        issues = validate_efficiency(1.5)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_normal_efficiency_ok(self):
        """Normal efficiency should be OK."""
        issues = validate_efficiency(0.97)
        assert not any(i.severity == ValidationSeverity.ERROR for i in issues)
    
    def test_low_efficiency_warning(self):
        """Low efficiency should generate warning."""
        issues = validate_efficiency(0.4)
        assert len(issues) >= 1
        assert issues[0].severity == ValidationSeverity.WARNING


class TestFullValidation:
    """Test complete input validation."""
    
    def test_all_valid_inputs(self):
        """All valid inputs should pass."""
        result = validate_all_inputs(
            pressure_bar=68.0,
            of_ratio=6.0,
            expansion_ratio=50.0,
            throat_area_cm2=100.0,
            eta_cstar=0.97,
            eta_cf=0.98,
            fuel_type='H2',
            ambient_pressure_bar=0.0
        )
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_invalid_pressure_fails(self):
        """Invalid pressure should cause validation failure."""
        result = validate_all_inputs(
            pressure_bar=-10.0,  # Invalid
            of_ratio=6.0,
            expansion_ratio=50.0,
            throat_area_cm2=100.0,
            eta_cstar=0.97,
            eta_cf=0.98
        )
        assert not result.is_valid
        assert len(result.errors) >= 1
    
    def test_multiple_errors_collected(self):
        """Multiple errors should all be collected."""
        result = validate_all_inputs(
            pressure_bar=-10.0,  # Error
            of_ratio=-5.0,       # Error
            expansion_ratio=0.5,  # Error
            throat_area_cm2=-10.0,  # Error
            eta_cstar=2.0,       # Error
            eta_cf=-0.5          # Error
        )
        assert not result.is_valid
        assert len(result.errors) >= 4  # At least 4 errors
    
    def test_warnings_dont_invalidate(self):
        """Warnings should not cause validation failure."""
        result = validate_all_inputs(
            pressure_bar=0.5,  # Warning - low
            of_ratio=0.7,      # Warning - low
            expansion_ratio=5.0,
            throat_area_cm2=100.0,
            eta_cstar=0.97,
            eta_cf=0.98
        )
        assert result.is_valid
        assert len(result.warnings) >= 1


class TestTemperatureValidation:
    """Test temperature validation."""
    
    def test_negative_temp_is_error(self):
        """Negative temperature should be error."""
        issues = validate_temperature(-100.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_zero_temp_is_error(self):
        """Zero temperature should be error."""
        issues = validate_temperature(0.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_extreme_high_temp_is_error(self):
        """Temperature above 6000K should be error."""
        issues = validate_temperature(7000.0)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_normal_temp_ok(self):
        """Normal combustion temperature should be OK."""
        issues = validate_temperature(3500.0)
        assert not any(i.severity == ValidationSeverity.ERROR for i in issues)
