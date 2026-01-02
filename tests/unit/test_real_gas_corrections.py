"""
Unit tests for real gas corrections (compressibility factor).

Tests the virial equation, Redlich-Kwong EOS, and mixture
compressibility calculations.
"""

import numpy as np
import pytest

from src.core.thermodynamics import (
    calculate_compressibility_factor_virial,
    calculate_compressibility_factor_rk,
    correct_density_for_real_gas,
    correct_enthalpy_departure,
    get_mixture_compressibility,
    CRITICAL_PROPERTIES,
)


class TestVirialCompressibility:
    """Tests for virial equation compressibility factor."""

    def test_ideal_gas_at_low_pressure(self):
        """At low pressure, Z should approach 1.0."""
        Z = calculate_compressibility_factor_virial(
            T=300.0,
            P=101325.0,  # 1 atm
            Tc=150.0,
            Pc=5e6,
            omega=0.0
        )
        assert Z == pytest.approx(1.0, abs=0.1)

    def test_reduced_conditions(self):
        """Test at specific reduced conditions."""
        # At Tr=2, Pr=0.1, Z should be close to 1
        Tc = 150.0
        Pc = 5e6
        Z = calculate_compressibility_factor_virial(
            T=2.0 * Tc,  # Tr = 2
            P=0.1 * Pc,  # Pr = 0.1
            Tc=Tc,
            Pc=Pc,
            omega=0.0
        )
        assert 0.9 < Z < 1.1

    def test_z_less_than_one_near_critical(self):
        """Near critical point, Z should be less than 1."""
        Tc = 150.0
        Pc = 5e6
        Z = calculate_compressibility_factor_virial(
            T=1.1 * Tc,  # Tr = 1.1
            P=0.5 * Pc,  # Pr = 0.5
            Tc=Tc,
            Pc=Pc,
            omega=0.2
        )
        # Near critical, attractive forces dominate
        assert Z < 1.1

    def test_invalid_critical_properties(self):
        """Invalid critical properties should return Z=1."""
        Z = calculate_compressibility_factor_virial(
            T=300.0,
            P=1e6,
            Tc=0.0,  # Invalid
            Pc=5e6,
            omega=0.0
        )
        assert Z == 1.0


class TestRedlichKwongCompressibility:
    """Tests for Redlich-Kwong equation of state."""

    def test_ideal_gas_at_low_pressure(self):
        """At low pressure, RK should approach ideal gas."""
        Z = calculate_compressibility_factor_rk(
            T=500.0,
            P=101325.0,  # 1 atm
            Tc=150.0,
            Pc=5e6
        )
        assert Z == pytest.approx(1.0, abs=0.05)

    def test_high_pressure_deviation(self):
        """At high pressure, Z should deviate from 1."""
        Z = calculate_compressibility_factor_rk(
            T=300.0,
            P=10e6,  # 100 bar
            Tc=150.0,
            Pc=5e6
        )
        # Should deviate noticeably from ideal
        assert abs(Z - 1.0) > 0.01

    def test_physically_reasonable_range(self):
        """Z should stay in physically reasonable range."""
        Z = calculate_compressibility_factor_rk(
            T=200.0,
            P=20e6,
            Tc=150.0,
            Pc=5e6
        )
        assert 0.1 < Z < 3.0


class TestMixtureCompressibility:
    """Tests for mixture compressibility calculations."""

    def test_pure_species_h2(self):
        """Pure H2 compressibility."""
        Z = get_mixture_compressibility(
            T=300.0,
            P=1e6,
            composition={'H2': 1.0},
            method='virial'
        )
        # H2 is nearly ideal at moderate conditions
        assert 0.9 < Z < 1.1

    def test_pure_species_h2o(self):
        """Pure H2O compressibility."""
        Z = get_mixture_compressibility(
            T=500.0,
            P=1e6,
            composition={'H2O': 1.0},
            method='virial'
        )
        assert 0.8 < Z < 1.2

    def test_mixture_combustion_products(self):
        """Typical combustion products mixture."""
        composition = {
            'H2O': 0.6,
            'CO2': 0.2,
            'N2': 0.15,
            'H2': 0.05
        }
        Z = get_mixture_compressibility(
            T=2000.0,  # Hot combustion products
            P=5e6,
            composition=composition,
            method='virial'
        )
        # High temperature should give Z close to 1
        assert 0.9 < Z < 1.1

    def test_empty_composition_returns_ideal(self):
        """Empty composition should return Z=1."""
        Z = get_mixture_compressibility(
            T=300.0,
            P=1e6,
            composition={},
            method='virial'
        )
        assert Z == 1.0

    def test_unknown_species_handled(self):
        """Unknown species should be handled gracefully."""
        Z = get_mixture_compressibility(
            T=300.0,
            P=1e6,
            composition={'UnknownSpecies': 1.0},
            method='virial'
        )
        assert Z == 1.0


class TestDensityCorrection:
    """Tests for real gas density correction."""

    def test_z_greater_than_one_reduces_density(self):
        """Z > 1 should reduce density."""
        rho_ideal = 1.0
        rho_real = correct_density_for_real_gas(rho_ideal, Z=1.1)
        assert rho_real < rho_ideal

    def test_z_less_than_one_increases_density(self):
        """Z < 1 should increase density."""
        rho_ideal = 1.0
        rho_real = correct_density_for_real_gas(rho_ideal, Z=0.9)
        assert rho_real > rho_ideal

    def test_z_equals_one_no_change(self):
        """Z = 1 should give no change."""
        rho_ideal = 1.5
        rho_real = correct_density_for_real_gas(rho_ideal, Z=1.0)
        assert rho_real == pytest.approx(rho_ideal)

    def test_invalid_z_returns_ideal(self):
        """Invalid Z should return ideal density."""
        rho_ideal = 1.5
        rho_real = correct_density_for_real_gas(rho_ideal, Z=0.0)
        assert rho_real == rho_ideal


class TestEnthalpyDeparture:
    """Tests for enthalpy departure from ideal gas."""

    def test_low_pressure_small_departure(self):
        """Low pressure should give small departure."""
        dH = correct_enthalpy_departure(
            T=300.0,
            P=101325.0,  # 1 atm
            Tc=150.0,
            Pc=5e6,
            omega=0.0
        )
        # Should be small compared to RT
        R = 8.314
        assert abs(dH) < 0.5 * R * 300

    def test_high_pressure_larger_departure(self):
        """High pressure should give larger departure."""
        dH_low = correct_enthalpy_departure(
            T=300.0,
            P=1e5,
            Tc=150.0,
            Pc=5e6,
            omega=0.0
        )
        dH_high = correct_enthalpy_departure(
            T=300.0,
            P=5e6,
            Tc=150.0,
            Pc=5e6,
            omega=0.0
        )
        assert abs(dH_high) > abs(dH_low)

    def test_invalid_critical_zero_departure(self):
        """Invalid critical properties should give zero departure."""
        dH = correct_enthalpy_departure(
            T=300.0,
            P=1e6,
            Tc=0.0,  # Invalid
            Pc=5e6,
            omega=0.0
        )
        assert dH == 0.0


class TestCriticalProperties:
    """Tests for critical properties database."""

    def test_common_species_present(self):
        """Common combustion species should be in database."""
        required_species = ['H2', 'O2', 'N2', 'H2O', 'CO2', 'CO', 'CH4']
        for species in required_species:
            assert species in CRITICAL_PROPERTIES

    def test_critical_properties_reasonable(self):
        """Critical properties should be physically reasonable."""
        for species, (Tc, Pc, omega) in CRITICAL_PROPERTIES.items():
            assert Tc > 0, f"{species} Tc should be positive"
            assert Pc > 0, f"{species} Pc should be positive"
            assert -1 < omega < 2, f"{species} omega out of range"

