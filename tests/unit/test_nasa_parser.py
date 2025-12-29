"""
Unit tests for NASA thermodynamic data parser.
"""

import numpy as np
import pytest

from src.core.types import SpeciesData
from src.utils.nasa_parser import (
    parse_nasa_file,
    create_sample_database,
    _parse_coefficient,
    _calculate_mw_from_name,
    NASAParserError,
)


class TestCoefficientParsing:
    """Test coefficient parsing from fixed-width fields."""
    
    def test_parse_standard_exponential(self):
        """Parse standard E notation."""
        result = _parse_coefficient(" 1.23456789E+01")
        assert abs(result - 12.3456789) < 1e-7
    
    def test_parse_fortran_d_notation(self):
        """Parse Fortran D notation."""
        result = _parse_coefficient(" 1.23456789D+01")
        assert abs(result - 12.3456789) < 1e-7
    
    def test_parse_negative_exponent(self):
        """Parse negative exponent."""
        result = _parse_coefficient(" 1.23456789E-05")
        assert abs(result - 1.23456789e-5) < 1e-12
    
    def test_parse_negative_number(self):
        """Parse negative coefficient."""
        result = _parse_coefficient("-1.23456789E+01")
        assert abs(result - (-12.3456789)) < 1e-7
    
    def test_parse_empty_field(self):
        """Empty field returns zero."""
        result = _parse_coefficient("               ")
        assert result == 0.0
    
    def test_parse_zero(self):
        """Parse explicit zero."""
        result = _parse_coefficient(" 0.00000000E+00")
        assert result == 0.0


class TestMolecularWeightCalculation:
    """Test molecular weight calculation from formulas."""
    
    def test_simple_formula_h2o(self):
        """Test H2O molecular weight."""
        mw = _calculate_mw_from_name("H2O")
        expected = 2 * 1.00794 + 15.9994  # ~18.015
        assert abs(mw - expected) < 0.01
    
    def test_simple_formula_co2(self):
        """Test CO2 molecular weight."""
        mw = _calculate_mw_from_name("CO2")
        expected = 12.0107 + 2 * 15.9994  # ~44.01
        assert abs(mw - expected) < 0.01
    
    def test_formula_with_phase_indicator(self):
        """Test formula with (G) suffix."""
        mw = _calculate_mw_from_name("H2O(G)")
        expected = 2 * 1.00794 + 15.9994
        assert abs(mw - expected) < 0.01
    
    def test_diatomic_molecule(self):
        """Test N2 molecular weight."""
        mw = _calculate_mw_from_name("N2")
        expected = 2 * 14.0067  # ~28.01
        assert abs(mw - expected) < 0.01
    
    def test_complex_formula_ch4(self):
        """Test CH4 molecular weight."""
        mw = _calculate_mw_from_name("CH4")
        expected = 12.0107 + 4 * 1.00794  # ~16.04
        assert abs(mw - expected) < 0.01


class TestSampleDatabase:
    """Test the built-in sample database."""
    
    @pytest.fixture
    def db(self):
        return create_sample_database()
    
    def test_contains_expected_species(self, db):
        """Check all expected species are present."""
        expected = ['H2O', 'O2', 'H2', 'N2', 'CO2', 'CO', 'OH', 'CH4']
        for species in expected:
            assert species in db, f"Missing species: {species}"
    
    def test_h2o_molecular_weight(self, db):
        """Verify H2O molecular weight."""
        h2o = db['H2O']
        assert abs(h2o.molecular_weight - 18.01528) < 1e-4
    
    def test_h2o_temperature_range(self, db):
        """Verify H2O temperature range."""
        h2o = db['H2O']
        assert h2o.t_low == 200.0
        assert h2o.t_mid == 1000.0
        assert h2o.t_high == 6000.0
    
    def test_h2o_coefficients_shape(self, db):
        """Verify coefficient array shapes."""
        h2o = db['H2O']
        assert h2o.coeffs_high.shape == (7,)
        assert h2o.coeffs_low.shape == (7,)
    
    def test_h2o_formation_enthalpy(self, db):
        """Verify H2O formation enthalpy."""
        h2o = db['H2O']
        # H2O: ΔHf = -241.826 kJ/mol = -241826 J/mol
        assert h2o.h_formation_298 is not None
        assert abs(h2o.h_formation_298 - (-241826.0)) < 10.0
    
    def test_reference_elements_zero_hf(self, db):
        """Reference elements should have zero formation enthalpy."""
        assert db['H2'].h_formation_298 == 0.0
        assert db['O2'].h_formation_298 == 0.0
        assert db['N2'].h_formation_298 == 0.0
    
    def test_co2_formation_enthalpy(self, db):
        """Verify CO2 formation enthalpy."""
        co2 = db['CO2']
        # CO2: ΔHf = -393.51 kJ/mol
        assert co2.h_formation_298 is not None
        assert abs(co2.h_formation_298 - (-393510.0)) < 100.0


class TestSpeciesDataClass:
    """Test SpeciesData dataclass functionality."""
    
    def test_species_repr(self):
        """Test string representation."""
        species = SpeciesData(
            name="TEST",
            molecular_weight=28.0,
            temp_ranges=[(200.0, 1000.0, 6000.0)]
        )
        repr_str = repr(species)
        assert "TEST" in repr_str
        assert "28.0" in repr_str
    
    def test_default_temperature_range(self):
        """Test default temperature properties."""
        species = SpeciesData(
            name="TEST",
            molecular_weight=28.0
        )
        # Defaults when no temp_ranges specified
        assert species.t_low == 200.0
        assert species.t_mid == 1000.0
        assert species.t_high == 6000.0
    
    def test_default_coefficients(self):
        """Test default coefficient arrays."""
        species = SpeciesData(
            name="TEST",
            molecular_weight=28.0
        )
        assert species.coeffs_high.shape == (7,)
        assert species.coeffs_low.shape == (7,)
        np.testing.assert_array_equal(species.coeffs_high, np.zeros(7))
        np.testing.assert_array_equal(species.coeffs_low, np.zeros(7))
