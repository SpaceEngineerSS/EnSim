"""
Unit tests for thermodynamic property calculations.

Tests verify NASA polynomial calculations against known reference values
from NASA Glenn database and CEA outputs.

Reference:
    - NASA/TP-2002-211556 (NASA Glenn Coefficients)
    - NIST Chemistry WebBook (thermodynamic data validation)
"""

import numpy as np
import pytest

from src.core.constants import GAS_CONSTANT
from src.core.thermodynamics import (
    cp_over_r,
    h_over_rt,
    s_over_r,
    g_over_rt,
    get_thermo_properties,
    calculate_cp,
    calculate_enthalpy,
    calculate_entropy,
)
from src.utils.nasa_parser import create_sample_database


class TestNASAPolynomialCalculations:
    """Test NASA 7-term polynomial thermodynamic calculations."""
    
    @pytest.fixture
    def species_db(self):
        """Create sample species database for testing."""
        return create_sample_database()
    
    @pytest.fixture
    def h2o(self, species_db):
        """Get H2O species data."""
        return species_db['H2O']
    
    @pytest.fixture
    def o2(self, species_db):
        """Get O2 species data."""
        return species_db['O2']
    
    def test_cp_over_r_h2o_low_temp(self, h2o):
        """Test Cp/R calculation for H2O at low temperature (500K)."""
        T = 500.0
        coeffs = h2o.coeffs_low  # T < 1000K uses low coefficients
        
        result = cp_over_r(T, coeffs)
        
        # H2O Cp at 500K should be approximately 4.0-4.2 R
        # Reference: NIST Chemistry WebBook
        assert 3.9 < result < 4.3, f"Cp/R at 500K = {result}, expected ~4.0-4.2"
    
    def test_cp_over_r_h2o_high_temp(self, h2o):
        """Test Cp/R calculation for H2O at high temperature (2000K)."""
        T = 2000.0
        coeffs = h2o.coeffs_high  # T >= 1000K uses high coefficients
        
        result = cp_over_r(T, coeffs)
        
        # H2O Cp at 2000K: Cp ≈ 51.7 J/(mol·K) -> Cp/R ≈ 6.2
        # Reference: NIST Chemistry WebBook
        assert 5.8 < result < 6.8, f"Cp/R at 2000K = {result}, expected ~6.0-6.5"
    
    def test_h_over_rt_h2o_298K(self, h2o):
        """Test H/(RT) calculation at reference temperature."""
        T = 298.15
        coeffs = h2o.coeffs_low
        
        result = h_over_rt(T, coeffs)
        
        # H/(RT) should be large negative for H2O (exothermic formation)
        # H2O: ΔHf = -241.83 kJ/mol -> H/(RT) ≈ -97.5 at 298K
        assert result < -90, f"H/(RT) at 298K = {result}, expected large negative"
    
    def test_s_over_r_h2o_298K(self, h2o):
        """Test S/R calculation at reference temperature."""
        T = 298.15
        coeffs = h2o.coeffs_low
        
        result = s_over_r(T, coeffs)
        
        # S/R for H2O at 298K: S° = 188.8 J/(mol·K) -> S/R ≈ 22.7
        assert 22 < result < 24, f"S/R at 298K = {result}, expected ~22.7"
    
    def test_temperature_range_switching(self, h2o):
        """Test that coefficient selection switches correctly at T_mid."""
        T_low = 999.0
        T_high = 1001.0
        
        # Get properties just below and above T_mid
        props_low = get_thermo_properties(
            T_low, h2o.coeffs_low, h2o.coeffs_high, T_mid=1000.0
        )
        props_high = get_thermo_properties(
            T_high, h2o.coeffs_low, h2o.coeffs_high, T_mid=1000.0
        )
        
        # Properties should be continuous across the transition
        # Allow small discontinuity (polynomial fit artifact)
        cp_diff = abs(props_low[0] - props_high[0])
        assert cp_diff < 0.1, f"Cp/R discontinuity at T_mid: {cp_diff}"
    
    def test_gibbs_free_energy_consistency(self, h2o):
        """Test G = H - TS relationship."""
        T = 1000.0
        coeffs = h2o.coeffs_high
        
        h_rt = h_over_rt(T, coeffs)
        s_r = s_over_r(T, coeffs)
        g_rt = g_over_rt(T, coeffs)
        
        # G/(RT) = H/(RT) - S/R
        expected_g_rt = h_rt - s_r
        
        assert abs(g_rt - expected_g_rt) < 1e-10, \
            f"G/(RT) = {g_rt}, expected {expected_g_rt}"
    
    def test_calculate_cp_dimensional(self, h2o):
        """Test dimensional Cp calculation."""
        T = 500.0
        
        cp = calculate_cp(T, h2o)
        
        # Cp in J/(mol·K), should be approximately 33-35 for H2O at 500K
        assert 30 < cp < 40, f"Cp at 500K = {cp} J/(mol·K), expected ~33-35"
    
    def test_calculate_enthalpy_dimensional(self, h2o):
        """Test dimensional enthalpy calculation."""
        T = 298.15
        
        h = calculate_enthalpy(T, h2o)
        
        # H2O enthalpy includes formation enthalpy (~-242 kJ/mol)
        # Full H at 298K should be around -240 to -245 kJ/mol
        h_kj = h / 1000.0
        assert -250 < h_kj < -235, f"H at 298K = {h_kj} kJ/mol"
    
    def test_get_coeffs_for_temp_low(self, h2o):
        """Test coefficient selection for low temperature."""
        T = 500.0
        coeffs = h2o.get_coeffs_for_temp(T)
        
        np.testing.assert_array_equal(coeffs, h2o.coeffs_low)
    
    def test_get_coeffs_for_temp_high(self, h2o):
        """Test coefficient selection for high temperature."""
        T = 2000.0
        coeffs = h2o.get_coeffs_for_temp(T)
        
        np.testing.assert_array_equal(coeffs, h2o.coeffs_high)
    
    def test_get_coeffs_for_temp_at_midpoint(self, h2o):
        """Test coefficient selection exactly at T_mid."""
        T = 1000.0
        coeffs = h2o.get_coeffs_for_temp(T)
        
        # At T_mid, should use high temperature coefficients
        np.testing.assert_array_equal(coeffs, h2o.coeffs_high)
    
    def test_get_coeffs_out_of_range_low(self, h2o):
        """Test that out-of-range temperature raises error."""
        T = 100.0  # Below T_low = 200K
        
        with pytest.raises(ValueError, match="outside valid range"):
            h2o.get_coeffs_for_temp(T)
    
    def test_get_coeffs_out_of_range_high(self, h2o):
        """Test that out-of-range temperature raises error."""
        T = 7000.0  # Above T_high = 6000K
        
        with pytest.raises(ValueError, match="outside valid range"):
            h2o.get_coeffs_for_temp(T)


class TestMultipleSpecies:
    """Test calculations for multiple species in database."""
    
    @pytest.fixture
    def species_db(self):
        return create_sample_database()
    
    def test_all_species_have_valid_coefficients(self, species_db):
        """Ensure all species have non-zero coefficient arrays."""
        for name, species in species_db.items():
            assert species.coeffs_high.shape == (7,), f"{name} high coeffs wrong shape"
            assert species.coeffs_low.shape == (7,), f"{name} low coeffs wrong shape"
            assert np.any(species.coeffs_high != 0), f"{name} has all-zero high coeffs"
            assert np.any(species.coeffs_low != 0), f"{name} has all-zero low coeffs"
    
    def test_all_species_positive_cp(self, species_db):
        """Heat capacity must be positive for all species."""
        for name, species in species_db.items():
            for T in [300.0, 500.0, 1000.0, 2000.0]:
                cp = calculate_cp(T, species)
                assert cp > 0, f"{name} has negative Cp at {T}K: {cp}"
    
    def test_element_h2_enthalpy_zero_formation(self, species_db):
        """Reference elements should have zero formation enthalpy correction."""
        h2 = species_db['H2']
        o2 = species_db['O2']
        n2 = species_db['N2']
        
        # These are elemental reference states with Hf° = 0
        assert h2.h_formation_298 == 0.0
        assert o2.h_formation_298 == 0.0
        assert n2.h_formation_298 == 0.0


class TestNumbaCompilation:
    """Test that Numba JIT compilation works correctly."""
    
    def test_numba_functions_are_callable(self):
        """Ensure JIT-compiled functions can be called."""
        coeffs = np.array([4.0, 0.001, 0.0, 0.0, 0.0, -30000.0, 2.0])
        T = 1000.0
        
        # These should not raise any Numba compilation errors
        result_cp = cp_over_r(T, coeffs)
        result_h = h_over_rt(T, coeffs)
        result_s = s_over_r(T, coeffs)
        result_g = g_over_rt(T, coeffs)
        
        assert isinstance(result_cp, float)
        assert isinstance(result_h, float)
        assert isinstance(result_s, float)
        assert isinstance(result_g, float)
    
    def test_numba_with_array_temperatures(self):
        """Test JIT functions work with looped calculations."""
        coeffs = np.array([4.0, 0.001, 0.0, 0.0, 0.0, -30000.0, 2.0])
        temperatures = np.linspace(300, 3000, 100)
        
        results = []
        for T in temperatures:
            results.append(cp_over_r(T, coeffs))
        
        assert len(results) == 100
        assert all(r > 0 for r in results)


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_very_low_temperature(self):
        """Test behavior near lower temperature bound."""
        coeffs = np.array([4.0, 0.001, 1e-6, 0.0, 0.0, -30000.0, 2.0])
        T = 200.0
        
        result = cp_over_r(T, coeffs)
        assert np.isfinite(result), f"Non-finite result at T={T}: {result}"
    
    def test_very_high_temperature(self):
        """Test behavior near upper temperature bound."""
        coeffs = np.array([3.0, 0.001, -1e-7, 1e-11, -1e-15, -30000.0, 2.0])
        T = 6000.0
        
        result = cp_over_r(T, coeffs)
        assert np.isfinite(result), f"Non-finite result at T={T}: {result}"
        assert result > 0, f"Negative Cp at T={T}: {result}"
    
    def test_enthalpy_at_low_temperature(self):
        """Test H/(RT) doesn't blow up at low T (a6/T term)."""
        coeffs = np.array([4.0, 0.001, 0.0, 0.0, 0.0, -30000.0, 2.0])
        T = 200.0
        
        result = h_over_rt(T, coeffs)
        # a6/T = -30000/200 = -150, should be finite
        assert np.isfinite(result), f"Non-finite H/(RT) at T={T}: {result}"
    
    def test_entropy_at_low_temperature(self):
        """Test S/R with ln(T) term at low T."""
        coeffs = np.array([4.0, 0.001, 0.0, 0.0, 0.0, -30000.0, 2.0])
        T = 200.0
        
        result = s_over_r(T, coeffs)
        # a1*ln(200) ≈ 4*5.3 ≈ 21
        assert np.isfinite(result), f"Non-finite S/R at T={T}: {result}"
