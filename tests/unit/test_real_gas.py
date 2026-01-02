"""
Unit tests for real gas thermodynamics and engine performance.

Tests the integration of:
- NASA thermo polynomial loader
- Combustion lookup tables
- RocketEngine with altitude correction
- Flow separation detection

References:
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed.
    - NASA RP-1311 (Gordon & McBride)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.chemistry import (
    load_nasa_thermo_dat,
    nasa_get_cp_r,
    nasa_get_h_rt,
    nasa_get_s_r,
    create_combustion_lookup_table,
    lookup_combustion_properties,
    bilinear_interpolate,
    GAS_CONSTANT,
)
from src.core.rocket_engine import (
    RocketEngine,
    FlowRegime,
    FlowStatus,
    calculate_flow_status,
)


# =============================================================================
# NASA Thermo Loader Tests
# =============================================================================

class TestNASAThermoLoader:
    """Test NASA 7-term polynomial loader and functions."""
    
    @pytest.fixture
    def species_db(self):
        """Use sample database for reliable testing."""
        from src.utils.nasa_parser import create_sample_database
        return create_sample_database()
    
    def test_load_species(self, species_db):
        """Should load all species from database."""
        assert len(species_db) > 0, "Should load at least one species"
        assert 'H2' in species_db, "Should load H2"
        assert 'O2' in species_db, "Should load O2"
        assert 'H2O' in species_db, "Should load H2O"
    
    def test_species_data_structure(self, species_db):
        """Species data should have correct structure."""
        h2 = species_db['H2']
        
        assert h2.name == 'H2'
        assert h2.molecular_weight > 0
        assert h2.coeffs_low.shape == (7,)
        assert h2.coeffs_high.shape == (7,)
    
    def test_h2_molecular_weight(self, species_db):
        """H2 molecular weight should be ~2 g/mol."""
        assert abs(species_db['H2'].molecular_weight - 2.016) < 0.1
    
    def test_cp_h2_at_1000k(self, species_db):
        """H2 Cp at 1000K should be ~29 J/(mol·K)."""
        h2 = species_db['H2']
        
        cp_r = nasa_get_cp_r(1000.0, h2.coeffs_low, h2.coeffs_high, h2.t_mid)
        cp = cp_r * GAS_CONSTANT
        
        # H2 Cp at 1000K is approximately 29-30 J/(mol·K)
        assert 28.0 < cp < 32.0, f"H2 Cp at 1000K should be ~29 J/(mol·K), got {cp:.1f}"
    
    def test_cp_increases_with_temperature(self, species_db):
        """Cp should generally increase with temperature for diatomics."""
        h2 = species_db['H2']
        
        cp_500 = nasa_get_cp_r(500.0, h2.coeffs_low, h2.coeffs_high, h2.t_mid)
        cp_2000 = nasa_get_cp_r(2000.0, h2.coeffs_low, h2.coeffs_high, h2.t_mid)
        
        assert cp_2000 > cp_500, "Cp should increase with temperature"
    
    def test_enthalpy_increases_with_temperature(self, species_db):
        """Enthalpy should increase with temperature."""
        h2 = species_db['H2']
        
        h_500 = nasa_get_h_rt(500.0, h2.coeffs_low, h2.coeffs_high, h2.t_mid) * 500.0
        h_2000 = nasa_get_h_rt(2000.0, h2.coeffs_low, h2.coeffs_high, h2.t_mid) * 2000.0
        
        assert h_2000 > h_500, "Enthalpy should increase with temperature"
    
    def test_entropy_increases_with_temperature(self):
        """Entropy should increase with temperature."""
        data = load_nasa_thermo_dat()
        h2 = data['H2']
        
        s_500 = nasa_get_s_r(500.0, h2['coeffs_low'], h2['coeffs_high'], 1000.0)
        s_2000 = nasa_get_s_r(2000.0, h2['coeffs_low'], h2['coeffs_high'], 1000.0)
        
        assert s_2000 > s_500, "Entropy should increase with temperature"


# =============================================================================
# Bilinear Interpolation Tests
# =============================================================================

class TestBilinearInterpolation:
    """Test bilinear interpolation for lookup tables."""
    
    def test_corner_values(self):
        """Interpolation at grid corners should return exact values."""
        x_grid = np.array([1.0, 2.0, 3.0])
        y_grid = np.array([10.0, 20.0])
        values = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        # Test corners
        assert_allclose(bilinear_interpolate(1.0, 10.0, x_grid, y_grid, values), 1.0)
        assert_allclose(bilinear_interpolate(3.0, 20.0, x_grid, y_grid, values), 6.0)
    
    def test_midpoint_interpolation(self):
        """Interpolation at midpoint should average corners."""
        x_grid = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        values = np.array([
            [0.0, 2.0],
            [2.0, 4.0]
        ])
        
        # Midpoint (0.5, 0.5) should give average = 2.0
        result = bilinear_interpolate(0.5, 0.5, x_grid, y_grid, values)
        assert_allclose(result, 2.0, atol=0.001)
    
    def test_clamping_at_boundaries(self):
        """Values outside grid should clamp to boundary values."""
        x_grid = np.array([1.0, 2.0])
        y_grid = np.array([10.0, 20.0])
        values = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        
        # Below x range
        result = bilinear_interpolate(0.0, 15.0, x_grid, y_grid, values)
        expected = 1.5  # Midpoint of left edge
        assert_allclose(result, expected, atol=0.001)


# =============================================================================
# Combustion Lookup Table Tests
# =============================================================================

class TestCombustionLookupTable:
    """Test combustion property lookup table."""
    
    def test_table_creation(self):
        """Should create lookup table with correct shape."""
        of = np.array([2.0, 2.5, 3.0])
        Pc = np.array([1e6, 3e6, 5e6])
        
        table = create_combustion_lookup_table(of, Pc)
        
        assert table['T_chamber'].shape == (3, 3)
        assert table['gamma'].shape == (3, 3)
        assert table['M_mol'].shape == (3, 3)
    
    def test_temperature_range(self):
        """Chamber temperature should be in reasonable range."""
        of = np.array([2.0, 2.5, 3.0, 3.5])
        Pc = np.array([1e6, 3e6, 5e6])
        
        table = create_combustion_lookup_table(of, Pc)
        
        assert np.all(table['T_chamber'] >= 2000), "T_c should be >= 2000K"
        assert np.all(table['T_chamber'] < 4000), "T_c should be < 4000K"
    
    def test_gamma_range(self):
        """Gamma should be in reasonable range for combustion products."""
        of = np.array([2.0, 3.0])
        Pc = np.array([3e6])
        
        table = create_combustion_lookup_table(of, Pc)
        
        assert np.all(table['gamma'] >= 1.1), "Gamma should be >= 1.1"
        assert np.all(table['gamma'] <= 1.3), "Gamma should be <= 1.3"
    
    def test_lookup_interpolation(self):
        """Lookup should interpolate between grid points."""
        of = np.array([2.0, 3.0])
        Pc = np.array([1e6, 5e6])
        
        table = create_combustion_lookup_table(of, Pc)
        
        # Lookup at grid point
        T1 = lookup_combustion_properties(
            2.0, 1e6,
            table['of_grid'], table['Pc_grid'],
            table['T_chamber'], table['gamma'], table['M_mol']
        )[0]
        
        # Should match table value
        assert_allclose(T1, table['T_chamber'][0, 0], atol=1.0)


# =============================================================================
# Rocket Engine Tests
# =============================================================================

class TestRocketEngine:
    """Test RocketEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine (similar to Merlin 1D)."""
        return RocketEngine(
            Pc_design=10e6,    # 10 MPa
            At=0.03,           # ~0.03 m² throat
            epsilon=16.0,      # Moderate expansion ratio
            T_chamber=3500,    # K
            gamma=1.2,
            M_mol=22.0,        # g/mol
        )
    
    @pytest.fixture
    def high_expansion_engine(self):
        """Create high-expansion engine (like vacuum stage)."""
        return RocketEngine(
            Pc_design=5e6,
            At=0.05,
            epsilon=80.0,      # High expansion (vacuum optimized)
            T_chamber=3500,
            gamma=1.2,
            M_mol=22.0,
        )
    
    def test_vacuum_thrust_positive(self, engine):
        """Vacuum thrust should be positive."""
        state = engine.update_state(P_ambient=0.0, throttle=1.0)
        assert state.thrust > 0, "Vacuum thrust should be positive"
    
    def test_vacuum_isp_reasonable(self, engine):
        """Vacuum Isp should be in reasonable range."""
        state = engine.update_state(P_ambient=0.0, throttle=1.0)
        assert 250 < state.isp < 400, f"Isp should be 250-400s, got {state.isp:.0f}"
    
    def test_isp_increases_with_altitude(self, engine):
        """ISP should increase as altitude increases (pressure decreases)."""
        P_sl = 101325.0   # Sea level
        P_10km = 26500.0  # ~10 km altitude
        P_40km = 287.0    # ~40 km altitude
        P_vac = 0.0       # Vacuum
        
        state_sl = engine.update_state(P_ambient=P_sl)
        state_10km = engine.update_state(P_ambient=P_10km)
        state_40km = engine.update_state(P_ambient=P_40km)
        state_vac = engine.update_state(P_ambient=P_vac)
        
        assert state_10km.isp > state_sl.isp, "Isp at 10km should exceed sea level"
        assert state_40km.isp > state_10km.isp, "Isp at 40km should exceed 10km"
        assert state_vac.isp >= state_40km.isp, "Vacuum Isp should be highest"
    
    def test_thrust_increases_with_altitude(self, engine):
        """Thrust should increase as altitude increases."""
        state_sl = engine.update_state(P_ambient=101325.0)
        state_vac = engine.update_state(P_ambient=0.0)
        
        assert state_vac.thrust > state_sl.thrust, "Vacuum thrust should exceed SL"
    
    def test_flow_separation_high_expansion_at_sl(self, high_expansion_engine):
        """High-expansion nozzle should show separation at sea level."""
        state = high_expansion_engine.update_state(P_ambient=101325.0)
        
        assert state.flow_status.regime in [FlowRegime.SEPARATED, FlowRegime.OVEREXPANDED]
        assert state.flow_status.thrust_loss_factor < 1.0
    
    def test_flow_attached_in_vacuum(self, high_expansion_engine):
        """Even high-expansion nozzle should be attached in vacuum."""
        state = high_expansion_engine.update_state(P_ambient=0.0)
        
        assert state.flow_status.regime == FlowRegime.ATTACHED
        assert state.flow_status.thrust_loss_factor == 1.0
    
    def test_throttle_reduces_thrust(self, engine):
        """Throttle < 1.0 should reduce thrust proportionally."""
        state_full = engine.update_state(P_ambient=0.0, throttle=1.0)
        state_half = engine.update_state(P_ambient=0.0, throttle=0.5)
        
        # Thrust should be approximately proportional to throttle
        ratio = state_half.thrust / state_full.thrust
        assert 0.4 < ratio < 0.6, f"Thrust ratio should be ~0.5, got {ratio:.2f}"
    
    def test_engine_off_zero_thrust(self, engine):
        """Zero throttle should give zero thrust."""
        state = engine.update_state(P_ambient=0.0, throttle=0.0)
        
        assert state.thrust == 0.0
        assert state.is_firing == False


# =============================================================================
# Flow Status Tests
# =============================================================================

class TestFlowStatus:
    """Test flow status calculation."""
    
    def test_vacuum_attached(self):
        """Flow in vacuum should always be attached."""
        status = calculate_flow_status(P_exit=1000.0, P_ambient=0.0)
        
        assert status.regime == FlowRegime.ATTACHED
        assert status.thrust_loss_factor == 1.0
        assert status.side_load_warning == False
    
    def test_underexpanded(self):
        """Pe > Pa should be underexpanded."""
        status = calculate_flow_status(P_exit=200000.0, P_ambient=100000.0)
        
        assert status.regime == FlowRegime.UNDEREXPANDED
        assert status.thrust_loss_factor == 1.0
    
    def test_summerfield_separation(self):
        """Pe/Pa < 0.4 should trigger separation."""
        # Pe/Pa = 0.3 < 0.4 (Summerfield threshold)
        status = calculate_flow_status(P_exit=30000.0, P_ambient=100000.0)
        
        assert status.regime == FlowRegime.SEPARATED
        assert status.thrust_loss_factor < 1.0
        assert status.side_load_warning == True
    
    def test_mild_overexpansion(self):
        """0.4 < Pe/Pa < 0.6 should be mildly overexpanded."""
        # Pe/Pa = 0.5
        status = calculate_flow_status(P_exit=50000.0, P_ambient=100000.0)
        
        assert status.regime == FlowRegime.OVEREXPANDED
        assert 0.85 < status.thrust_loss_factor < 1.0
    
    def test_thrust_loss_factor_range(self):
        """Thrust loss factor should always be between 0.5 and 1.0."""
        # Very low Pe/Pa
        status = calculate_flow_status(P_exit=1000.0, P_ambient=100000.0)
        
        assert 0.5 <= status.thrust_loss_factor <= 1.0
