"""
Unit tests for shifting equilibrium module.

Tests the shifting equilibrium nozzle flow calculations
and comparison with frozen flow.
"""

import pytest
import numpy as np
from src.core.shifting_equilibrium import (
    FlowStation,
    ShiftingFlowResult,
    solve_shifting_equilibrium,
    compare_frozen_vs_shifting,
    isentropic_temperature_ratio,
    isentropic_pressure_ratio,
    solve_mach_supersonic,
)


class TestIsentropicRelations:
    """Test basic isentropic flow relations."""
    
    def test_temperature_ratio_at_mach_one(self):
        """At M=1, T/T0 = 2/(γ+1)."""
        gamma = 1.2
        T_ratio = isentropic_temperature_ratio(1.0, gamma)
        expected = 2.0 / (gamma + 1.0)
        assert abs(T_ratio - expected) < 1e-10
    
    def test_temperature_ratio_decreases_with_mach(self):
        """Higher Mach should give lower T/T0."""
        gamma = 1.2
        T1 = isentropic_temperature_ratio(2.0, gamma)
        T2 = isentropic_temperature_ratio(3.0, gamma)
        assert T2 < T1
    
    def test_pressure_ratio_at_mach_one(self):
        """At M=1, P/P0 = (2/(γ+1))^(γ/(γ-1))."""
        gamma = 1.2
        P_ratio = isentropic_pressure_ratio(1.0, gamma)
        expected = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        assert abs(P_ratio - expected) < 1e-10


class TestMachSolver:
    """Test supersonic Mach number solver."""
    
    def test_area_ratio_1_gives_mach_1(self):
        """A/A* = 1 should give M ≈ 1."""
        gamma = 1.2
        M = solve_mach_supersonic(1.001, gamma)  # Just above 1
        assert abs(M - 1.0) < 0.01
    
    def test_larger_area_gives_higher_mach(self):
        """Larger area ratio should give higher Mach."""
        gamma = 1.2
        M1 = solve_mach_supersonic(5.0, gamma)
        M2 = solve_mach_supersonic(50.0, gamma)
        assert M2 > M1
    
    def test_typical_expansion_ratio(self):
        """Test typical expansion ratio."""
        gamma = 1.2
        M = solve_mach_supersonic(50.0, gamma)
        # For ε=50, γ=1.2, M should be around 4-5
        assert 3.5 < M < 5.5


class TestShiftingEquilibrium:
    """Test shifting equilibrium solver."""
    
    @pytest.fixture
    def h2o2_initial_composition(self):
        """Typical H2/O2 combustion products at 3600K."""
        return {
            'H2O': 0.68,
            'H2': 0.12,
            'OH': 0.10,
            'H': 0.04,
            'O2': 0.04,
            'O': 0.02
        }
    
    def test_solver_returns_result(self, h2o2_initial_composition):
        """Solver should return ShiftingFlowResult."""
        area_ratios = np.linspace(1.0, 50.0, 20)
        
        result = solve_shifting_equilibrium(
            T_chamber=3600.0,
            P_chamber=1e7,  # 100 bar
            gamma_chamber=1.2,
            mean_mw_chamber=18.0,
            initial_composition=h2o2_initial_composition,
            area_ratios=area_ratios
        )
        
        assert isinstance(result, ShiftingFlowResult)
        assert len(result.stations) == len(area_ratios)
    
    def test_temperature_decreases_through_nozzle(self, h2o2_initial_composition):
        """Temperature should decrease through nozzle."""
        area_ratios = np.linspace(1.0, 50.0, 20)
        
        result = solve_shifting_equilibrium(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma_chamber=1.2,
            mean_mw_chamber=18.0,
            initial_composition=h2o2_initial_composition,
            area_ratios=area_ratios
        )
        
        temps = [s.temperature for s in result.stations]
        # Temperature should generally decrease
        assert temps[-1] < temps[0]
    
    def test_velocity_increases_through_nozzle(self, h2o2_initial_composition):
        """Velocity should increase through nozzle."""
        area_ratios = np.linspace(1.0, 50.0, 20)
        
        result = solve_shifting_equilibrium(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma_chamber=1.2,
            mean_mw_chamber=18.0,
            initial_composition=h2o2_initial_composition,
            area_ratios=area_ratios
        )
        
        velocities = [s.velocity for s in result.stations]
        # Velocity should increase
        assert velocities[-1] > velocities[0]
    
    def test_isp_improvement_positive(self, h2o2_initial_composition):
        """Shifting should give higher Isp than frozen."""
        area_ratios = np.linspace(1.0, 50.0, 20)
        
        result = solve_shifting_equilibrium(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma_chamber=1.2,
            mean_mw_chamber=18.0,
            initial_composition=h2o2_initial_composition,
            area_ratios=area_ratios,
            frozen_comparison=True
        )
        
        # Shifting should be >= frozen (recombination adds energy)
        assert result.isp_shifting >= result.isp_frozen
    
    def test_reasonable_isp_values(self, h2o2_initial_composition):
        """Isp should be in reasonable range for H2/O2."""
        area_ratios = np.linspace(1.0, 50.0, 20)
        
        result = solve_shifting_equilibrium(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma_chamber=1.2,
            mean_mw_chamber=18.0,
            initial_composition=h2o2_initial_composition,
            area_ratios=area_ratios
        )
        
        # H2/O2 should give Isp in 400-500s range
        assert 350 < result.isp_shifting < 550


class TestComparison:
    """Test frozen vs shifting comparison function."""
    
    def test_comparison_returns_string(self):
        """Comparison should return formatted string."""
        result = compare_frozen_vs_shifting(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma=1.2,
            mean_mw=18.0,
            expansion_ratio=50.0
        )
        
        assert isinstance(result, str)
        assert "Isp" in result
    
    def test_comparison_shows_improvement(self):
        """Comparison should show improvement percentage."""
        result = compare_frozen_vs_shifting(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma=1.2,
            mean_mw=18.0,
            expansion_ratio=50.0
        )
        
        assert "%" in result


class TestPropertyProfiles:
    """Test property profile extraction."""
    
    def test_get_property_profile(self):
        """Should be able to extract property profiles."""
        initial_comp = {'H2O': 0.8, 'H2': 0.1, 'OH': 0.1}
        area_ratios = np.linspace(1.0, 50.0, 20)
        
        result = solve_shifting_equilibrium(
            T_chamber=3600.0,
            P_chamber=1e7,
            gamma_chamber=1.2,
            mean_mw_chamber=18.0,
            initial_composition=initial_comp,
            area_ratios=area_ratios
        )
        
        temp_profile = result.get_property_profile('temperature')
        mach_profile = result.get_property_profile('mach')
        
        assert len(temp_profile) == 20
        assert len(mach_profile) == 20
        assert np.all(mach_profile >= 1.0)  # All supersonic
