"""
Unit tests for boundary layer corrections in nozzle flow.

Tests the boundary layer displacement thickness, loss factor,
and throat Reynolds number calculations.
"""

import numpy as np
import pytest

from src.core.propulsion import (
    calculate_boundary_layer_displacement_thickness,
    calculate_boundary_layer_loss_factor,
    calculate_throat_reynolds,
)


class TestBoundaryLayerDisplacementThickness:
    """Tests for displacement thickness calculation."""

    def test_zero_reynolds_returns_zero(self):
        """Zero Reynolds number should give zero thickness."""
        delta = calculate_boundary_layer_displacement_thickness(
            Re_throat=0.0,
            throat_radius=0.05,
            x_position=0.1,
            M_local=2.0
        )
        assert delta == 0.0

    def test_zero_position_returns_zero(self):
        """Zero position should give zero thickness."""
        delta = calculate_boundary_layer_displacement_thickness(
            Re_throat=1e6,
            throat_radius=0.05,
            x_position=0.0,
            M_local=2.0
        )
        assert delta == 0.0

    def test_turbulent_regime(self):
        """High Reynolds number should give turbulent BL."""
        delta = calculate_boundary_layer_displacement_thickness(
            Re_throat=1e6,
            throat_radius=0.05,
            x_position=0.2,
            M_local=2.5
        )
        # Should be positive and reasonable (mm scale)
        assert 0 < delta < 0.01  # Less than 1 cm

    def test_compressibility_effect(self):
        """Higher Mach should increase displacement thickness."""
        delta_low_mach = calculate_boundary_layer_displacement_thickness(
            Re_throat=1e6,
            throat_radius=0.05,
            x_position=0.2,
            M_local=1.5
        )
        delta_high_mach = calculate_boundary_layer_displacement_thickness(
            Re_throat=1e6,
            throat_radius=0.05,
            x_position=0.2,
            M_local=3.5
        )
        assert delta_high_mach > delta_low_mach


class TestBoundaryLayerLossFactor:
    """Tests for overall boundary layer loss factor."""

    def test_returns_reasonable_range(self):
        """Loss factor should be between 0.95 and 1.0."""
        factor = calculate_boundary_layer_loss_factor(
            Re_throat=1e6,
            area_ratio=50.0,
            throat_radius=0.05,
            nozzle_length=0.5,
            gamma=1.2
        )
        assert 0.95 <= factor <= 1.0

    def test_higher_reynolds_less_loss(self):
        """Higher Reynolds should give less loss (higher factor)."""
        factor_low_re = calculate_boundary_layer_loss_factor(
            Re_throat=1e5,
            area_ratio=50.0,
            throat_radius=0.05,
            nozzle_length=0.5
        )
        factor_high_re = calculate_boundary_layer_loss_factor(
            Re_throat=1e7,
            area_ratio=50.0,
            throat_radius=0.05,
            nozzle_length=0.5
        )
        assert factor_high_re >= factor_low_re

    def test_zero_reynolds_default_loss(self):
        """Zero Reynolds should return default 2% loss."""
        factor = calculate_boundary_layer_loss_factor(
            Re_throat=0.0,
            area_ratio=50.0,
            throat_radius=0.05,
            nozzle_length=0.5
        )
        assert factor == pytest.approx(0.98, abs=0.001)


class TestThroatReynolds:
    """Tests for throat Reynolds number calculation."""

    def test_positive_reynolds(self):
        """Should return positive Reynolds number."""
        Re = calculate_throat_reynolds(
            P_chamber=7e6,  # 70 bar
            T_chamber=3500.0,
            throat_diameter=0.1,
            mean_molecular_weight=18.0,
            gamma=1.2
        )
        assert Re > 0

    def test_reynolds_order_of_magnitude(self):
        """Reynolds should be ~1e6-1e8 for typical rocket."""
        Re = calculate_throat_reynolds(
            P_chamber=7e6,
            T_chamber=3500.0,
            throat_diameter=0.1,
            mean_molecular_weight=20.0,
            gamma=1.2
        )
        assert 1e5 < Re < 1e9

    def test_higher_pressure_higher_reynolds(self):
        """Higher chamber pressure should give higher Reynolds."""
        Re_low = calculate_throat_reynolds(
            P_chamber=1e6,
            T_chamber=3500.0,
            throat_diameter=0.1,
            mean_molecular_weight=20.0
        )
        Re_high = calculate_throat_reynolds(
            P_chamber=10e6,
            T_chamber=3500.0,
            throat_diameter=0.1,
            mean_molecular_weight=20.0
        )
        assert Re_high > Re_low

    def test_larger_throat_higher_reynolds(self):
        """Larger throat diameter should give higher Reynolds."""
        Re_small = calculate_throat_reynolds(
            P_chamber=7e6,
            T_chamber=3500.0,
            throat_diameter=0.05,
            mean_molecular_weight=20.0
        )
        Re_large = calculate_throat_reynolds(
            P_chamber=7e6,
            T_chamber=3500.0,
            throat_diameter=0.2,
            mean_molecular_weight=20.0
        )
        assert Re_large > Re_small


class TestPerformanceWithBoundaryLayer:
    """Integration tests for performance with BL correction."""

    def test_performance_with_bl_correction(self):
        """Performance calculation should work with BL correction."""
        from src.core.propulsion import NozzleConditions, calculate_performance

        nozzle = NozzleConditions(
            area_ratio=50.0,
            chamber_pressure=7e6,
            ambient_pressure=0.0,
            throat_area=0.01  # 10 cm² = 0.001 m²
        )

        result = calculate_performance(
            T_chamber=3500.0,
            P_chamber=7e6,
            gamma=1.2,
            mean_molecular_weight=18.0,
            nozzle=nozzle,
            include_boundary_layer=True
        )

        # Isp should be reasonable (300-500 s)
        assert 250 < result.isp < 550

    def test_bl_reduces_performance(self):
        """BL correction should reduce performance slightly."""
        from src.core.propulsion import NozzleConditions, calculate_performance

        nozzle = NozzleConditions(
            area_ratio=50.0,
            chamber_pressure=7e6,
            ambient_pressure=0.0,
            throat_area=0.01
        )

        result_with_bl = calculate_performance(
            T_chamber=3500.0,
            P_chamber=7e6,
            gamma=1.2,
            mean_molecular_weight=18.0,
            nozzle=nozzle,
            include_boundary_layer=True
        )

        result_without_bl = calculate_performance(
            T_chamber=3500.0,
            P_chamber=7e6,
            gamma=1.2,
            mean_molecular_weight=18.0,
            nozzle=nozzle,
            include_boundary_layer=False
        )

        # With BL should be slightly lower
        assert result_with_bl.isp <= result_without_bl.isp

