"""Verification tests for regenerative-cooling correlations."""

import numpy as np
import pytest

from ensim.core.cooling import (
    CoolantType,
    CoolingChannel,
    CoolingSystemDesign,
    _mach_from_area_ratio,
    analyze_cooling_system,
    bartz_heat_transfer_coefficient,
    calculate_thermal_profile,
    coolant_heat_transfer_coefficient,
    smooth_channel_darcy_friction_factor,
)


def test_bartz_throat_value_matches_published_equation_form():
    inputs = {
        "D_throat": 0.1,
        "P_chamber": 7.0e6,
        "c_star": 1700.0,
        "T_chamber": 3500.0,
        "gamma": 1.2,
        "Pr": 0.7,
        "mu_ref": 5.0e-5,
        "area_ratio": 1.0,
        "local_diameter": 0.1,
        "throat_radius_of_curvature": 0.15,
        "wall_temperature": 1000.0,
        "molecular_weight": 22.0,
    }
    actual = bartz_heat_transfer_coefficient(**inputs)

    specific_gas_constant = 8314.46261815324 / inputs["molecular_weight"]
    heat_capacity = inputs["gamma"] * specific_gas_constant / (inputs["gamma"] - 1.0)
    temperature_factor = 1.0 + 0.5 * (inputs["gamma"] - 1.0)
    sigma = (
        0.5 * inputs["wall_temperature"] / inputs["T_chamber"] * temperature_factor + 0.5
    ) ** -0.68 * temperature_factor**-0.12
    expected = (
        0.026
        / inputs["D_throat"] ** 0.2
        * (inputs["mu_ref"] ** 0.2 * heat_capacity / inputs["Pr"] ** 0.6)
        * (inputs["P_chamber"] / inputs["c_star"]) ** 0.8
        * (inputs["D_throat"] / inputs["throat_radius_of_curvature"]) ** 0.1
        * sigma
    )
    assert actual == pytest.approx(expected, rel=2e-14)


def test_bartz_pressure_scaling_is_pc_to_power_point_eight():
    common = (0.1, 7.0e6, 1700.0, 3500.0, 1.2, 0.7, 5.0e-5, 1.0, 0.1)
    baseline = bartz_heat_transfer_coefficient(*common)
    doubled = bartz_heat_transfer_coefficient(common[0], 2.0 * common[1], *common[2:])
    assert doubled / baseline == pytest.approx(2.0**0.8, rel=2e-14)


@pytest.mark.parametrize("supersonic", [False, True])
def test_area_mach_solver_satisfies_isentropic_relation(supersonic):
    gamma = 1.2
    area_ratio = 4.0
    mach = _mach_from_area_ratio(area_ratio, gamma, supersonic)
    recovered = (2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * mach**2)) ** (
        (gamma + 1.0) / (2.0 * (gamma - 1.0))
    ) / mach
    assert recovered == pytest.approx(area_ratio, rel=2e-13)
    assert (mach > 1.0) is supersonic


def test_gnielinski_and_smooth_channel_friction_match_definitions():
    reynolds = 100_000.0
    prandtl = 2.0
    conductivity = 0.12
    diameter = 0.002
    friction = (0.79 * np.log(reynolds) - 1.64) ** -2
    nusselt = (
        (friction / 8.0)
        * (reynolds - 1000.0)
        * prandtl
        / (1.0 + 12.7 * np.sqrt(friction / 8.0) * (prandtl ** (2.0 / 3.0) - 1.0))
    )
    assert coolant_heat_transfer_coefficient(
        reynolds, prandtl, conductivity, diameter
    ) == pytest.approx(nusselt * conductivity / diameter)
    assert smooth_channel_darcy_friction_factor(1000.0) == pytest.approx(0.064)


def test_counterflow_coolant_heats_toward_chamber():
    design = CoolingSystemDesign(
        channels=CoolingChannel(0.002, 0.003, 0.001, 0.001, 40, 1.0),
        coolant=CoolantType.WATER,
        coolant_inlet_temp=290.0,
        coolant_inlet_pressure=8.0e6,
        coolant_mass_flow=2.0,
        wall_material="OFHC Copper",
        wall_thermal_conductivity=385.0,
        wall_melting_point=1200.0,
        coolant_flow_from_exit=True,
    )
    result = analyze_cooling_system(
        design,
        [(0.0, 0.1), (1.0, 0.2)],
        {
            "T_chamber": 3400.0,
            "P_chamber": 7.0e6,
            "gamma": 1.2,
            "c_star": 1700.0,
            "molecular_weight": 22.0,
        },
        num_stations=11,
    )
    assert result[0].axial_position == pytest.approx(0.0)
    assert result[-1].axial_position == pytest.approx(1.0)
    assert result[0].coolant_temp > result[-1].coolant_temp
    assert result[0].coolant_pressure < result[-1].coolant_pressure
    assert result[-1].margin_to_critical_temperature == pytest.approx(647.1 - 290.0)


def test_analysis_requires_explicit_chamber_state():
    design = CoolingSystemDesign(
        channels=CoolingChannel(0.002, 0.003, 0.001, 0.001, 40, 1.0),
        coolant=CoolantType.WATER,
        coolant_inlet_temp=290.0,
        coolant_inlet_pressure=8.0e6,
        coolant_mass_flow=2.0,
        wall_material="OFHC Copper",
        wall_thermal_conductivity=385.0,
        wall_melting_point=1200.0,
    )
    with pytest.raises(ValueError, match="Missing chamber conditions"):
        analyze_cooling_system(design, [(0.0, 0.1), (1.0, 0.2)], {})


def test_analysis_rejects_nonphysical_channel_geometry():
    design = CoolingSystemDesign(
        channels=CoolingChannel(0.0, 0.003, 0.001, 0.001, 40, 1.0),
        coolant=CoolantType.WATER,
        coolant_inlet_temp=290.0,
        coolant_inlet_pressure=8.0e6,
        coolant_mass_flow=2.0,
        wall_material="OFHC Copper",
        wall_thermal_conductivity=385.0,
        wall_melting_point=1200.0,
    )
    with pytest.raises(ValueError, match="channel dimensions"):
        analyze_cooling_system(
            design,
            [(0.0, 0.1), (1.0, 0.2)],
            {
                "T_chamber": 3400.0,
                "P_chamber": 7.0e6,
                "gamma": 1.2,
                "c_star": 1700.0,
                "molecular_weight": 22.0,
            },
        )


def test_reduced_thermal_profile_uses_explicit_conical_geometry():
    result = calculate_thermal_profile(
        T_chamber=3400.0,
        P_chamber=7.0e6,
        c_star=1700.0,
        gamma=1.2,
        throat_diameter=0.1,
        expansion_ratio=16.0,
        wall_thickness=0.001,
        wall_conductivity=300.0,
        coolant_temp=300.0,
        coolant_htc=20_000.0,
        material_limit=1500.0,
        contraction_ratio=4.0,
        convergent_half_angle_deg=45.0,
        divergent_half_angle_deg=10.0,
        num_stations=21,
    )

    expected_convergent_length = (0.2 - 0.1) / (2.0 * np.tan(np.radians(45.0)))
    expected_divergent_length = (0.4 - 0.1) / (2.0 * np.tan(np.radians(10.0)))
    assert result.x_position[0] == pytest.approx(-expected_convergent_length)
    assert result.x_position[-1] == pytest.approx(expected_divergent_length)
    assert np.all(np.isfinite(result.heat_flux))
    assert result.within_material_limit == (result.max_wall_temp < 1500.0)
