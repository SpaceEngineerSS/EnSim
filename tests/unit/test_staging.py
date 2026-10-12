import numpy as np
import pytest

from ensim.core.constants import G0
from ensim.core.staging import (
    StageEngine,
    create_custom_vehicle,
    create_heavy_lift_reference,
    create_medium_lift_reference,
)


def test_reference_first_stage_thrust_is_cluster_total():
    medium = create_medium_lift_reference()
    heavy = create_heavy_lift_reference()

    # Nine 845 kN engines and five 6.77 MN engines at sea level
    assert medium.stages[0].engine.get_thrust(0.0) == pytest.approx(9 * 845_000, rel=1e-6)
    assert heavy.stages[0].engine.get_thrust(0.0) == pytest.approx(5 * 6_770_000, rel=1e-6)
    assert heavy.stages[1].engine.get_thrust(200_000.0) == pytest.approx(5 * 1_033_000, rel=1e-6)


def test_reference_liftoff_thrust_to_weight_is_realistic():
    for vehicle in (create_medium_lift_reference(), create_heavy_lift_reference()):
        ratio = vehicle.get_thrust(0.0) / (vehicle.total_mass * G0)
        assert 1.1 < ratio < 1.8


def test_thrust_follows_ambient_pressure_between_sea_level_and_vacuum():
    engine = StageEngine(
        thrust_sl_per_engine=800.0, thrust_vac_per_engine=1000.0, isp_sl=250.0, isp_vac=300.0
    )
    assert engine.get_thrust(0.0) == pytest.approx(800.0)
    assert engine.get_thrust(150_000.0) == pytest.approx(1000.0, rel=1e-6)
    # US76: p(11 km) / p0 = 0.22403
    assert engine.get_thrust(11_000.0) == pytest.approx(1000.0 - 200.0 * 0.22403, rel=1e-3)


def test_mass_flow_is_constant_with_altitude_and_scales_with_throttle():
    engine = StageEngine(
        thrust_sl_per_engine=800.0,
        thrust_vac_per_engine=1000.0,
        isp_sl=240.0,
        isp_vac=300.0,
        num_engines=3,
        throttle_min=0.5,
    )
    expected = 3 * 1000.0 / (300.0 * G0)
    assert engine.get_mass_flow_rate(0.0) == pytest.approx(expected)
    assert engine.get_mass_flow_rate(50_000.0) == pytest.approx(expected)
    assert engine.get_mass_flow_rate(0.0, throttle=0.7) == pytest.approx(0.7 * expected)
    assert engine.get_mass_flow_rate(0.0, throttle=0.2) == pytest.approx(0.5 * expected)


def test_zero_throttle_shuts_engines_down():
    engine = StageEngine(
        thrust_sl_per_engine=0.0, thrust_vac_per_engine=1000.0, isp_sl=0.0, isp_vac=300.0
    )
    assert engine.get_thrust(100_000.0, throttle=0.0) == 0.0
    assert engine.get_mass_flow_rate(throttle=0.0) == 0.0


def test_total_delta_v_is_sum_of_serial_stages():
    vehicle = create_custom_vehicle(
        [
            {
                "dry_mass": 1000.0,
                "propellant_mass": 9000.0,
                "thrust_vac_per_engine": 2e5,
                "isp_vac": 300.0,
            },
            {
                "dry_mass": 100.0,
                "propellant_mass": 900.0,
                "thrust_vac_per_engine": 2e4,
                "isp_vac": 350.0,
            },
        ],
        payload_mass=50.0,
    )
    upper = 350.0 * G0 * np.log((100.0 + 900.0 + 50.0) / (100.0 + 50.0))
    lower = 300.0 * G0 * np.log((1000.0 + 9000.0 + 1050.0) / (1000.0 + 1050.0))
    assert vehicle.get_total_delta_v() == pytest.approx(upper + lower)


def test_custom_vehicle_requires_explicit_thrust():
    with pytest.raises(KeyError):
        create_custom_vehicle([{"dry_mass": 1.0, "propellant_mass": 1.0, "isp_vac": 300.0}])


def test_engine_validation():
    with pytest.raises(ValueError):
        StageEngine(
            thrust_sl_per_engine=0.0,
            thrust_vac_per_engine=1.0,
            isp_sl=0.0,
            isp_vac=300.0,
            num_engines=0,
        )
    with pytest.raises(ValueError):
        StageEngine(
            thrust_sl_per_engine=0.0,
            thrust_vac_per_engine=1.0,
            isp_sl=0.0,
            isp_vac=300.0,
            throttle_min=0.9,
            throttle_max=0.5,
        )
