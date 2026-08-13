import pytest

from ensim.core.recovery import Parachute, estimate_descent, size_parachute


def test_terminal_descent_rate_satisfies_force_balance():
    chute = Parachute(diameter=1.2, cd=1.5)
    mass = 8.0
    density = 1.1
    velocity = chute.get_descent_rate(mass, density)

    drag = 0.5 * density * velocity**2 * chute.cda
    assert drag == pytest.approx(mass * 9.80665)


def test_parachute_sizing_recovers_target_velocity():
    diameter = size_parachute(mass=12.0, target_descent_rate=6.0, cd=1.4, rho=1.225)
    chute = Parachute(diameter=diameter, cd=1.4)
    assert chute.get_descent_rate(12.0, 1.225) == pytest.approx(6.0)


def test_descent_summary_reports_kinematics_without_safety_verdict():
    summary = estimate_descent(1_000.0, 10.0, Parachute(diameter=1.0), rho_avg=1.1)
    assert set(summary) == {"descent_rate", "descent_time", "kinetic_energy"}
    assert summary["descent_time"] == pytest.approx(1_000.0 / summary["descent_rate"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 0.0},
        {"target_descent_rate": 0.0},
        {"cd": 0.0},
        {"rho": 0.0},
    ],
)
def test_parachute_sizing_rejects_nonpositive_inputs(kwargs):
    inputs = {"mass": 10.0, "target_descent_rate": 5.0, "cd": 1.5, "rho": 1.225}
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        size_parachute(**inputs)
