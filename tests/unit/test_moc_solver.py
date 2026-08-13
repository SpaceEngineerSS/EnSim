import numpy as np
import pytest

from ensim.core.moc_solver import (
    generate_mln_contour,
    inverse_prandtl_meyer,
    prandtl_meyer_angle,
)


def _critical_area_ratio(mach, gamma):
    factor = 2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * mach**2)
    return factor ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))) / mach


def test_prandtl_meyer_reference_value_and_inverse():
    angle = prandtl_meyer_angle(2.0, 1.4)
    assert np.degrees(angle) == pytest.approx(26.3798, abs=1e-4)
    assert inverse_prandtl_meyer(angle, 1.4) == pytest.approx(2.0, abs=1e-8)


def test_planar_minimum_length_contour_reaches_design_exit():
    contour, mesh = generate_mln_contour(3.0, gamma=1.2, throat_radius=0.05, n_char_lines=40)

    assert contour.geometry == "planar"
    assert len(contour.x) == 41
    assert len(mesh.points) == 40
    assert np.all(np.diff(contour.x) > 0.0)
    assert np.all(np.diff(contour.y) > 0.0)
    assert contour.M[-1] == pytest.approx(3.0, abs=1e-8)
    assert contour.theta[-1] == pytest.approx(0.0, abs=1e-12)
    assert contour.exit_radius / contour.throat_radius == pytest.approx(
        _critical_area_ratio(3.0, 1.2), rel=2e-3
    )


def test_contour_converges_as_characteristic_count_increases():
    coarse, _ = generate_mln_contour(3.0, gamma=1.2, n_char_lines=10)
    fine, _ = generate_mln_contour(3.0, gamma=1.2, n_char_lines=40)
    exact = _critical_area_ratio(3.0, 1.2)
    assert abs(fine.exit_radius - exact) < abs(coarse.exit_radius - exact)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"M_exit": 1.0},
        {"M_exit": 3.0, "gamma": 1.0},
        {"M_exit": 3.0, "throat_radius": 0.0},
        {"M_exit": 3.0, "n_char_lines": 2},
        {"M_exit": 3.0, "use_variable_gamma": True},
    ],
)
def test_invalid_or_unsupported_design_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_mln_contour(**kwargs)
