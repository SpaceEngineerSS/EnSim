import numpy as np
import pytest

from ensim.core.geometry import generate_bell_contour
from ensim.core.propulsion import get_nozzle_profile


def test_bell_contour_meets_endpoint_geometry_and_angles():
    contour = generate_bell_contour(
        throat_radius=0.05,
        expansion_ratio=40.0,
        theta_n=35.0,
        theta_e=8.0,
        percent_bell=80.0,
        n_points=100,
    )

    assert np.all(np.diff(contour.x) > 0.0)
    assert contour.area_ratio.min() == pytest.approx(1.0)
    assert contour.area_ratio[-1] == pytest.approx(40.0)
    assert contour.wall_angle[0] == pytest.approx(35.0)
    assert contour.wall_angle[-1] == pytest.approx(8.0)


def test_nozzle_profile_uses_supplied_local_area_ratios():
    area_ratios = np.array([1.0, 1.25, 2.5, 8.0, 40.0])
    profile = get_nozzle_profile(
        gamma=1.2,
        T_chamber=3500.0,
        P_chamber=7.0e6,
        exit_area_ratio=40.0,
        R_specific=380.0,
        area_ratios=area_ratios,
    )

    assert np.array_equal(profile["area_ratio"], area_ratios)
    assert np.all(np.diff(profile["mach"]) > 0.0)
    assert np.all(np.diff(profile["temperature"]) < 0.0)


def test_chamber_contour_honours_l_star_and_area_ratios():
    import numpy as np

    from ensim.core.geometry import chamber_contour

    at = 1.0e-3
    contour = chamber_contour(
        at, expansion_ratio=25.0, contraction_ratio=4.0, characteristic_length=1.1
    )
    assert np.all(np.diff(contour.x) > 0.0)
    assert contour.area_ratio[0] == pytest.approx(4.0)
    assert contour.area_ratio[-1] == pytest.approx(25.0, rel=1e-6)
    assert contour.r.min() == pytest.approx(contour.throat_radius)
    upstream = contour.x <= 0.0
    volume = np.sum(
        np.pi
        * 0.5
        * (contour.r[upstream][1:] ** 2 + contour.r[upstream][:-1] ** 2)
        * np.diff(contour.x[upstream])
    )
    assert volume == pytest.approx(1.1 * at, rel=0.02)
