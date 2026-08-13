import numpy as np
import pytest

from ensim.core.aero import analyze_rocket, calculate_cn_alpha_fins, calculate_total_cp
from ensim.core.rocket import create_default_rocket


def test_fin_normal_force_uses_mid_chord_line_length():
    rocket = create_default_rocket()
    fin = rocket.fins.fin
    sweep = np.tan(np.radians(fin.sweep_angle)) * fin.span
    mid_chord = np.hypot(fin.span, sweep + 0.5 * (fin.tip_chord - fin.root_chord))
    interference = 1.0 + 0.5 * rocket.body.diameter / (0.5 * rocket.body.diameter + fin.span)
    expected = (
        interference
        * 4.0
        * rocket.fins.count
        * (fin.span / rocket.body.diameter) ** 2
        / (1.0 + np.sqrt(1.0 + (2.0 * mid_chord / (fin.root_chord + fin.tip_chord)) ** 2))
    )
    assert calculate_cn_alpha_fins(rocket.fins, rocket.body.diameter) == pytest.approx(expected)


def test_total_cp_is_normal_force_weighted_and_aft_of_default_cg():
    rocket = create_default_rocket()
    cp, cn_total, components = calculate_total_cp(rocket)
    expected = (
        components["cn_nose"] * components["cp_nose"]
        + components["cn_fins"] * components["cp_fins"]
    ) / cn_total
    result = analyze_rocket(rocket)

    assert cp == pytest.approx(expected)
    assert result.stability_margin >= 1.0
    assert result.is_stable
    assert result.axial_drag_coefficient == pytest.approx(0.45)
