"""Barrowman small-angle static-stability relations for slender finned rockets."""

from dataclasses import dataclass

import numpy as np

from ensim.core.rocket import BodyTube, FinSet, NoseCone, NoseShape, Rocket


@dataclass
class AeroResult:
    """Normal-force, center-of-pressure and static-margin result."""

    cn_alpha_nose: float
    cn_alpha_body: float
    cn_alpha_fins: float
    cn_alpha_total: float
    cp_nose: float
    cp_body: float
    cp_fins: float
    cp_total: float
    cg: float
    stability_margin: float
    is_stable: bool
    axial_drag_coefficient: float


def calculate_cn_alpha_nose(nose: NoseCone, reference_diameter: float | None = None) -> float:
    """Return the Barrowman nose normal-force slope in radian inverse."""
    diameter = nose.diameter if reference_diameter is None else reference_diameter
    if nose.diameter <= 0.0 or diameter <= 0.0:
        raise ValueError("Nose and reference diameters must be positive")
    return 2.0 * (nose.diameter / diameter) ** 2


def calculate_cp_nose(nose: NoseCone) -> float:
    """Return nose center of pressure measured from the tip."""
    if nose.length <= 0.0:
        raise ValueError("Nose length must be positive")
    factors = {
        NoseShape.CONICAL: 2.0 / 3.0,
        NoseShape.OGIVE: 0.466,
        NoseShape.PARABOLIC: 0.5,
        NoseShape.ELLIPTICAL: 1.0 / 3.0,
    }
    if nose.shape not in factors:
        raise ValueError(f"Unsupported nose shape: {nose.shape.value}")
    return factors[nose.shape] * nose.length


def calculate_cn_alpha_body(body: BodyTube) -> float:
    """Return the zero contribution of a constant-diameter cylindrical section."""
    if body.length <= 0.0 or body.diameter <= 0.0:
        raise ValueError("Body dimensions must be positive")
    return 0.0


def calculate_cn_alpha_fins(fins: FinSet, body_diameter: float) -> float:
    """Return the Barrowman fin-set normal-force slope in radian inverse."""
    fin = fins.fin
    if (
        fins.count <= 0
        or body_diameter <= 0.0
        or fin.span <= 0.0
        or fin.root_chord <= 0.0
        or fin.tip_chord < 0.0
        or fin.root_chord + fin.tip_chord <= 0.0
    ):
        raise ValueError("Fin count, body diameter and fin dimensions are invalid")

    sweep_displacement = np.tan(np.radians(fin.sweep_angle)) * fin.span
    mid_chord_displacement = sweep_displacement + 0.5 * (fin.tip_chord - fin.root_chord)
    mid_chord_line = np.hypot(fin.span, mid_chord_displacement)
    body_radius = 0.5 * body_diameter
    interference = 1.0 + body_radius / (body_radius + fin.span)
    denominator = 1.0 + np.sqrt(
        1.0 + (2.0 * mid_chord_line / (fin.root_chord + fin.tip_chord)) ** 2
    )
    return interference * 4.0 * fins.count * (fin.span / body_diameter) ** 2 / denominator


def calculate_cp_fins(fins: FinSet, leading_edge_position: float) -> float:
    """Return fin-set center of pressure measured from the nose tip."""
    fin = fins.fin
    chord_sum = fin.root_chord + fin.tip_chord
    if fin.span <= 0.0 or fin.root_chord <= 0.0 or fin.tip_chord < 0.0 or chord_sum <= 0.0:
        raise ValueError("Fin dimensions are invalid")
    sweep_displacement = np.tan(np.radians(fin.sweep_angle)) * fin.span
    local_cp = (
        sweep_displacement * (fin.root_chord + 2.0 * fin.tip_chord) / (3.0 * chord_sum)
        + (chord_sum - fin.root_chord * fin.tip_chord / chord_sum) / 6.0
    )
    return leading_edge_position + local_cp


def calculate_total_cp(rocket: Rocket) -> tuple[float, float, dict[str, float]]:
    """Combine component centers of pressure by normal-force-slope weighting."""
    reference_diameter = rocket.reference_diameter
    cn_nose = calculate_cn_alpha_nose(rocket.nose, reference_diameter)
    cn_body = calculate_cn_alpha_body(rocket.body)
    cn_fins = calculate_cn_alpha_fins(rocket.fins, reference_diameter)
    cp_nose = calculate_cp_nose(rocket.nose)
    cp_fins = calculate_cp_fins(rocket.fins, rocket.fins.position)
    cn_total = cn_nose + cn_body + cn_fins
    if cn_total <= 0.0:
        raise ValueError("Total normal-force slope must be positive")
    cp_total = (cn_nose * cp_nose + cn_fins * cp_fins) / cn_total
    return (
        cp_total,
        cn_total,
        {
            "cn_nose": cn_nose,
            "cn_body": cn_body,
            "cn_fins": cn_fins,
            "cp_nose": cp_nose,
            "cp_fins": cp_fins,
        },
    )


def calculate_stability_margin(rocket: Rocket, time: float = 0.0) -> float:
    """Return static margin in body calibers, positive when CP is aft of CG."""
    cp_total, _, _ = calculate_total_cp(rocket)
    return (cp_total - rocket.get_cg_at_time(time)) / rocket.reference_diameter


def analyze_rocket(rocket: Rocket, time: float = 0.0) -> AeroResult:
    """Evaluate the implemented Barrowman static-stability subset."""
    cp_total, cn_total, components = calculate_total_cp(rocket)
    cg = rocket.get_cg_at_time(time)
    margin = (cp_total - cg) / rocket.reference_diameter
    return AeroResult(
        cn_alpha_nose=components["cn_nose"],
        cn_alpha_body=components["cn_body"],
        cn_alpha_fins=components["cn_fins"],
        cn_alpha_total=cn_total,
        cp_nose=components["cp_nose"],
        cp_body=rocket.nose.length + 0.5 * rocket.body.length,
        cp_fins=components["cp_fins"],
        cp_total=cp_total,
        cg=cg,
        stability_margin=margin,
        is_stable=margin >= 1.0,
        axial_drag_coefficient=rocket.axial_drag_coefficient,
    )
