"""
Advanced Nozzle Geometry - Rao Bell Contour.

Implements the Rao parabolic approximation for optimum
thrust bell nozzle profiles.
"""

from dataclasses import dataclass

import numpy as np

# numpy 2 renamed trapz to trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class NozzleContour:
    """Complete nozzle contour definition."""

    x: np.ndarray  # Axial position (m) from throat
    r: np.ndarray  # Radius (m)
    area_ratio: np.ndarray  # Local A/At
    wall_angle: np.ndarray  # Wall angle (degrees)
    contour_type: str  # 'bell', 'conical', 'parabolic'

    # Key dimensions
    throat_radius: float  # m
    exit_radius: float  # m
    length: float  # Total length (m)
    percent_bell: float  # Bell percentage (80%, 90%, etc.)


def generate_bell_contour(
    throat_radius: float,
    expansion_ratio: float,
    theta_n: float = 35.0,  # Entry angle (degrees)
    theta_e: float = 8.0,  # Exit angle (degrees)
    percent_bell: float = 80.0,  # Percent of ideal 15° cone length
    n_points: int = 100,
) -> NozzleContour:
    """
    Generate Rao parabolic bell nozzle contour.

    Uses the parabolic approximation method by G.V.R. Rao (1958).

    Args:
        throat_radius: Throat radius (m)
        expansion_ratio: Exit area ratio (Ae/At)
        theta_n: Initial expansion angle at throat (degrees)
        theta_e: Exit angle (degrees)
        percent_bell: Nozzle length as percent of 15° cone
        n_points: Number of contour points

    Returns:
        NozzleContour with full geometry definition

    Reference:
        Rao, G.V.R. "Exhaust Nozzle Contour for Optimum Thrust",
        Jet Propulsion, Vol. 28, 1958, pp. 377-382.
    """
    if throat_radius <= 0.0 or expansion_ratio <= 1.0:
        raise ValueError("Throat radius must be positive and expansion ratio above one")
    if not 0.0 < theta_e < theta_n < 90.0:
        raise ValueError("Angles must satisfy 0 < exit angle < initial angle < 90 degrees")
    if not 50.0 <= percent_bell <= 100.0:
        raise ValueError("Bell length must be 50-100 % of the 15 degree cone")
    if n_points < 8:
        raise ValueError("At least eight contour points are required")

    Rt = throat_radius
    Re = Rt * np.sqrt(expansion_ratio)

    # Downstream throat arc of radius 0.382 Rt (Rao)
    R_downstream = 0.382 * Rt

    # Reference 15° conical nozzle length
    L_cone_15 = (Re - Rt) / np.tan(np.radians(15))
    L_bell = L_cone_15 * percent_bell / 100.0

    theta_n_rad = np.radians(theta_n)

    # Starting point of parabola (end of throat arc)
    x_start = R_downstream * np.sin(theta_n_rad)
    r_start = Rt + R_downstream * (1 - np.cos(theta_n_rad))

    # End point (nozzle exit)
    x_end = L_bell
    r_end = Re

    # Parabolic coefficients
    # r = a + b*x + c*x²
    # Using boundary conditions:
    # r(x_start) = r_start
    # r(x_end) = r_end
    # dr/dx(x_start) = tan(theta_n)
    # dr/dx(x_end) = tan(theta_e)

    # We use 2 parabolas joined smoothly (Rao's method uses 2 sections)
    # For simplicity, use single parabola with bezier-like interpolation

    # Create parametric points
    t = np.linspace(0, 1, n_points)

    # Bezier-like parabolic interpolation
    x_parabola, r_parabola = _parabolic_bell(x_start, r_start, theta_n, x_end, r_end, theta_e, t)

    # Add throat circular arc section
    n_throat = n_points // 4
    theta_arc = np.linspace(np.pi / 2 + theta_n_rad, np.pi / 2, n_throat)
    x_throat = R_downstream * np.sin(theta_arc[::-1] - np.pi / 2)
    r_throat = Rt + R_downstream * (1 - np.cos(theta_arc[::-1] - np.pi / 2))

    # Combine throat + parabola
    x = np.concatenate([x_throat, x_parabola[1:]])
    r = np.concatenate([r_throat, r_parabola[1:]])

    # Calculate area ratio and wall angle
    area_ratio = (r / Rt) ** 2

    # Wall angle from gradient
    wall_angle = np.zeros_like(x)
    wall_angle[1:-1] = np.degrees(np.arctan(np.gradient(r, x)[1:-1]))
    wall_angle[0] = theta_n
    wall_angle[-1] = theta_e

    return NozzleContour(
        x=x,
        r=r,
        area_ratio=area_ratio,
        wall_angle=wall_angle,
        contour_type="bell",
        throat_radius=Rt,
        exit_radius=Re,
        length=x[-1],
        percent_bell=percent_bell,
    )


def _parabolic_bell(
    x0: float, r0: float, theta0: float, x1: float, r1: float, theta1: float, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate parabolic bell contour using quadratic Bezier approach.

    Connects two points with specified entry/exit angles.
    """
    # Control point from tangent intersection
    tan0 = np.tan(np.radians(theta0))
    tan1 = np.tan(np.radians(theta1))

    # Find intersection of tangent lines
    # Line 1: r = r0 + tan0*(x - x0)
    # Line 2: r = r1 + tan1*(x - x1)
    # x_c: r0 + tan0*(x_c - x0) = r1 + tan1*(x_c - x1)

    if abs(tan0 - tan1) > 1e-10:
        x_c = (r1 - r0 + tan0 * x0 - tan1 * x1) / (tan0 - tan1)
        r_c = r0 + tan0 * (x_c - x0)
    else:
        # Parallel tangents - use midpoint
        x_c = (x0 + x1) / 2
        r_c = (r0 + r1) / 2

    # Quadratic Bezier curve
    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x_c + t**2 * x1
    r = (1 - t) ** 2 * r0 + 2 * (1 - t) * t * r_c + t**2 * r1

    return x, r


@dataclass(frozen=True)
class ChamberContour:
    """Wall of a thrust chamber from injector face to nozzle exit.

    ``x`` is measured from the throat (negative upstream); ``r`` is the wall
    radius. ``supersonic`` flags stations downstream of the throat.
    """

    x: np.ndarray
    r: np.ndarray
    supersonic: np.ndarray
    throat_radius: float
    chamber_radius: float
    chamber_length: float  # injector face to throat, m
    nozzle_length: float  # throat to exit, m

    @property
    def area_ratio(self) -> np.ndarray:
        return (self.r / self.throat_radius) ** 2


def chamber_contour(
    throat_area: float,
    expansion_ratio: float,
    contraction_ratio: float = 4.0,
    characteristic_length: float = 1.0,
    convergence_angle: float = 30.0,
    percent_bell: float = 80.0,
    n_points: int = 160,
) -> ChamberContour:
    """Cylinder, conical convergence, throat arc and Rao bell.

    The cylinder length is chosen so that the chamber volume up to the
    throat equals L* times the throat area. The upstream throat arc has
    radius 1.5 Rt (Huzel & Huang).
    """
    if throat_area <= 0.0 or expansion_ratio <= 1.0 or contraction_ratio <= 1.0:
        raise ValueError("Throat area, expansion ratio and contraction ratio are invalid")
    if characteristic_length <= 0.0 or not 10.0 <= convergence_angle <= 60.0:
        raise ValueError("L* must be positive and the convergence angle 10-60 degrees")

    rt = np.sqrt(throat_area / np.pi)
    rc = rt * np.sqrt(contraction_ratio)
    theta = np.radians(convergence_angle)
    arc_radius = 1.5 * rt

    # Upstream throat arc, from the throat back to the tangent with the cone
    phi = np.linspace(0.0, theta, 24)
    x_arc = -arc_radius * np.sin(phi)
    r_arc = rt + arc_radius * (1.0 - np.cos(phi))
    cone_length = max(rc - r_arc[-1], 0.0) / np.tan(theta)
    x_cone_start = x_arc[-1] - cone_length

    r1 = r_arc[-1]
    convergent_volume = np.pi / 3.0 * cone_length * (rc * rc + rc * r1 + r1 * r1)
    convergent_volume += _trapezoid(np.pi * r_arc**2, -x_arc)
    cylinder_length = max(characteristic_length * throat_area - convergent_volume, 0.0) / (
        np.pi * rc * rc
    )
    x_face = x_cone_start - cylinder_length

    bell = generate_bell_contour(rt, expansion_ratio, percent_bell=percent_bell, n_points=n_points)
    upstream_x = np.concatenate(
        [
            np.linspace(x_face, x_cone_start, 12, endpoint=False),
            np.linspace(x_cone_start, x_arc[-1], 16, endpoint=False),
            x_arc[::-1],
        ]
    )
    upstream_r = np.concatenate(
        [
            np.full(12, rc),
            np.linspace(rc, r1, 16, endpoint=False),
            r_arc[::-1],
        ]
    )
    x = np.concatenate([upstream_x, bell.x[1:]])
    r = np.concatenate([upstream_r, bell.r[1:]])
    return ChamberContour(
        x=x,
        r=r,
        supersonic=x > 0.0,
        throat_radius=rt,
        chamber_radius=rc,
        chamber_length=-x_face,
        nozzle_length=float(bell.x[-1]),
    )
