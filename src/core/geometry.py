"""
Advanced Nozzle Geometry - Rao Bell Contour.

Implements the Rao parabolic approximation for optimum
thrust bell nozzle profiles.
"""

from dataclasses import dataclass

import numpy as np


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
    theta_e: float = 8.0,   # Exit angle (degrees)
    percent_bell: float = 80.0,  # Percent of ideal 15° cone length
    n_points: int = 100
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
    Rt = throat_radius
    eps = expansion_ratio
    Re = Rt * np.sqrt(eps)

    # Throat circular arc section
    # Standard throat design: 1.5*Rt upstream circular arc
    # 0.382*Rt downstream circular arc
    1.5 * Rt
    R_downstream = 0.382 * Rt

    # Reference 15° conical nozzle length
    L_cone_15 = (Re - Rt) / np.tan(np.radians(15))
    L_bell = L_cone_15 * percent_bell / 100.0

    # Convert angles to radians
    theta_n_rad = np.radians(theta_n)
    np.radians(theta_e)

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
    x_parabola, r_parabola = _parabolic_bell(
        x_start, r_start, theta_n,
        x_end, r_end, theta_e,
        t
    )

    # Add throat circular arc section
    n_throat = n_points // 4
    theta_arc = np.linspace(np.pi/2 + theta_n_rad, np.pi/2, n_throat)
    x_throat = R_downstream * np.sin(theta_arc[::-1] - np.pi/2)
    r_throat = Rt + R_downstream * (1 - np.cos(theta_arc[::-1] - np.pi/2))

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
        contour_type='bell',
        throat_radius=Rt,
        exit_radius=Re,
        length=x[-1],
        percent_bell=percent_bell
    )


def _parabolic_bell(
    x0: float, r0: float, theta0: float,
    x1: float, r1: float, theta1: float,
    t: np.ndarray
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
        x_c = (r1 - r0 + tan0*x0 - tan1*x1) / (tan0 - tan1)
        r_c = r0 + tan0 * (x_c - x0)
    else:
        # Parallel tangents - use midpoint
        x_c = (x0 + x1) / 2
        r_c = (r0 + r1) / 2

    # Quadratic Bezier curve
    x = (1-t)**2 * x0 + 2*(1-t)*t * x_c + t**2 * x1
    r = (1-t)**2 * r0 + 2*(1-t)*t * r_c + t**2 * r1

    return x, r


def generate_conical_contour(
    throat_radius: float,
    expansion_ratio: float,
    half_angle: float = 15.0,
    n_points: int = 100
) -> NozzleContour:
    """
    Generate simple conical nozzle contour.

    Args:
        throat_radius: Throat radius (m)
        expansion_ratio: Exit area ratio
        half_angle: Nozzle half-angle (degrees)
        n_points: Number of points

    Returns:
        NozzleContour
    """
    Rt = throat_radius
    Re = Rt * np.sqrt(expansion_ratio)

    L = (Re - Rt) / np.tan(np.radians(half_angle))

    x = np.linspace(0, L, n_points)
    r = Rt + x * np.tan(np.radians(half_angle))

    area_ratio = (r / Rt) ** 2
    wall_angle = np.full_like(x, half_angle)

    return NozzleContour(
        x=x,
        r=r,
        area_ratio=area_ratio,
        wall_angle=wall_angle,
        contour_type='conical',
        throat_radius=Rt,
        exit_radius=Re,
        length=L,
        percent_bell=100.0  # 100% of 15° cone for reference
    )


def contour_to_3d_mesh(contour: NozzleContour, n_circumferential: int = 36):
    """
    Convert 2D contour to 3D mesh points for visualization.

    Args:
        contour: NozzleContour object
        n_circumferential: Points around circumference

    Returns:
        Tuple of (x_grid, y_grid, z_grid) for surface rendering
    """
    theta = np.linspace(0, 2*np.pi, n_circumferential)

    X, Theta = np.meshgrid(contour.x, theta)
    R = np.tile(contour.r, (n_circumferential, 1))

    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)

    return X, Y, Z
