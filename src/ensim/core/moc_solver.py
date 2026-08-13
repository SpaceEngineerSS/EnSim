"""Planar minimum-length nozzle design by the method of characteristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GAMMA_DEFAULT = 1.2


@dataclass(frozen=True)
class CharacteristicPoint:
    x: float
    y: float
    M: float
    theta: float
    nu: float
    T: float = 0.0
    P: float = 0.0


@dataclass
class MOCMesh:
    points: list[list[CharacteristicPoint]] = field(default_factory=list)
    wall_points: list[CharacteristicPoint] = field(default_factory=list)
    centerline_points: list[CharacteristicPoint] = field(default_factory=list)
    M_exit: float = 2.0
    gamma: float = GAMMA_DEFAULT
    throat_radius: float = 1.0
    x_mesh: np.ndarray | None = None
    y_mesh: np.ndarray | None = None
    mach_mesh: np.ndarray | None = None


@dataclass(frozen=True)
class NozzleContourMOC:
    x: np.ndarray
    y: np.ndarray
    M: np.ndarray
    theta: np.ndarray
    M_exit: float
    gamma: float
    throat_radius: float
    exit_radius: float
    length: float
    geometry: str = "planar"


def _validate_gamma(gamma: float) -> None:
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")


def prandtl_meyer_angle(M: float, gamma: float) -> float:
    """Return the calorically-perfect-gas Prandtl-Meyer angle in radians."""
    _validate_gamma(gamma)
    if not np.isfinite(M) or M < 1.0:
        raise ValueError("Mach number must be finite and at least one")
    root = np.sqrt(M * M - 1.0)
    ratio = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    return float(ratio * np.arctan(root / ratio) - np.arctan(root))


def inverse_prandtl_meyer(
    nu: float, gamma: float, tol: float = 1e-10, max_iter: int = 100
) -> float:
    """Invert the Prandtl-Meyer function with a bracketed bisection solve."""
    _validate_gamma(gamma)
    if not np.isfinite(nu) or nu < 0.0:
        raise ValueError("Prandtl-Meyer angle must be finite and non-negative")
    nu_limit = 0.5 * np.pi * (np.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    if nu >= nu_limit:
        raise ValueError("Prandtl-Meyer angle exceeds its finite-Mach limit")
    if nu == 0.0:
        return 1.0

    lower, upper = 1.0, 2.0
    while prandtl_meyer_angle(upper, gamma) < nu:
        upper *= 2.0
    for _ in range(max_iter):
        midpoint = 0.5 * (lower + upper)
        residual = prandtl_meyer_angle(midpoint, gamma) - nu
        if abs(residual) <= tol:
            return midpoint
        if residual < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def mach_angle(M: float) -> float:
    if not np.isfinite(M) or M < 1.0:
        raise ValueError("Mach number must be finite and at least one")
    return float(np.arcsin(1.0 / M))


def isentropic_temperature_ratio(M: float, gamma: float) -> float:
    _validate_gamma(gamma)
    return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * M * M)


def isentropic_pressure_ratio(M: float, gamma: float) -> float:
    return isentropic_temperature_ratio(M, gamma) ** (gamma / (gamma - 1.0))


def _intersection(a: tuple[float, float], slope_a: float, b: tuple[float, float], slope_b: float):
    denominator = slope_a - slope_b
    if abs(denominator) < 1e-12:
        raise ArithmeticError("Characteristic lines are numerically parallel")
    x = (b[1] - a[1] + slope_a * a[0] - slope_b * b[0]) / denominator
    return x, a[1] + slope_a * (x - a[0])


def _flow_point(x, y, theta, nu, gamma, T0, P0) -> CharacteristicPoint:
    mach = inverse_prandtl_meyer(float(nu), gamma)
    return CharacteristicPoint(
        x=float(x),
        y=float(y),
        M=mach,
        theta=float(theta),
        nu=float(nu),
        T=float(T0 * isentropic_temperature_ratio(mach, gamma)),
        P=float(P0 * isentropic_pressure_ratio(mach, gamma)),
    )


def generate_mln_contour(
    M_exit: float,
    gamma: float = GAMMA_DEFAULT,
    throat_radius: float = 1.0,
    n_char_lines: int = 20,
    T0: float = 3000.0,
    P0: float = 1e7,
    use_variable_gamma: bool = False,
) -> tuple[NozzleContourMOC, MOCMesh]:
    """Design a symmetric, planar, sharp-corner minimum-length nozzle.

    The solution assumes steady, inviscid, irrotational, isentropic flow of a
    calorically perfect gas. ``throat_radius`` is retained for API compatibility
    and represents the planar throat half-height.
    """
    _validate_gamma(gamma)
    if not np.isfinite(M_exit) or M_exit <= 1.0:
        raise ValueError("M_exit must be finite and greater than one")
    if not np.isfinite(throat_radius) or throat_radius <= 0.0:
        raise ValueError("throat_radius must be finite and positive")
    if not isinstance(n_char_lines, (int, np.integer)) or n_char_lines < 3:
        raise ValueError("n_char_lines must be an integer of at least three")
    if not np.isfinite(T0) or T0 <= 0.0 or not np.isfinite(P0) or P0 <= 0.0:
        raise ValueError("stagnation temperature and pressure must be positive")
    if use_variable_gamma:
        raise ValueError(
            "variable-gamma MOC is not implemented; supply a representative constant gamma"
        )

    nu_exit = prandtl_meyer_angle(M_exit, gamma)
    theta_max = 0.5 * nu_exit
    theta_first = min(np.deg2rad(0.05), theta_max / n_char_lines)
    theta_step = (theta_max - theta_first) / (n_char_lines - 1)

    families: list[list[CharacteristicPoint]] = []
    invariants: list[dict[str, np.ndarray]] = []

    for family_index in range(n_char_lines):
        count = n_char_lines + 1 - family_index
        theta = np.zeros(count)
        nu = np.zeros(count)
        k_minus = np.zeros(count)
        k_plus = np.zeros(count)
        mach = np.zeros(count)
        mu = np.zeros(count)

        for point_index in range(count - 1):
            if family_index == 0:
                theta[point_index] = theta_first + theta_step * point_index
                nu[point_index] = theta[point_index]
                k_minus[point_index] = theta[point_index] + nu[point_index]
            else:
                k_minus[point_index] = invariants[family_index - 1]["k_minus"][point_index + 1]
                if point_index == 0:
                    theta[point_index] = 0.0
                    nu[point_index] = k_minus[point_index]
                else:
                    k_plus_in = k_plus[point_index - 1]
                    theta[point_index] = 0.5 * (k_minus[point_index] + k_plus_in)
                    nu[point_index] = 0.5 * (k_minus[point_index] - k_plus_in)
            k_plus[point_index] = theta[point_index] - nu[point_index]
            mach[point_index] = inverse_prandtl_meyer(float(nu[point_index]), gamma)
            mu[point_index] = mach_angle(float(mach[point_index]))

        for array in (theta, nu, k_minus, k_plus, mach, mu):
            array[-1] = array[-2]

        x = np.zeros(count)
        y = np.zeros(count)
        for point_index in range(count):
            if family_index == 0:
                if point_index == 0:
                    slope = np.tan(theta[0] - mu[0])
                    x[0] = -1.0 / slope
                    y[0] = 0.0
                elif point_index < count - 1:
                    incoming = np.tan(
                        0.5
                        * (
                            theta[point_index - 1]
                            + mu[point_index - 1]
                            + theta[point_index]
                            + mu[point_index]
                        )
                    )
                    ray = np.tan(theta[point_index] - mu[point_index])
                    x[point_index], y[point_index] = _intersection(
                        (x[point_index - 1], y[point_index - 1]), incoming, (0.0, 1.0), ray
                    )
                else:
                    incoming = np.tan(
                        0.5
                        * (
                            theta[point_index - 1]
                            + mu[point_index - 1]
                            + theta[point_index]
                            + mu[point_index]
                        )
                    )
                    wall_slope = np.tan(0.5 * (theta_max + theta[point_index]))
                    x[point_index], y[point_index] = _intersection(
                        (x[point_index - 1], y[point_index - 1]), incoming, (0.0, 1.0), wall_slope
                    )
            else:
                previous = invariants[family_index - 1]
                if point_index == 0:
                    incoming = np.tan(
                        0.5 * (theta[0] + previous["theta"][1] - mu[0] - previous["mu"][1])
                    )
                    x[0] = previous["x"][1] - previous["y"][1] / incoming
                    y[0] = 0.0
                elif point_index < count - 1:
                    left_slope = np.tan(
                        0.5
                        * (
                            theta[point_index - 1]
                            + mu[point_index - 1]
                            + theta[point_index]
                            + mu[point_index]
                        )
                    )
                    right_slope = np.tan(
                        0.5
                        * (
                            theta[point_index]
                            + previous["theta"][point_index + 1]
                            - mu[point_index]
                            - previous["mu"][point_index + 1]
                        )
                    )
                    x[point_index], y[point_index] = _intersection(
                        (x[point_index - 1], y[point_index - 1]),
                        left_slope,
                        (previous["x"][point_index + 1], previous["y"][point_index + 1]),
                        right_slope,
                    )
                else:
                    previous_wall = (previous["x"][-1], previous["y"][-1])
                    left_slope = np.tan(
                        0.5
                        * (
                            theta[point_index - 1]
                            + mu[point_index - 1]
                            + theta[point_index]
                            + mu[point_index]
                        )
                    )
                    wall_slope = np.tan(0.5 * (previous["theta"][-1] + theta[point_index]))
                    x[point_index], y[point_index] = _intersection(
                        (x[point_index - 1], y[point_index - 1]),
                        left_slope,
                        previous_wall,
                        wall_slope,
                    )

        x_dimensional = x * throat_radius
        y_dimensional = y * throat_radius
        family = [
            _flow_point(x_dimensional[j], y_dimensional[j], theta[j], nu[j], gamma, T0, P0)
            for j in range(count)
        ]
        families.append(family)
        invariants.append(
            {
                "theta": theta,
                "nu": nu,
                "k_minus": k_minus,
                "k_plus": k_plus,
                "mach": mach,
                "mu": mu,
                "x": x,
                "y": y,
            }
        )

    throat = CharacteristicPoint(
        0.0,
        throat_radius,
        1.0,
        theta_max,
        0.0,
        T0 * 2.0 / (gamma + 1.0),
        P0 * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0)),
    )
    wall = [throat, *(family[-1] for family in families)]
    centerline = [family[0] for family in families]
    contour = NozzleContourMOC(
        x=np.asarray([point.x for point in wall]),
        y=np.asarray([point.y for point in wall]),
        M=np.asarray([point.M for point in wall]),
        theta=np.asarray([point.theta for point in wall]),
        M_exit=M_exit,
        gamma=gamma,
        throat_radius=throat_radius,
        exit_radius=wall[-1].y,
        length=wall[-1].x,
    )
    mesh = MOCMesh(
        points=families,
        wall_points=wall,
        centerline_points=centerline,
        M_exit=M_exit,
        gamma=gamma,
        throat_radius=throat_radius,
    )
    _create_mesh_arrays(mesh)
    return contour, mesh


def _create_mesh_arrays(mesh: MOCMesh) -> None:
    width = max(map(len, mesh.points))
    shape = (len(mesh.points), width)
    mesh.x_mesh = np.full(shape, np.nan)
    mesh.y_mesh = np.full(shape, np.nan)
    mesh.mach_mesh = np.full(shape, np.nan)
    for row, family in enumerate(mesh.points):
        for column, point in enumerate(family):
            mesh.x_mesh[row, column] = point.x
            mesh.y_mesh[row, column] = point.y
            mesh.mach_mesh[row, column] = point.M


def compare_contours(
    moc_contour: NozzleContourMOC, rao_x: np.ndarray, rao_y: np.ndarray
) -> dict[str, float]:
    if len(rao_x) < 2 or len(rao_x) != len(rao_y):
        raise ValueError("Rao coordinate arrays must have equal lengths of at least two")
    difference = moc_contour.y - np.interp(moc_contour.x, rao_x, rao_y)
    return {
        "max_diff": float(np.max(np.abs(difference))),
        "mean_diff": float(np.mean(difference)),
        "moc_length": moc_contour.length,
        "moc_exit_radius": moc_contour.exit_radius,
    }


def export_contour_csv(contour: NozzleContourMOC, filepath: str | Path) -> None:
    data = np.column_stack((contour.x, contour.y, contour.M, contour.theta))
    np.savetxt(filepath, data, delimiter=",", header="x_m,y_m,Mach,theta_rad", comments="")


def export_mesh_vtk(mesh: MOCMesh, filepath: str | Path) -> None:
    try:
        import pyvista as pv
    except ImportError as error:
        raise ImportError("PyVista is required for VTK export") from error
    if mesh.x_mesh is None or mesh.y_mesh is None or mesh.mach_mesh is None:
        raise ValueError("mesh arrays are not initialized")
    valid = np.isfinite(mesh.x_mesh) & np.isfinite(mesh.y_mesh) & np.isfinite(mesh.mach_mesh)
    points = np.column_stack(
        (mesh.x_mesh[valid], mesh.y_mesh[valid], np.zeros(np.count_nonzero(valid)))
    )
    cloud = pv.PolyData(points)
    cloud["Mach"] = mesh.mach_mesh[valid]
    cloud.save(filepath)
