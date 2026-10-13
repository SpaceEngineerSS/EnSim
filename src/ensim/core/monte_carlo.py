"""
Monte Carlo dispersion of 6-DOF flights.

Each run perturbs thrust, Isp, burn time, drag, launch elevation, fin
misalignment and wind, flies the vehicle with :func:`simulate_flight_6dof`
and records the impact point and performance. Runs are independent, so they
are spread over worker processes.

Landing statistics use only runs that reached the ground before
``max_time``; runs still airborne are counted separately.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2, truncnorm

from ensim.core.flight_6dof import PhysicsViolationError, SimulationCancelled, simulate_flight_6dof
from ensim.core.rocket import Rocket
from ensim.core.sampling import lognormal_factor

ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


@dataclass
class DispersionConfig:
    """Uncertainty model for a dispersion study.

    Multiplicative factors (thrust, Isp, burn time, Cd) are lognormal with mean
    one and the given coefficient of variation. Launch elevation is normal,
    fin misalignment uniform, wind speed a normal truncated at zero.
    """

    num_simulations: int = 100

    thrust_sigma: float = 0.02
    isp_sigma: float = 0.01
    burn_time_sigma: float = 0.01
    cd_sigma: float = 0.05

    fin_misalignment_max: float = 0.5  # deg, uniform in [-max, max]
    launch_angle_sigma: float = 0.5  # deg

    wind_speed_mean: float = 3.0  # m/s at 10 m
    wind_speed_sigma: float = 2.0
    wind_direction_mean: float = 0.0  # deg, direction the wind blows from
    wind_direction_sigma: float = 30.0
    randomize_wind_dir: bool = True  # uniform over 0-360 deg instead

    seed: int | None = None
    n_workers: int | None = None  # None = all cores, 1 = run in this process
    ellipse_confidence: float = 0.997

    def __post_init__(self) -> None:
        if self.num_simulations < 2:
            raise ValueError("At least two simulations are required")
        magnitudes = (
            self.thrust_sigma,
            self.isp_sigma,
            self.burn_time_sigma,
            self.cd_sigma,
            self.fin_misalignment_max,
            self.launch_angle_sigma,
            self.wind_speed_sigma,
            self.wind_direction_sigma,
        )
        if any(value < 0.0 for value in magnitudes):
            raise ValueError("Dispersion magnitudes cannot be negative")
        if self.wind_speed_mean < 0.0:
            raise ValueError("Mean wind speed cannot be negative")
        if not 0.0 < self.ellipse_confidence < 1.0:
            raise ValueError("Ellipse confidence must lie between zero and one")
        if self.n_workers is not None and self.n_workers < 1:
            raise ValueError("Worker count must be positive")


@dataclass
class SimulationResult:
    """Outcome of one dispersed flight."""

    run_id: int
    landing_x: float  # m east of the pad, NaN if not landed
    landing_y: float  # m north of the pad, NaN if not landed
    apogee: float  # m
    max_velocity: float  # m/s
    max_mach: float
    burnout_altitude: float  # m
    flight_time: float  # s
    thrust_factor: float
    isp_factor: float
    cd_factor: float
    wind_speed: float
    wind_direction: float
    landed: bool = False
    success: bool = True
    error_message: str = ""


@dataclass
class DispersionResult:
    """Statistics of a dispersion study."""

    config: DispersionConfig
    total_time_seconds: float
    simulations_per_second: float

    landing_x_mean: float
    landing_y_mean: float
    landing_x_std: float
    landing_y_std: float
    cep_radius: float  # m, median miss distance from the mean impact point
    ellipse_semi_major: float
    ellipse_semi_minor: float
    ellipse_angle_deg: float  # from east, counter-clockwise
    ellipse_confidence: float

    apogee_mean: float
    apogee_std: float
    max_velocity_mean: float
    max_velocity_std: float
    flight_time_mean: float
    flight_time_std: float

    landing_points: np.ndarray  # (n_landed, 2)
    apogees: np.ndarray  # (n_successful,)

    results: list[SimulationResult] = field(default_factory=list)
    n_successful: int = 0
    n_failed: int = 0
    n_landed: int = 0

    @property
    def num_simulations(self) -> int:
        return self.n_successful + self.n_failed

    @property
    def success_rate(self) -> float:
        total = self.n_successful + self.n_failed
        return self.n_successful / total if total else 0.0


def compute_cep(points: np.ndarray, center: np.ndarray) -> float:
    """Radius around ``center`` that contains half of ``points``."""
    distances = np.sqrt(np.sum((np.asarray(points) - center) ** 2, axis=1))
    return float(np.percentile(distances, 50))


def compute_confidence_ellipse(
    points: np.ndarray, confidence: float = 0.997
) -> tuple[float, float, float]:
    """Covariance ellipse enclosing ``confidence`` of a bivariate normal fit.

    Returns (semi-major, semi-minor, angle of the major axis from x in degrees).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("At least two two-dimensional points are required")
    if not np.all(np.isfinite(points)) or not 0.0 < confidence < 1.0:
        raise ValueError("Points and confidence level must be finite and valid")

    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    scale = chi2.ppf(confidence, df=2)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return (
        float(np.sqrt(scale * eigenvalues[0])),
        float(np.sqrt(scale * eigenvalues[1])),
        float(angle),
    )


def _generate_perturbations(config: DispersionConfig, rng: np.random.Generator) -> dict[str, float]:
    if config.wind_speed_sigma == 0.0:
        wind_speed = config.wind_speed_mean
    else:
        lower = -config.wind_speed_mean / config.wind_speed_sigma
        wind_speed = float(
            truncnorm.rvs(
                lower,
                np.inf,
                loc=config.wind_speed_mean,
                scale=config.wind_speed_sigma,
                random_state=rng,
            )
        )
    if config.randomize_wind_dir:
        wind_direction = float(rng.uniform(0.0, 360.0))
    else:
        wind_direction = float(
            (config.wind_direction_mean + rng.normal(0.0, config.wind_direction_sigma)) % 360.0
        )
    return {
        "thrust_factor": lognormal_factor(rng, config.thrust_sigma),
        "isp_factor": lognormal_factor(rng, config.isp_sigma),
        "burn_time_factor": lognormal_factor(rng, config.burn_time_sigma),
        "cd_factor": lognormal_factor(rng, config.cd_sigma),
        "launch_angle_offset": float(rng.normal(0.0, config.launch_angle_sigma)),
        "fin_misalignment": float(
            rng.uniform(-config.fin_misalignment_max, config.fin_misalignment_max)
        ),
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
    }


def _run_single_simulation(
    args: tuple[int, Rocket, dict[str, Any], dict[str, float]],
) -> SimulationResult:
    """Fly one dispersed case. Runs in a worker process."""
    run_id, rocket, base, perturbation = args
    common = {
        "run_id": run_id,
        "thrust_factor": perturbation["thrust_factor"],
        "isp_factor": perturbation["isp_factor"],
        "cd_factor": perturbation["cd_factor"],
        "wind_speed": perturbation["wind_speed"],
        "wind_direction": perturbation["wind_direction"],
    }
    elevation = min(90.0, base["launch_angle_deg"] + perturbation["launch_angle_offset"])
    try:
        flight = simulate_flight_6dof(
            rocket=rocket,
            thrust_vac=base["thrust_vac"] * perturbation["thrust_factor"],
            isp_vac=base["isp_vac"] * perturbation["isp_factor"],
            burn_time=base["burn_time"] * perturbation["burn_time_factor"],
            launch_angle_deg=elevation,
            wind_speed=perturbation["wind_speed"],
            wind_direction_deg=perturbation["wind_direction"],
            cd_factor=perturbation["cd_factor"],
            fin_misalignment_deg=perturbation["fin_misalignment"],
            **base["flight_options"],
        )
    except (PhysicsViolationError, ValueError, FloatingPointError) as error:
        return SimulationResult(
            landing_x=np.nan,
            landing_y=np.nan,
            apogee=0.0,
            max_velocity=0.0,
            max_mach=0.0,
            burnout_altitude=0.0,
            flight_time=0.0,
            success=False,
            error_message=str(error),
            **common,
        )

    landed = flight.ground_impact and flight.landing_x is not None
    return SimulationResult(
        landing_x=flight.landing_x if landed else np.nan,
        landing_y=flight.landing_y if landed else np.nan,
        apogee=flight.apogee_altitude,
        max_velocity=flight.max_velocity,
        max_mach=flight.max_mach,
        burnout_altitude=flight.burnout_altitude,
        flight_time=flight.flight_time,
        landed=landed,
        success=flight.success,
        error_message="" if flight.success else (flight.abort_reason or ""),
        **common,
    )


def run_monte_carlo(
    rocket: Rocket,
    thrust_vac: float,
    isp_vac: float,
    burn_time: float,
    config: DispersionConfig | None = None,
    launch_angle_deg: float = 85.0,
    launch_azimuth_deg: float = 0.0,
    exit_area: float = 0.0,
    dt: float = 0.02,
    max_time: float = 300.0,
    use_adaptive: bool = False,
    rail_length: float = 3.0,
    chute_diameter: float = 0.0,
    deploy_at_apogee: bool = True,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    flight_options: dict[str, Any] | None = None,
) -> DispersionResult:
    """Run a dispersion study.

    Args:
        rocket: Nominal vehicle. Each run receives its own copy.
        thrust_vac: Nominal vacuum thrust, N.
        isp_vac: Nominal vacuum specific impulse, s.
        burn_time: Nominal burn time, s.
        config: Uncertainty model; defaults to :class:`DispersionConfig`.
        launch_angle_deg: Nominal rail elevation above the horizon, deg.
        launch_azimuth_deg: Rail heading clockwise from north, deg.
        exit_area: Nozzle exit area, m², for the ambient-pressure thrust loss.
        dt: Integration step, s.
        max_time: Time limit of each flight, s.
        use_adaptive: Use the adaptive integrator.
        rail_length: Launch rail length, m.
        chute_diameter: Main parachute diameter, m (0 = none).
        deploy_at_apogee: Open the main at apogee rather than at a set height.
        progress: Called with (completed, total) after each run.
        cancel: Polled between runs; return True to stop.
        flight_options: Further keyword arguments for every flight (drogue,
            fill time, WGS-84 site, ...). They override the explicit ones.

    Raises:
        SimulationCancelled: ``cancel`` returned True.
        RuntimeError: Fewer than two runs succeeded.
    """
    config = config or DispersionConfig()
    if (
        thrust_vac <= 0.0
        or isp_vac <= 0.0
        or burn_time <= 0.0
        or exit_area < 0.0
        or dt <= 0.0
        or max_time <= 0.0
        or not 0.0 < launch_angle_deg <= 90.0
    ):
        raise ValueError("Nominal flight and integration inputs are invalid")

    rng = np.random.default_rng(config.seed)
    base = {
        "thrust_vac": thrust_vac,
        "isp_vac": isp_vac,
        "burn_time": burn_time,
        "launch_angle_deg": launch_angle_deg,
        "flight_options": {
            "exit_area": exit_area,
            "dt": dt,
            "max_time": max_time,
            "launch_azimuth_deg": launch_azimuth_deg,
            "use_adaptive": use_adaptive,
            "rail_length": rail_length,
            "chute_diameter": chute_diameter,
            "deploy_at_apogee": deploy_at_apogee,
            **(flight_options or {}),
        },
    }
    jobs = [
        (index, rocket, base, _generate_perturbations(config, rng))
        for index in range(config.num_simulations)
    ]
    total = len(jobs)
    results: list[SimulationResult] = []
    start = time.perf_counter()

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise SimulationCancelled(f"Cancelled after {len(results)} of {total} runs")

    workers = config.n_workers or multiprocessing.cpu_count()
    if workers == 1:
        for job in jobs:
            check_cancel()
            results.append(_run_single_simulation(job))
            if progress:
                progress(len(results), total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(_run_single_simulation, job) for job in jobs}
            try:
                while pending:
                    done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        results.append(future.result())
                        if progress:
                            progress(len(results), total)
                    check_cancel()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    elapsed = max(time.perf_counter() - start, 1e-9)

    results.sort(key=lambda r: r.run_id)
    successful = [r for r in results if r.success]
    landed = [r for r in successful if r.landed]
    if len(successful) < 2:
        reasons = {r.error_message for r in results if not r.success}
        raise RuntimeError(
            "Not enough successful simulations for analysis: " + "; ".join(sorted(reasons))
        )

    apogees = np.array([r.apogee for r in successful])
    velocities = np.array([r.max_velocity for r in successful])
    flight_times = np.array([r.flight_time for r in successful])
    landing_points = np.array([[r.landing_x, r.landing_y] for r in landed]).reshape(-1, 2)

    if len(landed) >= 2:
        mean = landing_points.mean(axis=0)
        std = landing_points.std(axis=0, ddof=1)
        cep = compute_cep(landing_points, mean)
        major, minor, angle = compute_confidence_ellipse(landing_points, config.ellipse_confidence)
    else:
        mean = std = np.full(2, np.nan)
        cep = major = minor = angle = float("nan")

    return DispersionResult(
        config=config,
        total_time_seconds=elapsed,
        simulations_per_second=len(results) / elapsed,
        landing_x_mean=float(mean[0]),
        landing_y_mean=float(mean[1]),
        landing_x_std=float(std[0]),
        landing_y_std=float(std[1]),
        cep_radius=cep,
        ellipse_semi_major=major,
        ellipse_semi_minor=minor,
        ellipse_angle_deg=angle,
        ellipse_confidence=config.ellipse_confidence,
        apogee_mean=float(apogees.mean()),
        apogee_std=float(apogees.std(ddof=1)),
        max_velocity_mean=float(velocities.mean()),
        max_velocity_std=float(velocities.std(ddof=1)),
        flight_time_mean=float(flight_times.mean()),
        flight_time_std=float(flight_times.std(ddof=1)),
        landing_points=landing_points,
        apogees=apogees,
        results=results,
        n_successful=len(successful),
        n_failed=len(results) - len(successful),
        n_landed=len(landed),
    )
