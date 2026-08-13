"""Reduced-order uncertainty propagation for ideal rocket-engine performance."""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm

from ensim.core.propulsion import NozzleConditions, calculate_performance


@dataclass(frozen=True)
class EngineUQInput:
    """Nominal ideal-nozzle inputs and fractional one-sigma uncertainties."""

    chamber_pressure: float = 10e6
    throat_area: float = 0.01
    gamma: float = 1.2
    chamber_temperature: float = 3500.0
    mean_molecular_weight: float = 18.0
    expansion_ratio: float = 50.0
    chamber_pressure_sigma: float = 0.02
    throat_area_sigma: float = 0.01
    gamma_sigma: float = 0.01


@dataclass
class EngineUQResult:
    """Samples and descriptive statistics from engine uncertainty propagation."""

    thrust_distribution: np.ndarray
    isp_distribution: np.ndarray
    cstar_distribution: np.ndarray
    n_samples: int
    n_requested: int
    n_failed: int
    runtime_seconds: float
    thrust_mean: float
    thrust_std: float
    thrust_p95: float
    thrust_p99: float
    isp_mean: float
    isp_std: float
    isp_p95: float
    isp_p99: float
    reliability: float | None
    threshold: float | None
    model: str = "ideal_frozen_calorically_perfect_nozzle"

    def get_confidence_interval(
        self,
        confidence: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Return central empirical intervals for thrust, Isp, and c*."""
        if not 0.0 < confidence < 1.0:
            raise ValueError("Confidence must lie between zero and one")
        alpha = 50.0 * (1.0 - confidence)
        return {
            "thrust": tuple(np.percentile(self.thrust_distribution, [alpha, 100.0 - alpha])),
            "isp": tuple(np.percentile(self.isp_distribution, [alpha, 100.0 - alpha])),
            "cstar": tuple(np.percentile(self.cstar_distribution, [alpha, 100.0 - alpha])),
        }


def evaluate_ideal_engine_performance(
    chamber_pressure: float,
    throat_area: float,
    gamma: float,
    chamber_temperature: float,
    mean_molecular_weight: float,
    expansion_ratio: float,
    ambient_pressure: float = 0.0,
) -> dict[str, float]:
    """Evaluate the shared one-dimensional frozen-nozzle model."""
    if (
        chamber_pressure <= 0.0
        or throat_area <= 0.0
        or not 1.0 < gamma < 2.0
        or chamber_temperature <= 0.0
        or mean_molecular_weight <= 0.0
        or expansion_ratio <= 1.0
        or ambient_pressure < 0.0
    ):
        raise ValueError("Ideal-engine inputs are outside their physical domain")
    result = calculate_performance(
        T_chamber=chamber_temperature,
        P_chamber=chamber_pressure,
        gamma=gamma,
        mean_molecular_weight=mean_molecular_weight,
        nozzle=NozzleConditions(
            area_ratio=expansion_ratio,
            chamber_pressure=chamber_pressure,
            ambient_pressure=ambient_pressure,
            throat_area=throat_area,
        ),
    )
    return {
        "thrust": result.thrust,
        "isp": result.isp,
        "c_star": result.c_star,
    }


def _evaluate_sample(args: tuple[int, float, float, float, float, float, float, float]):
    index, pressure, area, gamma, temperature, molecular_weight, expansion, ambient = args
    try:
        return index, evaluate_ideal_engine_performance(
            pressure,
            area,
            gamma,
            temperature,
            molecular_weight,
            expansion,
            ambient,
        )
    except (ValueError, FloatingPointError, OverflowError):
        return index, {"thrust": np.nan, "isp": np.nan, "c_star": np.nan}


class EngineUQAnalyzer:
    """Propagate aleatory input uncertainty through the ideal frozen-nozzle model."""

    def __init__(self, n_workers: int | None = None):
        if n_workers is not None and n_workers < 1:
            raise ValueError("Worker count must be positive")
        self.n_workers = n_workers

    def run(
        self,
        inputs: EngineUQInput,
        n_samples: int = 1000,
        ambient_pressure: float = 0.0,
        thrust_threshold: float | None = None,
        seed: int | None = None,
    ) -> EngineUQResult:
        """Draw independent samples and propagate them through the reduced-order model."""
        if n_samples < 2 or ambient_pressure < 0.0:
            raise ValueError("At least two samples and non-negative ambient pressure are required")
        nominal = (
            inputs.chamber_pressure,
            inputs.throat_area,
            inputs.chamber_temperature,
            inputs.mean_molecular_weight,
            inputs.expansion_ratio,
        )
        sigmas = (
            inputs.chamber_pressure_sigma,
            inputs.throat_area_sigma,
            inputs.gamma_sigma,
        )
        if any(value <= 0.0 for value in nominal) or any(value < 0.0 for value in sigmas):
            raise ValueError("Engine UQ nominal values and uncertainties are invalid")
        if not 1.0 < inputs.gamma < 2.0:
            raise ValueError("Nominal gamma must lie between one and two")
        if thrust_threshold is not None and thrust_threshold < 0.0:
            raise ValueError("Thrust threshold cannot be negative")

        rng = np.random.default_rng(seed)

        def positive_samples(value: float, fractional_sigma: float) -> np.ndarray:
            if fractional_sigma == 0.0:
                return np.full(n_samples, value)
            log_variance = np.log1p(fractional_sigma**2)
            return value * rng.lognormal(
                -0.5 * log_variance,
                np.sqrt(log_variance),
                n_samples,
            )

        pressure = positive_samples(
            inputs.chamber_pressure,
            inputs.chamber_pressure_sigma,
        )
        area = positive_samples(inputs.throat_area, inputs.throat_area_sigma)
        if inputs.gamma_sigma == 0.0:
            gamma = np.full(n_samples, inputs.gamma)
        else:
            scale = inputs.gamma * inputs.gamma_sigma
            gamma = truncnorm.rvs(
                (1.0 - inputs.gamma) / scale,
                (2.0 - inputs.gamma) / scale,
                loc=inputs.gamma,
                scale=scale,
                size=n_samples,
                random_state=rng,
            )

        args = [
            (
                index,
                pressure[index],
                area[index],
                gamma[index],
                inputs.chamber_temperature,
                inputs.mean_molecular_weight,
                inputs.expansion_ratio,
                ambient_pressure,
            )
            for index in range(n_samples)
        ]
        started = time.perf_counter()
        if self.n_workers == 1:
            indexed_results = [_evaluate_sample(sample) for sample in args]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                indexed_results = list(executor.map(_evaluate_sample, args))
        runtime = time.perf_counter() - started

        indexed_results.sort(key=lambda item: item[0])
        samples = [sample for _, sample in indexed_results]
        thrust = np.array([sample["thrust"] for sample in samples])
        isp = np.array([sample["isp"] for sample in samples])
        cstar = np.array([sample["c_star"] for sample in samples])
        valid = np.isfinite(thrust) & np.isfinite(isp) & np.isfinite(cstar)
        thrust, isp, cstar = thrust[valid], isp[valid], cstar[valid]
        if thrust.size < 2:
            raise RuntimeError("Fewer than two valid engine UQ samples were produced")

        reliability = (
            float(np.mean(thrust > thrust_threshold)) if thrust_threshold is not None else None
        )
        return EngineUQResult(
            thrust_distribution=thrust,
            isp_distribution=isp,
            cstar_distribution=cstar,
            n_samples=thrust.size,
            n_requested=n_samples,
            n_failed=n_samples - thrust.size,
            runtime_seconds=runtime,
            thrust_mean=float(np.mean(thrust)),
            thrust_std=float(np.std(thrust, ddof=1)),
            thrust_p95=float(np.percentile(thrust, 95)),
            thrust_p99=float(np.percentile(thrust, 99)),
            isp_mean=float(np.mean(isp)),
            isp_std=float(np.std(isp, ddof=1)),
            isp_p95=float(np.percentile(isp, 95)),
            isp_p99=float(np.percentile(isp, 99)),
            reliability=reliability,
            threshold=thrust_threshold,
        )

    def run_sequential(
        self,
        inputs: EngineUQInput,
        n_samples: int = 1000,
        ambient_pressure: float = 0.0,
        seed: int | None = None,
    ) -> EngineUQResult:
        """Run without child processes for GUI and constrained environments."""
        previous_workers = self.n_workers
        self.n_workers = 1
        try:
            return self.run(inputs, n_samples, ambient_pressure, seed=seed)
        finally:
            self.n_workers = previous_workers
