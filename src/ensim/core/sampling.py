"""Random sampling helpers shared by the uncertainty analyses."""

import numpy as np


def lognormal_factor(
    rng: np.random.Generator, fractional_sigma: float, size: int | None = None
) -> float | np.ndarray:
    """Positive multiplicative factor with mean 1 and the given coefficient of variation.

    A lognormal keeps physical quantities such as thrust or drag strictly
    positive even for large uncertainties, which a normal distribution does not.
    """
    if fractional_sigma < 0.0:
        raise ValueError("Fractional sigma cannot be negative")
    if fractional_sigma == 0.0:
        return 1.0 if size is None else np.ones(size)
    log_variance = np.log1p(fractional_sigma**2)
    sample = rng.lognormal(-0.5 * log_variance, np.sqrt(log_variance), size)
    return float(sample) if size is None else sample
