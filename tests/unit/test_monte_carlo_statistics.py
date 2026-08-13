"""Statistical checks for flight-dispersion sampling and reporting."""

import numpy as np
import pytest
from scipy.stats import chi2

from ensim.core.monte_carlo import (
    DispersionConfig,
    _generate_perturbations,
    compute_confidence_ellipse,
)


def test_positive_multiplicative_dispersions_preserve_physical_support():
    config = DispersionConfig(
        num_simulations=2,
        thrust_sigma=2.0,
        isp_sigma=2.0,
        burn_time_sigma=2.0,
        cd_sigma=2.0,
    )
    rng = np.random.default_rng(1234)
    samples = [_generate_perturbations(config, rng) for _ in range(5000)]
    for name in ("thrust_factor", "isp_factor", "burn_time_factor", "cd_factor"):
        values = np.array([sample[name] for sample in samples])
        assert np.all(values > 0.0)


def test_zero_uncertainty_returns_nominal_factors_and_wrapped_direction():
    config = DispersionConfig(
        num_simulations=2,
        thrust_sigma=0.0,
        isp_sigma=0.0,
        burn_time_sigma=0.0,
        cd_sigma=0.0,
        wind_speed_mean=4.0,
        wind_speed_sigma=0.0,
        randomize_wind_dir=False,
        wind_direction_mean=370.0,
        wind_direction_sigma=0.0,
    )
    sample = _generate_perturbations(config, np.random.default_rng(1))
    assert sample["thrust_factor"] == 1.0
    assert sample["isp_factor"] == 1.0
    assert sample["burn_time_factor"] == 1.0
    assert sample["cd_factor"] == 1.0
    assert sample["wind_speed"] == 4.0
    assert sample["wind_direction"] == 10.0


def test_confidence_ellipse_uses_bivariate_chi_square_scaling():
    scale_x, scale_y = 3.0, 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, 20_000, endpoint=False)
    points = np.column_stack((scale_x * np.cos(angles), scale_y * np.sin(angles)))
    confidence = 0.95
    major, minor, _ = compute_confidence_ellipse(points, confidence)
    factor = np.sqrt(chi2.ppf(confidence, df=2) / 2.0)
    assert major == pytest.approx(scale_x * factor, rel=2e-4)
    assert minor == pytest.approx(scale_y * factor, rel=2e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_simulations": 1},
        {"thrust_sigma": -0.1},
        {"wind_speed_mean": -1.0},
        {"ellipse_confidence": 1.0},
        {"n_workers": 0},
    ],
)
def test_invalid_dispersion_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DispersionConfig(**kwargs)
