"""Checks for the reduced-order engine uncertainty analysis."""

import numpy as np
import pytest

from ensim.core.engine_uq import EngineUQAnalyzer, EngineUQInput


def test_sequential_engine_monte_carlo_is_seed_reproducible():
    analyzer = EngineUQAnalyzer(n_workers=1)
    inputs = EngineUQInput(
        chamber_pressure_sigma=0.03,
        throat_area_sigma=0.02,
        gamma_sigma=0.01,
    )
    first = analyzer.run(inputs, n_samples=40, seed=82)
    second = analyzer.run(inputs, n_samples=40, seed=82)
    assert np.array_equal(first.thrust_distribution, second.thrust_distribution)
    assert np.array_equal(first.isp_distribution, second.isp_distribution)
    assert np.all(first.thrust_distribution > 0.0)


def test_result_identifies_reduced_order_model():
    result = EngineUQAnalyzer(n_workers=1).run(EngineUQInput(), n_samples=10, seed=1)
    assert result.model == "ideal_frozen_calorically_perfect_nozzle"
    assert result.reliability is None


def test_engine_monte_carlo_rejects_invalid_sample_count():
    with pytest.raises(ValueError, match="At least two"):
        EngineUQAnalyzer(n_workers=1).run(EngineUQInput(), n_samples=1)
