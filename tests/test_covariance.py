import numpy as np
import pytest

from emg_diffusion.analysis.covariance import (
    adapt_ar1_coefficient,
    adapt_channel_covariance,
    apply_robust_channel_scaler,
    ar1_covariance,
    covariance_diagnostics,
    estimate_ar1_coefficient,
    estimate_channel_covariance,
    fit_robust_channel_scaler,
    invert_robust_channel_scaler,
    lag_separability_metrics,
    lagged_cross_covariances,
    matrix_normal_nll_ar1,
    regularize_covariance,
    sample_matrix_normal_ar1,
    trace_normalize,
    validate_dense_matrix_normal_sampler,
    validate_production_projection_sampler,
)


def test_robust_channel_scaler_round_trip():
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(5, 3, 20))
    windows[:, 1] = 4.0 + 2.5 * windows[:, 1]

    scaler = fit_robust_channel_scaler(windows)
    transformed = apply_robust_channel_scaler(windows, scaler)
    reconstructed = invert_robust_channel_scaler(transformed, scaler)

    np.testing.assert_allclose(reconstructed, windows, rtol=1e-12, atol=1e-12)


def test_trace_normalization_and_eigenvalue_floor():
    covariance = np.array([[2.0, 1.95], [1.95, 2.0]])
    regularized = regularize_covariance(
        covariance, shrinkage=0.05, minimum_eigenvalue=0.01
    )
    diagnostics = covariance_diagnostics(regularized)

    assert np.trace(regularized) / 2 == pytest.approx(1.0)
    assert diagnostics["minimum_eigenvalue"] >= 0.01
    assert diagnostics["condition_number"] > 1.0


def test_adaptation_endpoints():
    population = trace_normalize(np.array([[2.0, 0.2], [0.2, 1.0]]))
    calibration = trace_normalize(np.array([[1.0, -0.1], [-0.1, 3.0]]))

    np.testing.assert_allclose(
        adapt_channel_covariance(population, calibration, 0.0), population
    )
    np.testing.assert_allclose(
        adapt_channel_covariance(population, calibration, 1.0), calibration
    )
    assert adapt_ar1_coefficient(0.2, 0.8, 0.25) == pytest.approx(0.35)


def test_ar1_estimator_recovers_simulated_coefficient():
    covariance = np.array([[1.0, 0.35], [0.35, 1.0]])
    rng = np.random.default_rng(11)
    windows = sample_matrix_normal_ar1(
        covariance, 0.72, 300, 800, rng=rng
    )

    estimated_channel = estimate_channel_covariance(windows, shrinkage=0.0)
    estimated_rho = estimate_ar1_coefficient(windows)

    assert estimated_rho == pytest.approx(0.72, abs=0.015)
    assert estimated_channel[0, 1] == pytest.approx(0.35, abs=0.03)


def test_dense_sampler_matches_channel_major_kronecker_covariance():
    channel = np.array([[1.0, 0.3], [0.3, 1.0]])
    temporal = ar1_covariance(5, 0.55)
    result = validate_dense_matrix_normal_sampler(
        channel,
        temporal,
        sample_count=30_000,
        batch_size=2_000,
        seed=19,
    )

    assert result.relative_frobenius_error < 0.035
    assert result.empirical_mean_norm < 0.06


def test_production_projection_validation_matches_analytic_variance():
    channel = np.array(
        [[1.0, 0.25, 0.1], [0.25, 1.0, 0.2], [0.1, 0.2, 1.0]]
    )
    result = validate_production_projection_sampler(
        channel,
        0.65,
        time_count=128,
        projection_count=6,
        sample_count=12_000,
        batch_size=400,
        seed=23,
    )

    assert result.root_mean_squared_relative_error < 0.04
    assert result.maximum_absolute_relative_error < 0.08


def test_true_covariance_has_lower_nll_than_identity():
    channel = np.array([[1.0, 0.65], [0.65, 1.0]])
    rng = np.random.default_rng(29)
    windows = sample_matrix_normal_ar1(channel, 0.8, 80, 1_500, rng=rng)

    true_nll = np.mean(matrix_normal_nll_ar1(windows, channel, 0.8))
    identity_nll = np.mean(
        matrix_normal_nll_ar1(windows, np.eye(2), 0.0)
    )

    assert true_nll < identity_nll


def test_lag_separability_is_near_exact_for_matrix_normal_samples():
    channel = np.array([[1.0, 0.4], [0.4, 1.0]])
    rng = np.random.default_rng(31)
    windows = sample_matrix_normal_ar1(channel, 0.7, 180, 2_000, rng=rng)
    lagged = lagged_cross_covariances(windows, [0, 1, 2, 5, 10])
    metrics = lag_separability_metrics(lagged)

    assert metrics["aggregate_relative_residual"] < 0.08
    assert metrics["lag_coefficients"][1] == pytest.approx(0.7, abs=0.02)


def test_invalid_covariance_and_ar1_values_fail_loudly():
    with pytest.raises(ValueError):
        regularize_covariance(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ar1_covariance(20, 1.0)
    with pytest.raises(ValueError):
        fit_robust_channel_scaler(np.ones((3, 2, 10)))
