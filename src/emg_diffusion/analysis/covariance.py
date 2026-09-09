from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import matmul_toeplitz
from scipy.signal import lfilter


@dataclass(frozen=True)
class RobustChannelScaler:
    center: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class DenseMonteCarloResult:
    sample_count: int
    target_covariance: np.ndarray
    empirical_covariance: np.ndarray
    relative_frobenius_error: float
    maximum_absolute_error: float
    empirical_mean_norm: float


@dataclass(frozen=True)
class ProjectionMonteCarloResult:
    sample_count: int
    projections: np.ndarray
    target_variances: np.ndarray
    empirical_variances: np.ndarray
    relative_errors: np.ndarray
    root_mean_squared_relative_error: float
    maximum_absolute_relative_error: float


def _as_windows(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(
            "windows must have shape [window, channel, time] or [channel, time]"
        )
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] < 2:
        raise ValueError("windows must contain at least one window and two time samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("windows contain non-finite values")
    return values


def fit_robust_channel_scaler(
    windows: np.ndarray,
    *,
    minimum_scale: float = 1e-12,
) -> RobustChannelScaler:
    values = _as_windows(windows)
    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive")
    by_channel = np.moveaxis(values, 1, 0).reshape(values.shape[1], -1)
    center = np.median(by_channel, axis=1)
    lower, upper = np.percentile(by_channel, [25.0, 75.0], axis=1)
    scale = upper - lower
    if np.any(scale < minimum_scale):
        bad = np.flatnonzero(scale < minimum_scale) + 1
        raise ValueError(f"near-constant channel scale for channel(s) {bad.tolist()}")
    return RobustChannelScaler(center=center, scale=scale)


def apply_robust_channel_scaler(
    windows: np.ndarray,
    scaler: RobustChannelScaler,
) -> np.ndarray:
    values = _as_windows(windows)
    channel_count = values.shape[1]
    if scaler.center.shape != (channel_count,) or scaler.scale.shape != (
        channel_count,
    ):
        raise ValueError("scaler dimensions do not match the window channels")
    if np.any(scaler.scale <= 0) or not np.all(np.isfinite(scaler.scale)):
        raise ValueError("scaler contains invalid channel scales")
    return (values - scaler.center[None, :, None]) / scaler.scale[None, :, None]


def invert_robust_channel_scaler(
    windows: np.ndarray,
    scaler: RobustChannelScaler,
) -> np.ndarray:
    values = _as_windows(windows)
    return values * scaler.scale[None, :, None] + scaler.center[None, :, None]


def demean_windows(windows: np.ndarray) -> np.ndarray:
    values = _as_windows(windows)
    return values - np.mean(values, axis=-1, keepdims=True)


def trace_normalize(covariance: np.ndarray) -> np.ndarray:
    covariance = _square_symmetric(covariance)
    mean_eigenvalue = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(mean_eigenvalue) or mean_eigenvalue <= 0:
        raise ValueError("covariance must have a positive finite trace")
    return covariance / mean_eigenvalue


def regularize_covariance(
    covariance: np.ndarray,
    *,
    shrinkage: float = 0.01,
    minimum_eigenvalue: float = 1e-6,
) -> np.ndarray:
    covariance = _square_symmetric(covariance)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if minimum_eigenvalue <= 0:
        raise ValueError("minimum_eigenvalue must be positive")

    scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("covariance must have a positive finite trace")
    regularized = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(
        covariance.shape[0]
    )
    regularized = trace_normalize(regularized)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalues = np.maximum(eigenvalues, minimum_eigenvalue)
    floored = (eigenvectors * eigenvalues) @ eigenvectors.T
    return trace_normalize(floored)


def estimate_channel_covariance(
    windows: np.ndarray,
    *,
    shrinkage: float = 0.01,
    minimum_eigenvalue: float = 1e-6,
    remove_window_mean: bool = True,
) -> np.ndarray:
    values = _as_windows(windows)
    if remove_window_mean:
        values = demean_windows(values)
    sample_count = values.shape[0] * values.shape[2]
    covariance = np.einsum("nct,ndt->cd", values, values, optimize=True)
    covariance /= sample_count
    return regularize_covariance(
        covariance,
        shrinkage=shrinkage,
        minimum_eigenvalue=minimum_eigenvalue,
    )


def estimate_ar1_coefficient(
    windows: np.ndarray,
    *,
    maximum_absolute_value: float = 0.995,
    remove_window_mean: bool = True,
) -> float:
    values = _as_windows(windows)
    if not 0 < maximum_absolute_value < 1:
        raise ValueError("maximum_absolute_value must lie in (0, 1)")
    if remove_window_mean:
        values = demean_windows(values)
    previous = values[..., :-1]
    following = values[..., 1:]
    denominator = float(np.sum(np.square(previous)))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("cannot estimate temporal dependence from zero-energy data")
    coefficient = float(np.sum(previous * following) / denominator)
    return float(
        np.clip(coefficient, -maximum_absolute_value, maximum_absolute_value)
    )


def ar1_covariance(length: int, coefficient: float) -> np.ndarray:
    _validate_ar1(length, coefficient)
    indices = np.arange(length)
    return coefficient ** np.abs(indices[:, None] - indices[None, :])


def adapt_channel_covariance(
    population: np.ndarray,
    calibration: np.ndarray,
    weight: float,
    *,
    minimum_eigenvalue: float = 1e-6,
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("adaptation weight must lie in [0, 1]")
    population = trace_normalize(population)
    calibration = trace_normalize(calibration)
    if population.shape != calibration.shape:
        raise ValueError("population and calibration covariance dimensions differ")
    mixed = (1.0 - weight) * population + weight * calibration
    return regularize_covariance(
        mixed,
        shrinkage=0.0,
        minimum_eigenvalue=minimum_eigenvalue,
    )


def adapt_ar1_coefficient(
    population: float,
    calibration: float,
    weight: float,
    *,
    maximum_absolute_value: float = 0.995,
) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("adaptation weight must lie in [0, 1]")
    adapted = (1.0 - weight) * population + weight * calibration
    return float(np.clip(adapted, -maximum_absolute_value, maximum_absolute_value))


def lagged_cross_covariances(
    windows: np.ndarray,
    lags: Sequence[int],
    *,
    remove_window_mean: bool = True,
) -> dict[int, np.ndarray]:
    values = _as_windows(windows)
    if remove_window_mean:
        values = demean_windows(values)
    result: dict[int, np.ndarray] = {}
    for lag in sorted(set(int(value) for value in lags)):
        if lag < 0 or lag >= values.shape[2]:
            raise ValueError(f"invalid lag {lag} for window length {values.shape[2]}")
        left = values[..., : values.shape[2] - lag] if lag else values
        right = values[..., lag:] if lag else values
        denominator = left.shape[0] * left.shape[2]
        result[lag] = np.einsum(
            "nct,ndt->cd", left, right, optimize=True
        ) / denominator
    return result


def lag_separability_metrics(
    cross_covariances: dict[int, np.ndarray],
) -> dict[str, object]:
    if 0 not in cross_covariances:
        raise ValueError("lag-zero covariance is required")
    base = _square_symmetric(cross_covariances[0])
    base_norm_squared = float(np.sum(np.square(base)))
    if base_norm_squared <= 0:
        raise ValueError("lag-zero covariance has zero Frobenius norm")

    coefficients: dict[int, float] = {}
    relative_residuals: dict[int, float] = {}
    numerator = 0.0
    denominator = 0.0
    for lag, matrix in sorted(cross_covariances.items()):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != base.shape:
            raise ValueError("lagged covariance dimensions differ")
        coefficient = float(np.sum(matrix * base) / base_norm_squared)
        residual = matrix - coefficient * base
        matrix_norm_squared = float(np.sum(np.square(matrix)))
        coefficients[lag] = coefficient
        relative_residuals[lag] = float(
            np.sqrt(np.sum(np.square(residual)) / matrix_norm_squared)
        ) if matrix_norm_squared > 0 else 0.0
        if lag > 0:
            numerator += float(np.sum(np.square(residual)))
            denominator += matrix_norm_squared

    aggregate = float(np.sqrt(numerator / denominator)) if denominator > 0 else 0.0
    return {
        "lag_coefficients": coefficients,
        "relative_residuals": relative_residuals,
        "aggregate_relative_residual": aggregate,
    }


def matrix_normal_nll_ar1(
    windows: np.ndarray,
    channel_covariance: np.ndarray,
    temporal_coefficient: float,
    *,
    remove_window_mean: bool = True,
    include_constant: bool = False,
    per_element: bool = True,
) -> np.ndarray:
    values = _as_windows(windows)
    if remove_window_mean:
        values = demean_windows(values)
    channel_covariance = _square_symmetric(channel_covariance)
    channel_count, time_count = values.shape[1:]
    if channel_covariance.shape != (channel_count, channel_count):
        raise ValueError("channel covariance dimensions do not match windows")
    _validate_ar1(time_count, temporal_coefficient)

    sign, channel_logdet = np.linalg.slogdet(channel_covariance)
    if sign <= 0:
        raise ValueError("channel covariance must be positive definite")
    temporal_logdet = (time_count - 1) * np.log1p(
        -(temporal_coefficient**2)
    )

    arranged = np.moveaxis(values, 1, 0).reshape(channel_count, -1)
    solved = np.linalg.solve(channel_covariance, arranged)
    solved = np.moveaxis(
        solved.reshape(channel_count, values.shape[0], time_count), 0, 1
    )
    pointwise = np.sum(values * solved, axis=1)
    endpoints = pointwise[:, 0] + pointwise[:, -1]
    interior = (
        (1.0 + temporal_coefficient**2)
        * np.sum(pointwise[:, 1:-1], axis=1)
    )
    adjacent = np.sum(
        values[..., :-1] * solved[..., 1:], axis=(1, 2)
    )
    quadratic = (
        endpoints + interior - 2.0 * temporal_coefficient * adjacent
    ) / (1.0 - temporal_coefficient**2)

    result = 0.5 * (
        time_count * channel_logdet
        + channel_count * temporal_logdet
        + quadratic
    )
    if include_constant:
        result += 0.5 * channel_count * time_count * np.log(2.0 * np.pi)
    if per_element:
        result /= channel_count * time_count
    return result


def sample_matrix_normal_dense(
    channel_covariance: np.ndarray,
    temporal_covariance: np.ndarray,
    sample_count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    channel_covariance = _square_symmetric(channel_covariance)
    temporal_covariance = _square_symmetric(temporal_covariance)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    channel_factor = np.linalg.cholesky(channel_covariance)
    temporal_factor = np.linalg.cholesky(temporal_covariance)
    innovations = rng.standard_normal(
        (sample_count, channel_covariance.shape[0], temporal_covariance.shape[0])
    )
    channel_colored = np.einsum(
        "cd,bdt->bct", channel_factor, innovations, optimize=True
    )
    return np.einsum(
        "bct,st->bcs", channel_colored, temporal_factor, optimize=True
    )


def sample_matrix_normal_ar1(
    channel_covariance: np.ndarray,
    temporal_coefficient: float,
    time_count: int,
    sample_count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    channel_covariance = _square_symmetric(channel_covariance)
    _validate_ar1(time_count, temporal_coefficient)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    channel_count = channel_covariance.shape[0]
    innovations = rng.standard_normal((sample_count, channel_count, time_count))
    temporal = np.empty_like(innovations)
    temporal[..., 0] = innovations[..., 0]
    if time_count > 1:
        scale = np.sqrt(1.0 - temporal_coefficient**2)
        initial = temporal_coefficient * temporal[..., 0, None]
        temporal[..., 1:], _ = lfilter(
            [scale],
            [1.0, -temporal_coefficient],
            innovations[..., 1:],
            axis=-1,
            zi=initial,
        )
    channel_factor = np.linalg.cholesky(channel_covariance)
    return np.einsum(
        "cd,bdt->bct", channel_factor, temporal, optimize=True
    )


def validate_dense_matrix_normal_sampler(
    channel_covariance: np.ndarray,
    temporal_covariance: np.ndarray,
    *,
    sample_count: int = 100_000,
    batch_size: int = 2_000,
    seed: int = 17,
) -> DenseMonteCarloResult:
    channel_covariance = _square_symmetric(channel_covariance)
    temporal_covariance = _square_symmetric(temporal_covariance)
    if sample_count <= 1 or batch_size <= 0:
        raise ValueError("sample_count must exceed one and batch_size must be positive")
    dimension = channel_covariance.shape[0] * temporal_covariance.shape[0]
    total = np.zeros(dimension, dtype=np.float64)
    second_moment = np.zeros((dimension, dimension), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))

    completed = 0
    while completed < sample_count:
        current = min(batch_size, sample_count - completed)
        samples = sample_matrix_normal_dense(
            channel_covariance,
            temporal_covariance,
            current,
            rng=rng,
        )
        flattened = samples.reshape(current, dimension)
        total += np.sum(flattened, axis=0)
        second_moment += flattened.T @ flattened
        completed += current

    mean = total / sample_count
    empirical = (
        second_moment - sample_count * np.outer(mean, mean)
    ) / (sample_count - 1)
    target = np.kron(channel_covariance, temporal_covariance)
    difference = empirical - target
    return DenseMonteCarloResult(
        sample_count=sample_count,
        target_covariance=target,
        empirical_covariance=empirical,
        relative_frobenius_error=float(
            np.linalg.norm(difference, ord="fro")
            / np.linalg.norm(target, ord="fro")
        ),
        maximum_absolute_error=float(np.max(np.abs(difference))),
        empirical_mean_norm=float(np.linalg.norm(mean)),
    )


def projection_variances_ar1(
    projections: np.ndarray,
    channel_covariance: np.ndarray,
    temporal_coefficient: float,
) -> np.ndarray:
    projections = np.asarray(projections, dtype=np.float64)
    if projections.ndim != 3:
        raise ValueError("projections must have shape [projection, channel, time]")
    channel_covariance = _square_symmetric(channel_covariance)
    if projections.shape[1] != channel_covariance.shape[0]:
        raise ValueError("projection and channel covariance dimensions differ")
    _validate_ar1(projections.shape[2], temporal_coefficient)
    first_column = temporal_coefficient ** np.arange(projections.shape[2])

    variances = []
    for projection in projections:
        channel_applied = channel_covariance @ projection
        fully_applied = matmul_toeplitz(
            (first_column, first_column),
            channel_applied.T,
            check_finite=False,
        ).T
        variances.append(float(np.sum(projection * fully_applied)))
    return np.asarray(variances)


def validate_production_projection_sampler(
    channel_covariance: np.ndarray,
    temporal_coefficient: float,
    *,
    time_count: int = 2000,
    projection_count: int = 12,
    sample_count: int = 5_000,
    batch_size: int = 100,
    seed: int = 42,
) -> ProjectionMonteCarloResult:
    channel_covariance = _square_symmetric(channel_covariance)
    if projection_count <= 0 or sample_count <= 1 or batch_size <= 0:
        raise ValueError("projection_count and batch_size must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    projections = rng.standard_normal(
        (projection_count, channel_covariance.shape[0], time_count)
    )
    norms = np.linalg.norm(projections.reshape(projection_count, -1), axis=1)
    projections /= norms[:, None, None]
    target = projection_variances_ar1(
        projections, channel_covariance, temporal_coefficient
    )

    projection_total = np.zeros(projection_count, dtype=np.float64)
    projection_squares = np.zeros(projection_count, dtype=np.float64)
    completed = 0
    while completed < sample_count:
        current = min(batch_size, sample_count - completed)
        samples = sample_matrix_normal_ar1(
            channel_covariance,
            temporal_coefficient,
            time_count,
            current,
            rng=rng,
        )
        values = np.einsum("bct,kct->bk", samples, projections, optimize=True)
        projection_total += np.sum(values, axis=0)
        projection_squares += np.sum(np.square(values), axis=0)
        completed += current

    means = projection_total / sample_count
    empirical = (
        projection_squares - sample_count * np.square(means)
    ) / (sample_count - 1)
    relative = (empirical - target) / target
    return ProjectionMonteCarloResult(
        sample_count=sample_count,
        projections=projections,
        target_variances=target,
        empirical_variances=empirical,
        relative_errors=relative,
        root_mean_squared_relative_error=float(np.sqrt(np.mean(relative**2))),
        maximum_absolute_relative_error=float(np.max(np.abs(relative))),
    )


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, float]:
    covariance = _square_symmetric(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "mean_eigenvalue": float(np.mean(eigenvalues)),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }


def _square_symmetric(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("covariance contains non-finite values")
    return 0.5 * (values + values.T)


def _validate_ar1(length: int, coefficient: float) -> None:
    if length < 2:
        raise ValueError("AR(1) covariance requires at least two time samples")
    if not np.isfinite(coefficient) or abs(coefficient) >= 1:
        raise ValueError("stationary AR(1) coefficient must lie in (-1, 1)")

