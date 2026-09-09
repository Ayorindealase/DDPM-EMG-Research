from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from emg_diffusion.analysis.covariance import (
    adapt_ar1_coefficient,
    adapt_channel_covariance,
    apply_robust_channel_scaler,
    covariance_diagnostics,
    estimate_ar1_coefficient,
    estimate_channel_covariance,
    fit_robust_channel_scaler,
    lag_separability_metrics,
    lagged_cross_covariances,
    matrix_normal_nll_ar1,
)
from emg_diffusion.data.windows import record_mask


def run_fold_covariance_feasibility(
    windows: np.ndarray,
    records: Sequence[Mapping[str, object]],
    fold: Mapping[str, object],
    calibration_allocations: Sequence[Mapping[str, object]],
    *,
    test_repetitions: Sequence[int] = (2, 5),
    eta_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
    lags: Sequence[int] = (0, 1, 2, 5, 10, 20, 50, 100, 200),
    covariance_shrinkage: float = 0.01,
    minimum_eigenvalue: float = 1e-6,
) -> dict[str, object]:
    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[0] != len(records):
        raise ValueError("windows and records are inconsistent")
    source_train = [int(value) for value in fold["source_train_subjects"]]
    source_validation = [
        int(value) for value in fold["source_validation_subjects"]
    ]
    pseudo_target = [int(value) for value in fold["pseudo_target_subjects"]]

    train_mask = record_mask(records, subjects=source_train)
    scaler = fit_robust_channel_scaler(windows[train_mask])
    scaled = apply_robust_channel_scaler(windows, scaler)
    population_covariance = estimate_channel_covariance(
        scaled[train_mask],
        shrinkage=covariance_shrinkage,
        minimum_eigenvalue=minimum_eigenvalue,
    )
    population_rho = estimate_ar1_coefficient(scaled[train_mask])

    validation_test_mask = record_mask(
        records,
        subjects=source_validation,
        repetitions=test_repetitions,
    )
    lagged = lagged_cross_covariances(scaled[validation_test_mask], lags)
    separability = lag_separability_metrics(lagged)

    budgets = sorted(
        {
            int(allocation["budget_repetitions"])
            for allocation in calibration_allocations
            if int(allocation["budget_repetitions"]) > 0
        }
    )
    tuning_rows: list[dict[str, object]] = []
    selected_eta: dict[int, float] = {0: 0.0}
    for budget in budgets:
        budget_allocations = [
            allocation
            for allocation in calibration_allocations
            if int(allocation["budget_repetitions"]) == budget
        ]
        scores_by_eta: defaultdict[float, list[float]] = defaultdict(list)
        for subject in source_validation:
            test_mask = record_mask(
                records,
                subjects=[subject],
                repetitions=test_repetitions,
            )
            test_windows = scaled[test_mask]
            for allocation in budget_allocations:
                calibration_mask = record_mask(
                    records,
                    subjects=[subject],
                    repetitions=allocation["selected_repetitions"],
                )
                calibration_windows = scaled[calibration_mask]
                calibration_covariance = estimate_channel_covariance(
                    calibration_windows,
                    shrinkage=covariance_shrinkage,
                    minimum_eigenvalue=minimum_eigenvalue,
                )
                calibration_rho = estimate_ar1_coefficient(calibration_windows)
                for eta in eta_grid:
                    adapted_covariance = adapt_channel_covariance(
                        population_covariance,
                        calibration_covariance,
                        float(eta),
                        minimum_eigenvalue=minimum_eigenvalue,
                    )
                    adapted_rho = adapt_ar1_coefficient(
                        population_rho, calibration_rho, float(eta)
                    )
                    score = float(
                        np.mean(
                            matrix_normal_nll_ar1(
                                test_windows,
                                adapted_covariance,
                                adapted_rho,
                            )
                        )
                    )
                    scores_by_eta[float(eta)].append(score)

        for eta in eta_grid:
            values = scores_by_eta[float(eta)]
            tuning_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "budget_repetitions": budget,
                    "eta": float(eta),
                    "mean_validation_nll": float(np.mean(values)),
                    "validation_unit_count": len(values),
                }
            )
        selected_eta[budget] = min(
            (float(value) for value in eta_grid),
            key=lambda eta: (
                float(np.mean(scores_by_eta[eta])),
                eta,
            ),
        )

    target_rows: list[dict[str, object]] = []
    identity_covariance = np.eye(windows.shape[1])
    for subject in pseudo_target:
        test_mask = record_mask(
            records,
            subjects=[subject],
            repetitions=test_repetitions,
        )
        test_windows = scaled[test_mask]
        identity_nll = float(
            np.mean(matrix_normal_nll_ar1(test_windows, identity_covariance, 0.0))
        )
        population_nll = float(
            np.mean(
                matrix_normal_nll_ar1(
                    test_windows, population_covariance, population_rho
                )
            )
        )
        target_rows.append(
            {
                "fold": int(fold["fold"]),
                "subject": subject,
                "budget_repetitions": 0,
                "allocation_id": "b0_none",
                "selected_repetitions": [],
                "eta": 0.0,
                "identity_nll": identity_nll,
                "population_nll": population_nll,
                "adapted_nll": population_nll,
                "adapted_minus_population_nll": 0.0,
            }
        )
        for allocation in calibration_allocations:
            budget = int(allocation["budget_repetitions"])
            if budget == 0:
                continue
            calibration_mask = record_mask(
                records,
                subjects=[subject],
                repetitions=allocation["selected_repetitions"],
            )
            calibration_windows = scaled[calibration_mask]
            calibration_covariance = estimate_channel_covariance(
                calibration_windows,
                shrinkage=covariance_shrinkage,
                minimum_eigenvalue=minimum_eigenvalue,
            )
            calibration_rho = estimate_ar1_coefficient(calibration_windows)
            eta = selected_eta[budget]
            adapted_covariance = adapt_channel_covariance(
                population_covariance,
                calibration_covariance,
                eta,
                minimum_eigenvalue=minimum_eigenvalue,
            )
            adapted_rho = adapt_ar1_coefficient(
                population_rho, calibration_rho, eta
            )
            adapted_nll = float(
                np.mean(
                    matrix_normal_nll_ar1(
                        test_windows, adapted_covariance, adapted_rho
                    )
                )
            )
            target_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "subject": subject,
                    "budget_repetitions": budget,
                    "allocation_id": str(allocation["allocation_id"]),
                    "selected_repetitions": [
                        int(value) for value in allocation["selected_repetitions"]
                    ],
                    "eta": eta,
                    "identity_nll": identity_nll,
                    "population_nll": population_nll,
                    "adapted_nll": adapted_nll,
                    "adapted_minus_population_nll": (
                        adapted_nll - population_nll
                    ),
                }
            )

    return {
        "fold": int(fold["fold"]),
        "source_train_subjects": source_train,
        "source_validation_subjects": source_validation,
        "pseudo_target_subjects": pseudo_target,
        "source_train_window_count": int(np.sum(train_mask)),
        "validation_test_window_count": int(np.sum(validation_test_mask)),
        "scaler_center": scaler.center.tolist(),
        "scaler_scale": scaler.scale.tolist(),
        "population_channel_covariance": population_covariance.tolist(),
        "population_temporal_ar1": population_rho,
        "population_channel_diagnostics": covariance_diagnostics(
            population_covariance
        ),
        "separability": separability,
        "selected_eta_by_budget": {
            str(key): value for key, value in sorted(selected_eta.items())
        },
        "validation_tuning": tuning_rows,
        "pseudo_target_evaluation": target_rows,
    }


def summarize_fold_results(
    fold_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for result in fold_results:
        evaluations = result["pseudo_target_evaluation"]
        for budget in (0, 1, 2):
            selected = [
                row
                for row in evaluations
                if int(row["budget_repetitions"]) == budget
            ]
            if not selected:
                continue
            subject_means = []
            for subject in result["pseudo_target_subjects"]:
                subject_rows = [
                    row for row in selected if int(row["subject"]) == int(subject)
                ]
                subject_means.append(
                    float(
                        np.mean(
                            [
                                float(row["adapted_minus_population_nll"])
                                for row in subject_rows
                            ]
                        )
                    )
                )
            rows.append(
                {
                    "fold": int(result["fold"]),
                    "budget_repetitions": budget,
                    "eta": float(result["selected_eta_by_budget"][str(budget)]),
                    "mean_adapted_minus_population_nll": float(
                        np.mean(subject_means)
                    ),
                    "improved_subject_count": int(
                        np.sum(np.asarray(subject_means) < 0.0)
                    ),
                    "subject_count": len(subject_means),
                    "separability_residual": float(
                        result["separability"]["aggregate_relative_residual"]
                    ),
                    "population_temporal_ar1": float(
                        result["population_temporal_ar1"]
                    ),
                }
            )
    return rows

