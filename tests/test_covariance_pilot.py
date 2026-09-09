import numpy as np

from emg_diffusion.analysis.covariance import sample_matrix_normal_ar1
from emg_diffusion.analysis.pilot import (
    run_fold_covariance_feasibility,
    summarize_fold_results,
)


def test_fold_feasibility_runs_without_using_unassigned_subjects():
    rng = np.random.default_rng(101)
    windows = []
    records = []
    for subject in (1, 2, 3, 4):
        channel = np.array(
            [
                [1.0, 0.15 + 0.04 * subject],
                [0.15 + 0.04 * subject, 1.0],
            ]
        )
        for repetition in (1, 2, 3):
            samples = sample_matrix_normal_ar1(
                channel,
                0.45 + 0.02 * subject,
                time_count=60,
                sample_count=2,
                rng=rng,
            )
            for movement, sample in enumerate(samples, start=1):
                windows.append(sample)
                records.append(
                    {
                        "subject": subject,
                        "repetition": repetition,
                        "movement": movement,
                    }
                )

    fold = {
        "fold": 1,
        "source_train_subjects": [1, 2],
        "source_validation_subjects": [3],
        "pseudo_target_subjects": [4],
    }
    allocations = [
        {
            "allocation_id": "b0_none",
            "budget_repetitions": 0,
            "selected_repetitions": [],
        },
        {
            "allocation_id": "b1_r1",
            "budget_repetitions": 1,
            "selected_repetitions": [1],
        },
        {
            "allocation_id": "b1_r3",
            "budget_repetitions": 1,
            "selected_repetitions": [3],
        },
        {
            "allocation_id": "b2_r1_r3",
            "budget_repetitions": 2,
            "selected_repetitions": [1, 3],
        },
    ]

    result = run_fold_covariance_feasibility(
        np.stack(windows),
        records,
        fold,
        allocations,
        test_repetitions=(2,),
        eta_grid=(0.0, 0.5, 1.0),
        lags=(0, 1, 2, 5),
    )
    summary = summarize_fold_results([result])

    assert result["source_train_window_count"] == 12
    assert result["validation_test_window_count"] == 2
    assert set(result["selected_eta_by_budget"]) == {"0", "1", "2"}
    assert len(result["pseudo_target_evaluation"]) == 4
    assert [row["budget_repetitions"] for row in summary] == [0, 1, 2]
