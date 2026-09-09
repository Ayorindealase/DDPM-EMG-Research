import copy

import pytest

from emg_diffusion.data.splits import (
    PilotSplitConfig,
    build_pilot_split,
    calibration_allocations,
    confirmatory_lock,
    guard_confirmatory_subjects,
    iter_window_manifest,
    validate_pilot_split,
    validate_window_manifest,
)


def small_config() -> PilotSplitConfig:
    return PilotSplitConfig(
        seed=17,
        exercise=1,
        subjects=tuple(range(1, 9)),
        development_count=4,
        fold_count=2,
        target_subjects_per_fold=2,
        validation_subjects_per_fold=1,
        calibration_repetitions=(1, 3),
        test_repetitions=(2,),
        expected_movements=(1,),
    )


def trial_rows(plan):
    rows = []
    for subject in plan["development_subjects"]:
        for repetition in (1, 2, 3):
            rows.append(
                {
                    "subject": str(subject),
                    "exercise": "1",
                    "file": f"S{subject}_E1_A1.mat",
                    "movement": "1",
                    "repetition": str(repetition),
                    "segment_index": "1",
                    "start_sample": str(repetition * 10_000),
                    "stop_sample": str(repetition * 10_000 + 3000),
                    "generator_length_samples": "2000",
                    "generator_step_samples": "1000",
                    "generator_window_count": "2",
                    "classifier_length_samples": "400",
                    "classifier_step_samples": "200",
                    "classifier_window_count": "14",
                }
            )
    return rows


def test_split_is_deterministic_disjoint_and_balanced():
    first = build_pilot_split(small_config())
    second = build_pilot_split(small_config())

    assert first == second
    summary = validate_pilot_split(first)
    assert summary == {
        "participant_count": 8,
        "development_count": 4,
        "confirmatory_count": 4,
        "fold_count": 2,
    }


def test_every_development_subject_is_target_once():
    plan = build_pilot_split(small_config())
    targets = [
        subject
        for fold in plan["folds"]
        for subject in fold["pseudo_target_subjects"]
    ]
    assert sorted(targets) == plan["development_subjects"]


def test_calibration_allocations_cover_zero_one_and_two_repetitions():
    allocations = calibration_allocations((1, 3, 4, 6))
    assert len(allocations) == 11
    assert [row["budget_repetitions"] for row in allocations].count(0) == 1
    assert [row["budget_repetitions"] for row in allocations].count(1) == 4
    assert [row["budget_repetitions"] for row in allocations].count(2) == 6


def test_confirmatory_guard_rejects_locked_subjects():
    plan = build_pilot_split(small_config())
    lock = confirmatory_lock(plan)

    guard_confirmatory_subjects(plan["development_subjects"], lock["confirmatory_subjects"])
    with pytest.raises(PermissionError, match="confirmatory participant"):
        guard_confirmatory_subjects(
            [lock["confirmatory_subjects"][0]], lock["confirmatory_subjects"]
        )


def test_manifest_respects_boundaries_roles_and_uniqueness():
    plan = build_pilot_split(small_config())
    rows = list(iter_window_manifest(trial_rows(plan), plan))
    validation = validate_window_manifest(rows, plan)

    assert validation["status"] == "pass"
    assert validation["window_count"] == 4 * 3 * (2 + 14)
    assert validation["window_type_counts"] == {
        "classifier": 4 * 3 * 14,
        "generator": 4 * 3 * 2,
    }
    assert validation["confirmatory_windows_exposed"] == 0
    assert validation["group_role_conflicts"] == 0


def test_manifest_validation_detects_duplicate_windows():
    plan = build_pilot_split(small_config())
    rows = list(iter_window_manifest(trial_rows(plan), plan))
    rows.append(copy.deepcopy(rows[0]))

    with pytest.raises(ValueError, match="duplicate window identifier"):
        validate_window_manifest(rows, plan)


def test_split_validation_detects_participant_overlap():
    plan = build_pilot_split(small_config())
    broken = copy.deepcopy(plan)
    broken["confirmatory_subjects"][0] = broken["development_subjects"][0]

    with pytest.raises(ValueError, match="overlap"):
        validate_pilot_split(broken)

