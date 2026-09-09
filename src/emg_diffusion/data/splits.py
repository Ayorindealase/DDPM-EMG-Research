from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PilotSplitConfig:
    seed: int = 20260830
    exercise: int = 1
    subjects: tuple[int, ...] = tuple(range(1, 41))
    development_count: int = 16
    fold_count: int = 4
    target_subjects_per_fold: int = 4
    validation_subjects_per_fold: int = 2
    calibration_repetitions: tuple[int, ...] = (1, 3, 4, 6)
    test_repetitions: tuple[int, ...] = (2, 5)
    expected_movements: tuple[int, ...] = tuple(range(1, 18))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_allocations(
    repetitions: Sequence[int],
) -> list[dict[str, object]]:
    repetitions = tuple(sorted(int(value) for value in repetitions))
    if len(repetitions) != len(set(repetitions)):
        raise ValueError("calibration repetitions must be unique")

    allocations: list[dict[str, object]] = [
        {
            "allocation_id": "b0_none",
            "budget_repetitions": 0,
            "selected_repetitions": [],
            "unused_repetitions": list(repetitions),
        }
    ]
    for budget in (1, 2):
        for selected in itertools.combinations(repetitions, budget):
            suffix = "_".join(f"r{value}" for value in selected)
            allocations.append(
                {
                    "allocation_id": f"b{budget}_{suffix}",
                    "budget_repetitions": budget,
                    "selected_repetitions": list(selected),
                    "unused_repetitions": [
                        value for value in repetitions if value not in selected
                    ],
                }
            )
    return allocations


def build_pilot_split(config: PilotSplitConfig) -> dict[str, object]:
    _validate_config(config)
    rng = np.random.Generator(np.random.PCG64(config.seed))
    participant_order = [int(value) for value in rng.permutation(config.subjects)]
    development_order = participant_order[: config.development_count]
    confirmatory_order = participant_order[config.development_count :]

    folds: list[dict[str, object]] = []
    for fold_index in range(config.fold_count):
        target_start = fold_index * config.target_subjects_per_fold
        target_stop = target_start + config.target_subjects_per_fold
        target = development_order[target_start:target_stop]
        remaining = [value for value in development_order if value not in target]

        fold_rng = np.random.Generator(
            np.random.PCG64(config.seed + 10_000 + fold_index)
        )
        remaining_order = [int(value) for value in fold_rng.permutation(remaining)]
        validation = remaining_order[: config.validation_subjects_per_fold]
        source_train = remaining_order[config.validation_subjects_per_fold :]
        folds.append(
            {
                "fold": fold_index + 1,
                "source_train_subjects": sorted(source_train),
                "source_validation_subjects": sorted(validation),
                "pseudo_target_subjects": sorted(target),
            }
        )

    plan: dict[str, object] = {
        "schema_version": 1,
        "dataset": "NinaPro DB2",
        "exercise": config.exercise,
        "split_seed": config.seed,
        "random_number_generator": "NumPy PCG64",
        "participant_universe": sorted(config.subjects),
        "participant_random_order": participant_order,
        "development_subjects": sorted(development_order),
        "development_subject_random_order": development_order,
        "confirmatory_subjects": sorted(confirmatory_order),
        "folds": folds,
        "repetition_policy": {
            "target_calibration_candidates": sorted(
                config.calibration_repetitions
            ),
            "target_test_repetitions": sorted(config.test_repetitions),
            "source_train_and_validation_repetitions": sorted(
                set(config.calibration_repetitions) | set(config.test_repetitions)
            ),
        },
        "expected_active_movements": sorted(config.expected_movements),
        "calibration_allocations": calibration_allocations(
            config.calibration_repetitions
        ),
    }
    validate_pilot_split(plan)
    return plan


def _validate_config(config: PilotSplitConfig) -> None:
    if len(config.subjects) != len(set(config.subjects)):
        raise ValueError("participant identifiers must be unique")
    if config.development_count <= 0 or config.development_count >= len(
        config.subjects
    ):
        raise ValueError("development_count must be between zero and subject count")
    if (
        config.fold_count * config.target_subjects_per_fold
        != config.development_count
    ):
        raise ValueError(
            "fold_count times target_subjects_per_fold must equal development_count"
        )
    remaining_per_fold = (
        config.development_count - config.target_subjects_per_fold
    )
    if not 0 < config.validation_subjects_per_fold < remaining_per_fold:
        raise ValueError("validation count leaves no source-training participants")
    if set(config.calibration_repetitions) & set(config.test_repetitions):
        raise ValueError("calibration and test repetitions must be disjoint")


def validate_pilot_split(plan: Mapping[str, object]) -> dict[str, int]:
    universe = _integer_set(plan["participant_universe"])
    development = _integer_set(plan["development_subjects"])
    confirmatory = _integer_set(plan["confirmatory_subjects"])
    if development & confirmatory:
        raise ValueError("development and confirmatory participants overlap")
    if development | confirmatory != universe:
        raise ValueError("development and confirmatory participants do not cover universe")

    target_counts: Counter[int] = Counter()
    for fold in plan["folds"]:  # type: ignore[index]
        source_train = _integer_set(fold["source_train_subjects"])
        source_validation = _integer_set(fold["source_validation_subjects"])
        pseudo_target = _integer_set(fold["pseudo_target_subjects"])
        if source_train & source_validation:
            raise ValueError("source-training and validation participants overlap")
        if source_train & pseudo_target or source_validation & pseudo_target:
            raise ValueError("source and pseudo-target participants overlap")
        if source_train | source_validation | pseudo_target != development:
            raise ValueError("a fold does not cover every development participant")
        target_counts.update(pseudo_target)

    if set(target_counts) != development or any(
        count != 1 for count in target_counts.values()
    ):
        raise ValueError("every development participant must be pseudo-target once")

    repetition_policy = plan["repetition_policy"]  # type: ignore[index]
    calibration = _integer_set(repetition_policy["target_calibration_candidates"])
    test = _integer_set(repetition_policy["target_test_repetitions"])
    if calibration & test:
        raise ValueError("target calibration and test repetitions overlap")

    return {
        "participant_count": len(universe),
        "development_count": len(development),
        "confirmatory_count": len(confirmatory),
        "fold_count": len(plan["folds"]),  # type: ignore[arg-type]
    }


def _integer_set(values: object) -> set[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("expected a sequence of integer identifiers")
    converted = {int(value) for value in values}
    if len(converted) != len(values):
        raise ValueError("identifier sequence contains duplicates")
    return converted


def confirmatory_lock(plan: Mapping[str, object]) -> dict[str, object]:
    validate_pilot_split(plan)
    return {
        "schema_version": 1,
        "dataset": plan["dataset"],
        "split_seed": plan["split_seed"],
        "status": "locked",
        "confirmatory_subjects": plan["confirmatory_subjects"],
        "permitted_before_confirmation": [
            "file-presence audit",
            "metadata and signal-integrity audit without model outcomes",
        ],
        "prohibited_before_confirmation": [
            "model fitting",
            "hyperparameter selection",
            "preprocessing selection",
            "covariance fitting",
            "early stopping",
            "outcome inspection",
        ],
    }


def guard_confirmatory_subjects(
    requested_subjects: Iterable[int],
    locked_subjects: Iterable[int],
) -> None:
    overlap = sorted(set(requested_subjects) & set(locked_subjects))
    if overlap:
        raise PermissionError(
            "confirmatory participant access is locked during development: "
            + ", ".join(str(value) for value in overlap)
        )


def iter_window_manifest(
    trial_rows: Iterable[Mapping[str, str]],
    plan: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    validate_pilot_split(plan)
    exercise = int(plan["exercise"])
    development = _integer_set(plan["development_subjects"])
    confirmatory = _integer_set(plan["confirmatory_subjects"])
    expected_movements = _integer_set(plan["expected_active_movements"])
    repetition_policy = plan["repetition_policy"]  # type: ignore[index]
    calibration_repetitions = _integer_set(
        repetition_policy["target_calibration_candidates"]
    )
    test_repetitions = _integer_set(
        repetition_policy["target_test_repetitions"]
    )
    expected_repetitions = calibration_repetitions | test_repetitions

    coverage: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
    selected_rows: list[Mapping[str, str]] = []
    for row in trial_rows:
        subject = int(row["subject"])
        if int(row["exercise"]) != exercise or subject not in development:
            continue
        guard_confirmatory_subjects([subject], confirmatory)
        movement = int(row["movement"])
        repetition = int(row["repetition"])
        coverage[subject].add((movement, repetition))
        selected_rows.append(row)

    expected_pairs = {
        (movement, repetition)
        for movement in expected_movements
        for repetition in expected_repetitions
    }
    for subject in sorted(development):
        missing = sorted(expected_pairs - coverage[subject])
        if missing:
            raise ValueError(
                f"subject {subject} is missing movement/repetition pairs: {missing}"
            )

    for row in selected_rows:
        subject = int(row["subject"])
        movement = int(row["movement"])
        repetition = int(row["repetition"])
        segment_index = int(row["segment_index"])
        trial_start = int(row["start_sample"])
        trial_stop = int(row["stop_sample"])
        group_key = (
            f"S{subject:02d}_E{exercise}_M{movement:02d}_"
            f"R{repetition}_G{segment_index}"
        )
        roles = {
            f"fold_{int(fold['fold'])}_role": _role_for_trial(
                fold,
                subject,
                repetition,
                calibration_repetitions,
                test_repetitions,
            )
            for fold in plan["folds"]  # type: ignore[index]
        }

        for window_type in ("generator", "classifier"):
            length = int(row[f"{window_type}_length_samples"])
            step = int(row[f"{window_type}_step_samples"])
            count = int(row[f"{window_type}_window_count"])
            for window_index in range(count):
                start = trial_start + window_index * step
                stop = start + length
                if stop > trial_stop:
                    raise ValueError(
                        f"{group_key} {window_type} window exceeds trial boundary"
                    )
                identity = (
                    f"{row['file']}|{group_key}|{window_type}|{start}|{stop}"
                )
                yield {
                    "window_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                    "subject": subject,
                    "exercise": exercise,
                    "file": row["file"],
                    "movement": movement,
                    "repetition": repetition,
                    "segment_index": segment_index,
                    "group_key": group_key,
                    "window_type": window_type,
                    "window_index": window_index,
                    "start_sample": start,
                    "stop_sample": stop,
                    "length_samples": length,
                    "step_samples": step,
                    **roles,
                }


def _role_for_trial(
    fold: Mapping[str, object],
    subject: int,
    repetition: int,
    calibration_repetitions: set[int],
    test_repetitions: set[int],
) -> str:
    if subject in _integer_set(fold["source_train_subjects"]):
        return "source_train"
    if subject in _integer_set(fold["source_validation_subjects"]):
        return "source_validation"
    if subject not in _integer_set(fold["pseudo_target_subjects"]):
        raise ValueError(f"subject {subject} has no role in fold {fold['fold']}")
    if repetition in calibration_repetitions:
        return "target_calibration_candidate"
    if repetition in test_repetitions:
        return "target_test"
    raise ValueError(f"unexpected target repetition {repetition}")


def validate_window_manifest(
    rows: Iterable[Mapping[str, object]],
    plan: Mapping[str, object],
) -> dict[str, object]:
    validate_pilot_split(plan)
    development = _integer_set(plan["development_subjects"])
    confirmatory = _integer_set(plan["confirmatory_subjects"])
    repetition_policy = plan["repetition_policy"]  # type: ignore[index]
    calibration = _integer_set(repetition_policy["target_calibration_candidates"])
    test = _integer_set(repetition_policy["target_test_repetitions"])
    fold_numbers = [int(fold["fold"]) for fold in plan["folds"]]  # type: ignore[index]

    identifiers: set[str] = set()
    group_roles: dict[tuple[int, str], str] = {}
    role_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    type_counts: Counter[str] = Counter()
    subject_counts: Counter[int] = Counter()
    row_count = 0

    for row in rows:
        row_count += 1
        window_id = str(row["window_id"])
        if window_id in identifiers:
            raise ValueError(f"duplicate window identifier: {window_id}")
        identifiers.add(window_id)

        subject = int(row["subject"])
        repetition = int(row["repetition"])
        if subject not in development or subject in confirmatory:
            raise ValueError(f"unauthorized participant in manifest: {subject}")
        if int(row["stop_sample"]) - int(row["start_sample"]) != int(
            row["length_samples"]
        ):
            raise ValueError(f"invalid window length for {window_id}")
        if int(row["start_sample"]) < 0:
            raise ValueError(f"negative window boundary for {window_id}")

        group_key = str(row["group_key"])
        for fold_number in fold_numbers:
            column = f"fold_{fold_number}_role"
            role = str(row[column])
            expected_role = _role_for_trial(
                plan["folds"][fold_number - 1],  # type: ignore[index]
                subject,
                repetition,
                calibration,
                test,
            )
            if role != expected_role:
                raise ValueError(
                    f"incorrect role for {window_id} in fold {fold_number}: "
                    f"{role} != {expected_role}"
                )
            group_role_key = (fold_number, group_key)
            previous = group_roles.setdefault(group_role_key, role)
            if previous != role:
                raise ValueError(
                    f"group {group_key} crosses roles in fold {fold_number}"
                )
            role_counts[f"fold_{fold_number}"][role] += 1

        type_counts[str(row["window_type"])] += 1
        subject_counts[subject] += 1

    if not identifiers:
        raise ValueError("window manifest is empty")
    if set(subject_counts) != development:
        missing = sorted(development - set(subject_counts))
        raise ValueError(f"development participants missing from manifest: {missing}")

    return {
        "status": "pass",
        "window_count": row_count,
        "unique_window_count": len(identifiers),
        "window_type_counts": dict(sorted(type_counts.items())),
        "participant_window_counts": {
            str(key): value for key, value in sorted(subject_counts.items())
        },
        "fold_role_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(role_counts.items())
        },
        "confirmatory_windows_exposed": 0,
        "group_role_conflicts": 0,
    }


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

