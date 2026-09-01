from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat


MAT_FILE_PATTERN = re.compile(
    r"S(?P<subject>\d+)_E(?P<exercise>\d+)_A(?P<acquisition>\d+)\.mat$",
    re.IGNORECASE,
)
EXPECTED_LABELS = {
    1: tuple(range(1, 18)),
    2: tuple(range(18, 41)),
    3: tuple(range(41, 50)),
}
EXPECTED_REPETITIONS = tuple(range(1, 7))
REQUIRED_VARIABLES = (
    "subject",
    "exercise",
    "emg",
    "stimulus",
    "restimulus",
    "repetition",
    "rerepetition",
)


@dataclass(frozen=True)
class WindowDefinition:
    name: str
    length_samples: int
    step_samples: int


@dataclass(frozen=True)
class AuditConfig:
    data_root: Path
    output_dir: Path
    sampling_rate_hz: int = 2000
    expected_channels: int = 12
    label_source: str = "refined"
    subjects: tuple[int, ...] | None = None
    extreme_repeat_threshold: float = 0.001
    max_tail_mismatch: int = 400
    generator_window: WindowDefinition = WindowDefinition("generator", 2000, 1000)
    classifier_window: WindowDefinition = WindowDefinition("classifier", 400, 200)


def count_windows(sample_count: int, length_samples: int, step_samples: int) -> int:
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if length_samples <= 0 or step_samples <= 0:
        raise ValueError("window length and step must be positive")
    if sample_count < length_samples:
        return 0
    return 1 + (sample_count - length_samples) // step_samples


def contiguous_trials(labels: np.ndarray, repetitions: np.ndarray) -> list[dict[str, int]]:
    labels = np.asarray(labels).reshape(-1)
    repetitions = np.asarray(repetitions).reshape(-1)
    if labels.shape != repetitions.shape:
        raise ValueError("label and repetition arrays must have the same length")
    if labels.size == 0:
        return []

    changes = np.flatnonzero(
        (labels[1:] != labels[:-1]) | (repetitions[1:] != repetitions[:-1])
    ) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [labels.size]))
    occurrence: Counter[tuple[int, int]] = Counter()
    trials: list[dict[str, int]] = []

    for start, stop in zip(starts, stops, strict=True):
        movement = int(labels[start])
        repetition = int(repetitions[start])
        if movement <= 0 or repetition <= 0:
            continue
        key = (movement, repetition)
        occurrence[key] += 1
        trials.append(
            {
                "movement": movement,
                "repetition": repetition,
                "segment_index": occurrence[key],
                "start_sample": int(start),
                "stop_sample": int(stop),
                "sample_count": int(stop - start),
            }
        )
    return trials


def parse_mat_identity(path: Path) -> tuple[int, int, int]:
    match = MAT_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unrecognized DB2 filename: {path.name}")
    return (
        int(match.group("subject")),
        int(match.group("exercise")),
        int(match.group("acquisition")),
    )


def discover_mat_files(data_root: Path, subjects: Sequence[int] | None = None) -> list[Path]:
    selected = set(subjects) if subjects is not None else None
    paths: list[tuple[int, int, int, Path]] = []
    for path in data_root.rglob("*.mat"):
        subject, exercise, acquisition = parse_mat_identity(path)
        if selected is None or subject in selected:
            paths.append((subject, exercise, acquisition, path))
    paths.sort(key=lambda item: item[:3])
    return [item[3] for item in paths]


def channel_statistics(
    emg: np.ndarray,
    *,
    subject: int,
    exercise: int,
    file_name: str,
    extreme_repeat_threshold: float,
) -> list[dict[str, object]]:
    if emg.ndim != 2:
        raise ValueError(f"EMG array must be two-dimensional, received {emg.shape}")

    rows: list[dict[str, object]] = []
    for channel_index in range(emg.shape[1]):
        values = np.asarray(emg[:, channel_index], dtype=np.float64)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        nonfinite_count = int(values.size - finite.size)

        row: dict[str, object] = {
            "subject": subject,
            "exercise": exercise,
            "file": file_name,
            "channel": channel_index + 1,
            "sample_count": int(values.size),
            "finite_count": int(finite.size),
            "nonfinite_count": nonfinite_count,
        }

        if finite.size == 0:
            row.update(
                {
                    "minimum": "",
                    "maximum": "",
                    "mean": "",
                    "standard_deviation": "",
                    "rms": "",
                    "p001": "",
                    "p01": "",
                    "median": "",
                    "p99": "",
                    "p999": "",
                    "zero_fraction": "",
                    "flat_difference_fraction": "",
                    "extreme_repeat_fraction": "",
                    "constant_channel": True,
                    "potential_saturation": False,
                }
            )
            rows.append(row)
            continue

        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        quantiles = np.percentile(finite, [0.1, 1.0, 50.0, 99.0, 99.9])
        valid_pairs = finite_mask[1:] & finite_mask[:-1]
        if np.any(valid_pairs):
            flat_difference_fraction = float(
                np.mean(np.diff(values)[valid_pairs] == 0.0)
            )
        else:
            flat_difference_fraction = 0.0

        minimum_fraction = float(np.mean(finite == minimum))
        maximum_fraction = float(np.mean(finite == maximum))
        extreme_repeat_fraction = max(minimum_fraction, maximum_fraction)
        constant = minimum == maximum

        row.update(
            {
                "minimum": minimum,
                "maximum": maximum,
                "mean": float(np.mean(finite)),
                "standard_deviation": float(np.std(finite, ddof=0)),
                "rms": float(np.sqrt(np.mean(np.square(finite)))),
                "p001": float(quantiles[0]),
                "p01": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p99": float(quantiles[3]),
                "p999": float(quantiles[4]),
                "zero_fraction": float(np.mean(finite == 0.0)),
                "flat_difference_fraction": flat_difference_fraction,
                "extreme_repeat_fraction": extreme_repeat_fraction,
                "constant_channel": constant,
                "potential_saturation": bool(
                    not constant and extreme_repeat_fraction >= extreme_repeat_threshold
                ),
            }
        )
        rows.append(row)
    return rows


def _scalar_integer(value: np.ndarray, variable: str) -> int:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{variable} must contain one value, received {array.shape}")
    return int(array[0])


def _integer_vector(value: np.ndarray, variable: str) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            raise ValueError(f"{variable} must contain finite integer labels")
    return array.astype(np.int32, copy=False)


def align_time_series(
    emg: np.ndarray,
    vectors: dict[str, np.ndarray],
    max_tail_mismatch: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    if max_tail_mismatch < 0:
        raise ValueError("max_tail_mismatch must be non-negative")

    lengths = {"emg": int(emg.shape[0])}
    lengths.update({name: int(values.size) for name, values in vectors.items()})
    aligned_length = min(lengths.values())
    if max(lengths.values()) - aligned_length > max_tail_mismatch:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            f"time-series length mismatch exceeds {max_tail_mismatch} sample(s): "
            f"{detail}"
        )

    removed = {name: length - aligned_length for name, length in lengths.items()}
    aligned_vectors = {
        name: values[:aligned_length] for name, values in vectors.items()
    }
    return emg[:aligned_length], aligned_vectors, removed


def _joined_integers(values: Iterable[int]) -> str:
    return ";".join(str(value) for value in sorted(set(values)))


def _add_issue(
    issues: list[dict[str, object]],
    *,
    severity: str,
    subject: int | str,
    exercise: int | str,
    file_name: str,
    category: str,
    detail: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "subject": subject,
            "exercise": exercise,
            "file": file_name,
            "category": category,
            "detail": detail,
        }
    )


def audit_file(
    path: Path,
    config: AuditConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    file_subject, file_exercise, acquisition = parse_mat_identity(path)
    issues: list[dict[str, object]] = []
    try:
        contents = loadmat(path, variable_names=REQUIRED_VARIABLES)
    except Exception as error:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="load_failure",
            detail=f"{type(error).__name__}: {error}",
        )
        return {}, [], [], [], issues

    missing = [name for name in REQUIRED_VARIABLES if name not in contents]
    if missing:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="missing_variables",
            detail=", ".join(missing),
        )
        return {}, [], [], [], issues

    emg = np.asarray(contents["emg"])
    if emg.ndim != 2:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="invalid_emg_shape",
            detail=str(emg.shape),
        )
        return {}, [], [], [], issues

    original_emg_samples, channel_count = emg.shape
    try:
        stored_subject = _scalar_integer(contents["subject"], "subject")
        stored_exercise = _scalar_integer(contents["exercise"], "exercise")
        vectors = {
            "stimulus": _integer_vector(contents["stimulus"], "stimulus"),
            "restimulus": _integer_vector(contents["restimulus"], "restimulus"),
            "repetition": _integer_vector(contents["repetition"], "repetition"),
            "rerepetition": _integer_vector(
                contents["rerepetition"], "rerepetition"
            ),
        }
        original_lengths = {
            "emg": original_emg_samples,
            **{name: int(values.size) for name, values in vectors.items()},
        }
        emg, vectors, removed_samples = align_time_series(
            emg,
            vectors,
            config.max_tail_mismatch,
        )
    except ValueError as error:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="invalid_metadata",
            detail=str(error),
        )
        return {}, [], [], [], issues

    stimulus = vectors["stimulus"]
    restimulus = vectors["restimulus"]
    repetition = vectors["repetition"]
    rerepetition = vectors["rerepetition"]
    sample_count = int(emg.shape[0])

    if any(removed_samples.values()):
        removed_detail = ", ".join(
            f"{name}={count}"
            for name, count in removed_samples.items()
            if count > 0
        )
        _add_issue(
            issues,
            severity="warning",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="trailing_length_alignment",
            detail=(
                f"aligned all streams to {sample_count} samples; "
                f"removed trailing samples: {removed_detail}"
            ),
        )

    if stored_subject != file_subject:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="subject_mismatch",
            detail=f"filename={file_subject}, variable={stored_subject}",
        )
    if stored_exercise != file_exercise:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="exercise_mismatch",
            detail=f"filename={file_exercise}, variable={stored_exercise}",
        )
    if channel_count != config.expected_channels:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="channel_count",
            detail=f"observed={channel_count}, expected={config.expected_channels}",
        )

    if config.label_source == "refined":
        labels, repetitions = restimulus, rerepetition
    else:
        labels, repetitions = stimulus, repetition

    active_labels = sorted(int(value) for value in np.unique(labels) if value > 0)
    active_repetitions = sorted(
        int(value) for value in np.unique(repetitions) if value > 0
    )
    expected_labels = list(EXPECTED_LABELS.get(file_exercise, ()))
    if active_labels != expected_labels:
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="movement_coverage",
            detail=f"observed={active_labels}, expected={expected_labels}",
        )
    if active_repetitions != list(EXPECTED_REPETITIONS):
        _add_issue(
            issues,
            severity="error",
            subject=file_subject,
            exercise=file_exercise,
            file_name=path.name,
            category="repetition_coverage",
            detail=(
                f"observed={active_repetitions}, "
                f"expected={list(EXPECTED_REPETITIONS)}"
            ),
        )

    segments = contiguous_trials(labels, repetitions)
    pair_counts = Counter((row["movement"], row["repetition"]) for row in segments)
    for (movement, rep), segment_count in pair_counts.items():
        if segment_count > 1:
            _add_issue(
                issues,
                severity="warning",
                subject=file_subject,
                exercise=file_exercise,
                file_name=path.name,
                category="fragmented_trial",
                detail=(
                    f"movement={movement}, repetition={rep}, "
                    f"segments={segment_count}"
                ),
            )

    trial_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for segment in segments:
        duration_seconds = segment["sample_count"] / config.sampling_rate_hz
        common = {
            "subject": file_subject,
            "exercise": file_exercise,
            "file": path.name,
            "label_source": config.label_source,
            "movement": segment["movement"],
            "repetition": segment["repetition"],
            "segment_index": segment["segment_index"],
            "start_sample": segment["start_sample"],
            "stop_sample": segment["stop_sample"],
            "sample_count": segment["sample_count"],
            "duration_seconds": duration_seconds,
        }
        trial_rows.append(common)
        window_rows.append(
            {
                **common,
                "generator_length_samples": config.generator_window.length_samples,
                "generator_step_samples": config.generator_window.step_samples,
                "generator_window_count": count_windows(
                    segment["sample_count"],
                    config.generator_window.length_samples,
                    config.generator_window.step_samples,
                ),
                "classifier_length_samples": config.classifier_window.length_samples,
                "classifier_step_samples": config.classifier_window.step_samples,
                "classifier_window_count": count_windows(
                    segment["sample_count"],
                    config.classifier_window.length_samples,
                    config.classifier_window.step_samples,
                ),
            }
        )

    channel_rows = channel_statistics(
        emg,
        subject=file_subject,
        exercise=file_exercise,
        file_name=path.name,
        extreme_repeat_threshold=config.extreme_repeat_threshold,
    )
    nonfinite_values = sum(int(row["nonfinite_count"]) for row in channel_rows)

    file_row: dict[str, object] = {
        "subject": file_subject,
        "exercise": file_exercise,
        "acquisition": acquisition,
        "file": path.name,
        "relative_path": str(path.relative_to(config.data_root)),
        "file_bytes": path.stat().st_size,
        "original_emg_samples": original_lengths["emg"],
        "original_stimulus_samples": original_lengths["stimulus"],
        "original_restimulus_samples": original_lengths["restimulus"],
        "original_repetition_samples": original_lengths["repetition"],
        "original_rerepetition_samples": original_lengths["rerepetition"],
        "trailing_samples_removed": sum(removed_samples.values()),
        "emg_samples_removed": removed_samples["emg"],
        "stimulus_samples_removed": removed_samples["stimulus"],
        "restimulus_samples_removed": removed_samples["restimulus"],
        "repetition_samples_removed": removed_samples["repetition"],
        "rerepetition_samples_removed": removed_samples["rerepetition"],
        "emg_samples": sample_count,
        "emg_channels": channel_count,
        "emg_dtype": str(emg.dtype),
        "sampling_rate_hz": config.sampling_rate_hz,
        "recording_duration_seconds": sample_count / config.sampling_rate_hz,
        "active_labels": _joined_integers(active_labels),
        "active_repetitions": _joined_integers(active_repetitions),
        "active_trial_segments": len(segments),
        "nonfinite_emg_values": nonfinite_values,
        "stimulus_restimulus_disagreement_samples": int(
            np.count_nonzero(stimulus != restimulus)
        ),
        "repetition_rerepetition_disagreement_samples": int(
            np.count_nonzero(repetition != rerepetition)
        ),
    }
    return file_row, trial_rows, window_rows, channel_rows, issues


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _participant_rows(
    file_rows: Sequence[dict[str, object]],
    window_rows: Sequence[dict[str, object]],
    channel_rows: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    files_by_subject: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    windows_by_subject: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    channels_by_subject: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    issues_by_subject: Counter[int] = Counter()

    for row in file_rows:
        files_by_subject[int(row["subject"])].append(row)
    for row in window_rows:
        windows_by_subject[int(row["subject"])].append(row)
    for row in channel_rows:
        channels_by_subject[int(row["subject"])].append(row)
    for row in issues:
        if isinstance(row["subject"], int):
            issues_by_subject[int(row["subject"])] += 1

    rows: list[dict[str, object]] = []
    for subject in sorted(files_by_subject):
        subject_files = files_by_subject[subject]
        subject_windows = windows_by_subject[subject]
        subject_channels = channels_by_subject[subject]
        rows.append(
            {
                "subject": subject,
                "file_count": len(subject_files),
                "exercise_count": len(
                    {int(row["exercise"]) for row in subject_files}
                ),
                "total_samples": sum(int(row["emg_samples"]) for row in subject_files),
                "total_duration_seconds": sum(
                    float(row["recording_duration_seconds"])
                    for row in subject_files
                ),
                "generator_window_count": sum(
                    int(row["generator_window_count"]) for row in subject_windows
                ),
                "classifier_window_count": sum(
                    int(row["classifier_window_count"]) for row in subject_windows
                ),
                "nonfinite_emg_values": sum(
                    int(row["nonfinite_emg_values"]) for row in subject_files
                ),
                "constant_channels": sum(
                    bool(row["constant_channel"]) for row in subject_channels
                ),
                "potential_saturation_channels": sum(
                    bool(row["potential_saturation"]) for row in subject_channels
                ),
                "issue_count": issues_by_subject[subject],
            }
        )
    return rows


def audit_dataset(config: AuditConfig) -> dict[str, object]:
    if config.label_source not in {"original", "refined"}:
        raise ValueError("label_source must be 'original' or 'refined'")
    if config.sampling_rate_hz <= 0 or config.expected_channels <= 0:
        raise ValueError("sampling rate and channel count must be positive")
    if not 0.0 <= config.extreme_repeat_threshold <= 1.0:
        raise ValueError("extreme_repeat_threshold must lie in [0, 1]")
    if config.max_tail_mismatch < 0:
        raise ValueError("max_tail_mismatch must be non-negative")

    paths = discover_mat_files(config.data_root, config.subjects)
    requested_subjects = (
        tuple(sorted(config.subjects))
        if config.subjects is not None
        else tuple(range(1, 41))
    )
    expected_pairs = {
        (subject, exercise)
        for subject in requested_subjects
        for exercise in EXPECTED_LABELS
    }
    observed_pairs = [parse_mat_identity(path)[:2] for path in paths]

    issues: list[dict[str, object]] = []
    observed_pair_counts = Counter(observed_pairs)
    for subject, exercise in sorted(expected_pairs - set(observed_pairs)):
        _add_issue(
            issues,
            severity="error",
            subject=subject,
            exercise=exercise,
            file_name="",
            category="missing_file",
            detail="expected one MATLAB file",
        )
    for (subject, exercise), count in sorted(observed_pair_counts.items()):
        if count > 1:
            _add_issue(
                issues,
                severity="error",
                subject=subject,
                exercise=exercise,
                file_name="",
                category="duplicate_file",
                detail=f"observed {count} MATLAB files",
            )

    file_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []

    for index, path in enumerate(paths, start=1):
        subject, exercise, _ = parse_mat_identity(path)
        print(
            f"[{index:03d}/{len(paths):03d}] "
            f"auditing subject {subject:02d}, exercise {exercise}"
        )
        file_row, trials, windows, channels, file_issues = audit_file(path, config)
        if file_row:
            file_rows.append(file_row)
        trial_rows.extend(trials)
        window_rows.extend(windows)
        channel_rows.extend(channels)
        issues.extend(file_issues)

    participant_rows = _participant_rows(
        file_rows, window_rows, channel_rows, issues
    )
    error_count = sum(row["severity"] == "error" for row in issues)
    warning_count = sum(row["severity"] == "warning" for row in issues)
    constant_channels = sum(bool(row["constant_channel"]) for row in channel_rows)
    saturation_candidates = sum(
        bool(row["potential_saturation"]) for row in channel_rows
    )

    summary: dict[str, object] = {
        "audit_status": "pass" if error_count == 0 else "fail",
        "generated_utc": datetime.now(UTC).isoformat(),
        "data_root": str(config.data_root.resolve()),
        "label_source": config.label_source,
        "sampling_rate_hz": config.sampling_rate_hz,
        "expected_channels": config.expected_channels,
        "max_tail_mismatch_samples": config.max_tail_mismatch,
        "max_tail_mismatch_milliseconds": (
            1000.0 * config.max_tail_mismatch / config.sampling_rate_hz
        ),
        "subjects_requested": list(requested_subjects),
        "subject_count": len({int(row["subject"]) for row in file_rows}),
        "expected_file_count": len(expected_pairs),
        "observed_file_count": len(paths),
        "successfully_loaded_file_count": len(file_rows),
        "trial_segment_count": len(trial_rows),
        "channel_record_count": len(channel_rows),
        "total_emg_samples": sum(int(row["emg_samples"]) for row in file_rows),
        "total_recording_hours": (
            sum(float(row["recording_duration_seconds"]) for row in file_rows)
            / 3600.0
        ),
        "generator_window_count": sum(
            int(row["generator_window_count"]) for row in window_rows
        ),
        "classifier_window_count": sum(
            int(row["classifier_window_count"]) for row in window_rows
        ),
        "nonfinite_emg_values": sum(
            int(row["nonfinite_emg_values"]) for row in file_rows
        ),
        "constant_channels": constant_channels,
        "potential_saturation_channels": saturation_candidates,
        "stimulus_restimulus_disagreement_samples": sum(
            int(row["stimulus_restimulus_disagreement_samples"])
            for row in file_rows
        ),
        "repetition_rerepetition_disagreement_samples": sum(
            int(row["repetition_rerepetition_disagreement_samples"])
            for row in file_rows
        ),
        "aligned_file_count": sum(
            int(row["trailing_samples_removed"]) > 0 for row in file_rows
        ),
        "emg_tail_samples_removed": sum(
            int(row["emg_samples_removed"]) for row in file_rows
        ),
        "label_tail_samples_removed": sum(
            int(row["stimulus_samples_removed"])
            + int(row["restimulus_samples_removed"])
            + int(row["repetition_samples_removed"])
            + int(row["rerepetition_samples_removed"])
            for row in file_rows
        ),
        "error_count": error_count,
        "warning_count": warning_count,
        "generator_window": {
            "length_samples": config.generator_window.length_samples,
            "step_samples": config.generator_window.step_samples,
        },
        "classifier_window": {
            "length_samples": config.classifier_window.length_samples,
            "step_samples": config.classifier_window.step_samples,
        },
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "file_inventory.csv", file_rows)
    _write_csv(config.output_dir / "trial_inventory.csv", trial_rows)
    _write_csv(config.output_dir / "channel_quality.csv", channel_rows)
    _write_csv(config.output_dir / "window_counts.csv", window_rows)
    _write_csv(config.output_dir / "participant_summary.csv", participant_rows)
    _write_csv(config.output_dir / "issues.csv", issues)
    _write_json(config.output_dir / "audit_summary.json", summary)
    _write_report(config.output_dir / "audit_report.txt", summary)
    return summary


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "NINAPRO DB2 DATA AUDIT",
        "======================",
        f"Status: {str(summary['audit_status']).upper()}",
        f"Generated (UTC): {summary['generated_utc']}",
        f"Source directory: {summary['data_root']}",
        f"Primary labels: {summary['label_source']}",
        (
            "Terminal alignment limit: "
            f"{summary['max_tail_mismatch_samples']} samples "
            f"({float(summary['max_tail_mismatch_milliseconds']):.3f} ms)"
        ),
        "",
        "DATA ACCOUNTING",
        f"Subjects: {summary['subject_count']}",
        (
            "MATLAB files: "
            f"{summary['successfully_loaded_file_count']} loaded / "
            f"{summary['expected_file_count']} expected"
        ),
        f"Trial segments: {summary['trial_segment_count']}",
        f"Total EMG samples: {summary['total_emg_samples']}",
        f"Total recording hours: {float(summary['total_recording_hours']):.3f}",
        "",
        "WINDOW ACCOUNTING",
        f"Generator windows: {summary['generator_window_count']}",
        f"Classifier windows: {summary['classifier_window_count']}",
        "",
        "QUALITY SCREEN",
        f"Non-finite EMG values: {summary['nonfinite_emg_values']}",
        f"Constant channel records: {summary['constant_channels']}",
        (
            "Potential saturation channel records: "
            f"{summary['potential_saturation_channels']}"
        ),
        (
            "Stimulus/restimulus disagreement samples: "
            f"{summary['stimulus_restimulus_disagreement_samples']}"
        ),
        (
            "Repetition/rerepetition disagreement samples: "
            f"{summary['repetition_rerepetition_disagreement_samples']}"
        ),
        f"Files aligned at the terminal boundary: {summary['aligned_file_count']}",
        f"EMG tail samples removed: {summary['emg_tail_samples_removed']}",
        f"Label tail samples removed: {summary['label_tail_samples_removed']}",
        f"Errors: {summary['error_count']}",
        f"Warnings: {summary['warning_count']}",
        "",
        "Interpret potential saturation as a screening flag, not an exclusion rule.",
        "Review issues.csv and participant_summary.csv before creating splits.",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the official NinaPro DB2 MATLAB release before modelling."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/ninapro_db2/extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_audit"),
    )
    parser.add_argument("--sampling-rate-hz", type=int, default=2000)
    parser.add_argument("--expected-channels", type=int, default=12)
    parser.add_argument(
        "--label-source",
        choices=("original", "refined"),
        default="refined",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Optional subject IDs for a limited engineering run.",
    )
    parser.add_argument("--generator-window-ms", type=int, default=1000)
    parser.add_argument("--generator-step-ms", type=int, default=500)
    parser.add_argument("--classifier-window-ms", type=int, default=200)
    parser.add_argument("--classifier-step-ms", type=int, default=100)
    parser.add_argument("--extreme-repeat-threshold", type=float, default=0.001)
    parser.add_argument(
        "--max-tail-mismatch",
        type=int,
        default=400,
        help=(
            "Largest terminal stream mismatch to align, in samples. The default "
            "equals one 200 ms classifier window at 2 kHz."
        ),
    )
    return parser


def _milliseconds_to_samples(milliseconds: int, sampling_rate_hz: int) -> int:
    if milliseconds <= 0:
        raise ValueError("window durations must be positive")
    samples = milliseconds * sampling_rate_hz
    if samples % 1000 != 0:
        raise ValueError(
            f"{milliseconds} ms is not an integer number of samples at "
            f"{sampling_rate_hz} Hz"
        )
    return samples // 1000


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = AuditConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sampling_rate_hz=args.sampling_rate_hz,
        expected_channels=args.expected_channels,
        label_source=args.label_source,
        subjects=tuple(args.subjects) if args.subjects is not None else None,
        extreme_repeat_threshold=args.extreme_repeat_threshold,
        max_tail_mismatch=args.max_tail_mismatch,
        generator_window=WindowDefinition(
            "generator",
            _milliseconds_to_samples(
                args.generator_window_ms, args.sampling_rate_hz
            ),
            _milliseconds_to_samples(args.generator_step_ms, args.sampling_rate_hz),
        ),
        classifier_window=WindowDefinition(
            "classifier",
            _milliseconds_to_samples(
                args.classifier_window_ms, args.sampling_rate_hz
            ),
            _milliseconds_to_samples(args.classifier_step_ms, args.sampling_rate_hz),
        ),
    )
    summary = audit_dataset(config)
    print(
        f"Audit {summary['audit_status']}: "
        f"{summary['successfully_loaded_file_count']} files, "
        f"{summary['error_count']} errors, "
        f"{summary['warning_count']} warnings"
    )
    return 0 if summary["audit_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
