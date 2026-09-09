from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from emg_diffusion.data.splits import guard_confirmatory_subjects


@dataclass(frozen=True)
class WindowBatch:
    values: np.ndarray
    records: tuple[dict[str, object], ...]


def read_window_manifest(
    path: Path,
    *,
    window_type: str | None = None,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if window_type is not None:
        rows = [row for row in rows if row["window_type"] == window_type]
    if not rows:
        raise ValueError(f"no matching windows found in {path}")
    return rows


def select_central_windows(
    rows: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("window_type") != "generator":
            continue
        grouped[str(row["group_key"])].append(dict(row))
    if not grouped:
        raise ValueError("no generator windows were supplied")

    selected = []
    for group_key, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: int(row["window_index"]))
        target_index = 0.5 * (len(ordered) - 1)
        chosen = min(
            ordered,
            key=lambda row: (
                abs(int(row["window_index"]) - target_index),
                int(row["window_index"]),
            ),
        )
        selected.append(chosen)
    return sorted(
        selected,
        key=lambda row: (
            int(row["subject"]),
            int(row["movement"]),
            int(row["repetition"]),
            int(row["segment_index"]),
        ),
    )


def load_ninapro_windows(
    rows: Sequence[Mapping[str, str]],
    data_root: Path,
    *,
    confirmatory_subjects: Iterable[int] = (),
    expected_channels: int = 12,
    dtype: np.dtype = np.dtype(np.float32),
) -> WindowBatch:
    if not rows:
        raise ValueError("cannot load an empty window selection")
    requested_subjects = {int(row["subject"]) for row in rows}
    guard_confirmatory_subjects(requested_subjects, confirmatory_subjects)

    file_index: dict[str, Path] = {}
    for path in data_root.rglob("*.mat"):
        if path.name in file_index:
            raise ValueError(f"duplicate MATLAB filename below data root: {path.name}")
        file_index[path.name] = path

    rows_by_file: defaultdict[str, list[tuple[int, Mapping[str, str]]]] = defaultdict(list)
    for output_index, row in enumerate(rows):
        rows_by_file[str(row["file"])].append((output_index, row))

    ordered_values: list[np.ndarray | None] = [None] * len(rows)
    ordered_records: list[dict[str, object] | None] = [None] * len(rows)
    for file_name, indexed_rows in sorted(rows_by_file.items()):
        if file_name not in file_index:
            raise FileNotFoundError(f"could not locate {file_name} below {data_root}")
        contents = loadmat(file_index[file_name], variable_names=["emg"])
        if "emg" not in contents:
            raise ValueError(f"{file_name} does not contain an emg array")
        emg = np.asarray(contents["emg"])
        if emg.ndim != 2 or emg.shape[1] != expected_channels:
            raise ValueError(f"unexpected EMG shape in {file_name}: {emg.shape}")

        for output_index, row in indexed_rows:
            start = int(row["start_sample"])
            stop = int(row["stop_sample"])
            if start < 0 or stop > emg.shape[0] or stop <= start:
                raise ValueError(
                    f"invalid [{start}, {stop}) boundary for {file_name}"
                )
            expected_length = int(row["length_samples"])
            values = np.asarray(emg[start:stop].T, dtype=dtype)
            if values.shape != (expected_channels, expected_length):
                raise ValueError(
                    f"window {row['window_id']} has shape {values.shape}, "
                    f"expected {(expected_channels, expected_length)}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"window {row['window_id']} is non-finite")
            ordered_values[output_index] = values
            ordered_records[output_index] = {
                "window_id": str(row["window_id"]),
                "subject": int(row["subject"]),
                "exercise": int(row["exercise"]),
                "movement": int(row["movement"]),
                "repetition": int(row["repetition"]),
                "segment_index": int(row["segment_index"]),
                "file": file_name,
                "start_sample": start,
                "stop_sample": stop,
            }

    if any(value is None for value in ordered_values) or any(
        record is None for record in ordered_records
    ):
        raise RuntimeError("one or more selected windows were not loaded")
    return WindowBatch(
        values=np.stack(ordered_values),  # type: ignore[arg-type]
        records=tuple(ordered_records),  # type: ignore[arg-type]
    )


def record_mask(
    records: Sequence[Mapping[str, object]],
    *,
    subjects: Iterable[int] | None = None,
    repetitions: Iterable[int] | None = None,
    movements: Iterable[int] | None = None,
) -> np.ndarray:
    allowed_subjects = set(subjects) if subjects is not None else None
    allowed_repetitions = set(repetitions) if repetitions is not None else None
    allowed_movements = set(movements) if movements is not None else None
    return np.asarray(
        [
            (allowed_subjects is None or int(row["subject"]) in allowed_subjects)
            and (
                allowed_repetitions is None
                or int(row["repetition"]) in allowed_repetitions
            )
            and (
                allowed_movements is None
                or int(row["movement"]) in allowed_movements
            )
            for row in records
        ],
        dtype=bool,
    )

