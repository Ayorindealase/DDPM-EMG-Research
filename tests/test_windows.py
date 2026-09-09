import csv

import numpy as np
import pytest
from scipy.io import savemat

from emg_diffusion.data.windows import (
    load_ninapro_windows,
    read_window_manifest,
    record_mask,
    select_central_windows,
)


def example_rows():
    rows = []
    for group, subject in (("g1", 1), ("g2", 2)):
        for index in range(3):
            rows.append(
                {
                    "window_id": f"{group}_{index}",
                    "subject": str(subject),
                    "exercise": "1",
                    "file": f"S{subject}_E1_A1.mat",
                    "movement": "1",
                    "repetition": "2",
                    "segment_index": "1",
                    "group_key": group,
                    "window_type": "generator",
                    "window_index": str(index),
                    "start_sample": str(index * 2),
                    "stop_sample": str(index * 2 + 4),
                    "length_samples": "4",
                }
            )
    return rows


def test_select_central_windows_returns_one_per_trial():
    selected = select_central_windows(example_rows())
    assert [row["window_id"] for row in selected] == ["g1_1", "g2_1"]


def test_load_windows_preserves_manifest_order_and_orientation(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    emg_1 = np.arange(48, dtype=float).reshape(12, 4).T
    emg_2 = (100 + np.arange(48, dtype=float)).reshape(12, 4).T
    savemat(data_root / "S1_E1_A1.mat", {"emg": emg_1})
    savemat(data_root / "S2_E1_A1.mat", {"emg": emg_2})
    rows = [example_rows()[0], example_rows()[3]]

    batch = load_ninapro_windows(rows, data_root, expected_channels=12)

    assert batch.values.shape == (2, 12, 4)
    np.testing.assert_array_equal(batch.values[0], emg_1.T)
    np.testing.assert_array_equal(batch.values[1], emg_2.T)
    assert [record["subject"] for record in batch.records] == [1, 2]


def test_load_windows_rejects_confirmatory_participant(tmp_path):
    with pytest.raises(PermissionError):
        load_ninapro_windows(
            [example_rows()[0]],
            tmp_path,
            confirmatory_subjects=[1],
        )


def test_record_mask_combines_subject_repetition_and_movement():
    records = [
        {"subject": 1, "repetition": 1, "movement": 1},
        {"subject": 1, "repetition": 2, "movement": 1},
        {"subject": 2, "repetition": 2, "movement": 2},
    ]
    mask = record_mask(
        records, subjects=[1], repetitions=[2], movements=[1]
    )
    np.testing.assert_array_equal(mask, [False, True, False])


def test_read_window_manifest_filters_type(tmp_path):
    path = tmp_path / "manifest.csv"
    rows = example_rows()[:1]
    rows.append({**rows[0], "window_id": "classifier", "window_type": "classifier"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = read_window_manifest(path, window_type="generator")
    assert [row["window_id"] for row in selected] == ["g1_0"]
