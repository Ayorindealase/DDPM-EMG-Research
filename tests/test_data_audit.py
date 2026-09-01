import numpy as np
import pytest

from emg_diffusion.data.audit import (
    align_time_series,
    channel_statistics,
    contiguous_trials,
    count_windows,
)


@pytest.mark.parametrize(
    ("sample_count", "length", "step", "expected"),
    [
        (1999, 2000, 1000, 0),
        (2000, 2000, 1000, 1),
        (3000, 2000, 1000, 2),
        (10000, 2000, 1000, 9),
        (10000, 400, 200, 49),
    ],
)
def test_count_windows(sample_count, length, step, expected):
    assert count_windows(sample_count, length, step) == expected


def test_count_windows_rejects_invalid_values():
    with pytest.raises(ValueError):
        count_windows(-1, 2000, 1000)
    with pytest.raises(ValueError):
        count_windows(2000, 0, 1000)


def test_contiguous_trials_keeps_repeated_segments_separate():
    labels = np.array([0, 1, 1, 0, 1, 1, 0, 2, 2])
    repetitions = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])

    trials = contiguous_trials(labels, repetitions)

    assert trials == [
        {
            "movement": 1,
            "repetition": 1,
            "segment_index": 1,
            "start_sample": 1,
            "stop_sample": 3,
            "sample_count": 2,
        },
        {
            "movement": 1,
            "repetition": 1,
            "segment_index": 2,
            "start_sample": 4,
            "stop_sample": 6,
            "sample_count": 2,
        },
        {
            "movement": 2,
            "repetition": 1,
            "segment_index": 1,
            "start_sample": 7,
            "stop_sample": 9,
            "sample_count": 2,
        },
    ]


def test_channel_statistics_detects_nonfinite_and_constant_channels():
    emg = np.column_stack(
        [
            np.ones(8),
            np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ]
    )

    rows = channel_statistics(
        emg,
        subject=1,
        exercise=1,
        file_name="test.mat",
        extreme_repeat_threshold=0.5,
    )

    assert rows[0]["constant_channel"] is True
    assert rows[0]["nonfinite_count"] == 0
    assert rows[1]["constant_channel"] is False
    assert rows[1]["nonfinite_count"] == 1


def test_contiguous_trials_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        contiguous_trials(np.array([1, 1]), np.array([1]))


def test_align_time_series_removes_one_trailing_sample():
    emg = np.arange(12, dtype=float).reshape(6, 2)
    vectors = {
        "stimulus": np.arange(6),
        "restimulus": np.arange(5),
        "repetition": np.arange(6),
        "rerepetition": np.arange(5),
    }

    aligned_emg, aligned_vectors, removed = align_time_series(
        emg, vectors, max_tail_mismatch=1
    )

    assert aligned_emg.shape == (5, 2)
    assert all(values.size == 5 for values in aligned_vectors.values())
    assert removed == {
        "emg": 1,
        "stimulus": 1,
        "restimulus": 0,
        "repetition": 1,
        "rerepetition": 0,
    }


def test_align_time_series_rejects_larger_mismatch():
    emg = np.zeros((8, 2))
    vectors = {"restimulus": np.zeros(6, dtype=int)}

    with pytest.raises(ValueError, match="length mismatch exceeds"):
        align_time_series(emg, vectors, max_tail_mismatch=1)
