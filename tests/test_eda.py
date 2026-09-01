from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emg_diffusion.visualization.eda import (
    NinaProRecord,
    gesture_channel_rms,
    moving_rms,
    power_spectrum,
    principal_component_projection,
    repetition_rms_matrix,
    save_figure,
    select_trial,
    state_window_rms,
    teager_kaiser_energy,
    trial_feature_matrix,
)


def test_moving_rms_is_one_for_constant_unit_signal():
    result = moving_rms(np.ones(21), window_samples=5)

    assert np.allclose(result[2:-2], 1.0)


def test_teager_kaiser_energy_for_sinusoid():
    angular_frequency = 0.4
    samples = np.arange(200)
    signal = np.sin(angular_frequency * samples)

    energy = teager_kaiser_energy(signal)

    expected = np.sin(angular_frequency) ** 2
    assert np.nanmean(energy) == pytest.approx(expected, rel=1e-10)


def test_power_spectrum_finds_tone_frequency():
    sampling_rate_hz = 2000
    time = np.arange(4000) / sampling_rate_hz
    signal = np.sin(2 * np.pi * 120 * time)

    frequencies, density, features = power_spectrum(
        signal, sampling_rate_hz, nperseg=2000
    )

    peak_frequency = frequencies[np.argmax(density)]
    assert peak_frequency == pytest.approx(120.0, abs=1.0)
    assert features["mean_frequency_hz"] == pytest.approx(120.0, abs=1.0)
    assert features["median_frequency_hz"] == pytest.approx(119.5, abs=1.0)


def test_select_trial_and_repetition_rms_matrix():
    labels = np.array([0, 1, 1, 0, 1, 1, 0])
    repetitions = np.array([0, 1, 1, 0, 2, 2, 0])
    emg = np.column_stack(
        [
            np.array([0, 3, 4, 0, 6, 8, 0], dtype=float),
            np.array([0, 0, 0, 0, 5, 12, 0], dtype=float),
        ]
    )
    record = NinaProRecord(
        path=Path("test.mat"),
        subject=1,
        exercise=1,
        sampling_rate_hz=2000,
        emg=emg,
        labels=labels,
        repetitions=repetitions,
        label_source="refined",
        removed_samples={},
    )

    trial = select_trial(record, movement=1, repetition=2)
    repetition_ids, rms = repetition_rms_matrix(record, movement=1)

    assert (trial.start_sample, trial.stop_sample) == (4, 6)
    assert repetition_ids.tolist() == [1, 2]
    assert rms[0, 0] == pytest.approx(5 / np.sqrt(2))
    assert rms[1, 0] == pytest.approx(10 / np.sqrt(2))
    assert rms[1, 1] == pytest.approx(13 / np.sqrt(2))


def test_moving_rms_rejects_short_signal():
    with pytest.raises(ValueError):
        moving_rms(np.ones(3), window_samples=5)


def test_save_figure_embeds_svg_accessibility(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    svg_path, png_path = save_figure(
        figure,
        tmp_path,
        "test_figure",
        title="Test title",
        description="Test description",
    )
    svg = svg_path.read_text(encoding="utf-8")
    plt.close(figure)

    assert "<title>Test title</title>" in svg
    assert "<desc>Test description</desc>" in svg
    assert png_path.exists()


def test_state_window_rms_separates_pure_windows():
    record = NinaProRecord(
        path=Path("test.mat"),
        subject=1,
        exercise=1,
        sampling_rate_hz=10,
        emg=np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [5.0, 6.0],
            ]
        ),
        labels=np.array([0, 0, 1, 1, 0, 1]),
        repetitions=np.array([0, 0, 1, 1, 0, 1]),
        label_source="refined",
        removed_samples={},
    )

    rest, active, excluded = state_window_rms(
        record, window_ms=200, step_ms=200
    )

    assert rest.shape == (1, 2)
    assert active.shape == (1, 2)
    assert excluded == 1
    assert rest[0].tolist() == pytest.approx([1.0, 2.0])
    assert active[0].tolist() == pytest.approx([3.0, 4.0])


def test_gesture_atlas_and_trial_features_have_expected_shape():
    record = NinaProRecord(
        path=Path("test.mat"),
        subject=1,
        exercise=1,
        sampling_rate_hz=10,
        emg=np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 1.0],
                [0.0, 0.0],
                [3.0, 4.0],
                [4.0, 3.0],
                [0.0, 0.0],
            ]
        ),
        labels=np.array([0, 1, 1, 0, 2, 2, 0]),
        repetitions=np.array([0, 1, 1, 0, 1, 1, 0]),
        label_source="refined",
        removed_samples={},
    )

    movements, mean_rms, standard_deviation = gesture_channel_rms(record)
    features, labels, repetitions, names = trial_feature_matrix(record)

    assert movements.tolist() == [1, 2]
    assert mean_rms.shape == (2, 2)
    assert standard_deviation.shape == (2, 2)
    assert features.shape == (2, 4)
    assert labels.tolist() == [1, 2]
    assert repetitions.tolist() == [1, 1]
    assert len(names) == 4


def test_principal_component_projection_standardizes_features():
    features = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 1.0],
            [3.0, 6.0, 1.0],
        ]
    )

    scores, explained_ratio, components = principal_component_projection(features)

    assert scores.shape == (4, 3)
    assert components.shape == (3, 3)
    assert explained_ratio.sum() == pytest.approx(1.0)
    assert explained_ratio[0] == pytest.approx(1.0)
