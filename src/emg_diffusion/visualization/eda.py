from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import spectrogram, welch

from emg_diffusion.data.audit import align_time_series, contiguous_trials


BLUE = "#0072B2"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#6B7280"
INK = "#17212B"


@dataclass(frozen=True)
class NinaProRecord:
    path: Path
    subject: int
    exercise: int
    sampling_rate_hz: int
    emg: np.ndarray
    labels: np.ndarray
    repetitions: np.ndarray
    label_source: str
    removed_samples: dict[str, int]

    @property
    def duration_seconds(self) -> float:
        return self.emg.shape[0] / self.sampling_rate_hz


@dataclass(frozen=True)
class TrialSelection:
    movement: int
    repetition: int
    segment_index: int
    start_sample: int
    stop_sample: int

    @property
    def sample_count(self) -> int:
        return self.stop_sample - self.start_sample


def load_ninapro_record(
    path: Path,
    *,
    sampling_rate_hz: int = 2000,
    label_source: str = "refined",
    max_tail_mismatch: int = 400,
) -> NinaProRecord:
    if label_source not in {"original", "refined"}:
        raise ValueError("label_source must be 'original' or 'refined'")

    variables = (
        "subject",
        "exercise",
        "emg",
        "stimulus",
        "restimulus",
        "repetition",
        "rerepetition",
    )
    contents = loadmat(path, variable_names=variables)
    missing = [name for name in variables if name not in contents]
    if missing:
        raise ValueError(f"missing variables: {', '.join(missing)}")

    emg = np.asarray(contents["emg"])
    if emg.ndim != 2:
        raise ValueError(f"EMG array must be two-dimensional, received {emg.shape}")

    vectors = {
        "stimulus": _integer_vector(contents["stimulus"], "stimulus"),
        "restimulus": _integer_vector(contents["restimulus"], "restimulus"),
        "repetition": _integer_vector(contents["repetition"], "repetition"),
        "rerepetition": _integer_vector(
            contents["rerepetition"], "rerepetition"
        ),
    }
    emg, vectors, removed_samples = align_time_series(
        emg, vectors, max_tail_mismatch
    )

    if label_source == "refined":
        labels = vectors["restimulus"]
        repetitions = vectors["rerepetition"]
    else:
        labels = vectors["stimulus"]
        repetitions = vectors["repetition"]

    return NinaProRecord(
        path=path,
        subject=_scalar_integer(contents["subject"], "subject"),
        exercise=_scalar_integer(contents["exercise"], "exercise"),
        sampling_rate_hz=sampling_rate_hz,
        emg=emg,
        labels=labels,
        repetitions=repetitions,
        label_source=label_source,
        removed_samples=removed_samples,
    )


def select_trial(
    record: NinaProRecord,
    *,
    movement: int,
    repetition: int,
    segment_index: int = 1,
) -> TrialSelection:
    candidates = [
        segment
        for segment in contiguous_trials(record.labels, record.repetitions)
        if segment["movement"] == movement
        and segment["repetition"] == repetition
        and segment["segment_index"] == segment_index
    ]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one matching trial segment; "
            f"found {len(candidates)}"
        )
    segment = candidates[0]
    return TrialSelection(
        movement=movement,
        repetition=repetition,
        segment_index=segment_index,
        start_sample=segment["start_sample"],
        stop_sample=segment["stop_sample"],
    )


def moving_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if values.size < window_samples:
        raise ValueError("signal must be at least as long as the RMS window")
    kernel = np.full(window_samples, 1.0 / window_samples)
    return np.sqrt(np.convolve(np.square(values), kernel, mode="same"))


def teager_kaiser_energy(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    energy = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size >= 3:
        energy[1:-1] = (
            np.square(values[1:-1]) - values[:-2] * values[2:]
        )
    return energy


def power_spectrum(
    signal: np.ndarray,
    sampling_rate_hz: int,
    *,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("at least two samples are required")
    segment_length = min(nperseg, values.size)
    frequencies, density = welch(
        values,
        fs=sampling_rate_hz,
        nperseg=segment_length,
        noverlap=segment_length // 2,
        detrend="constant",
    )
    total_power = float(np.trapezoid(density, frequencies))
    if total_power <= 0.0:
        mean_frequency = float("nan")
        median_frequency = float("nan")
    else:
        mean_frequency = float(
            np.trapezoid(frequencies * density, frequencies) / total_power
        )
        cumulative = np.concatenate(
            ([0.0], np.cumsum((density[1:] + density[:-1]) * np.diff(frequencies) / 2))
        )
        median_frequency = float(
            np.interp(total_power / 2.0, cumulative, frequencies)
        )
    return frequencies, density, {
        "total_power": total_power,
        "mean_frequency_hz": mean_frequency,
        "median_frequency_hz": median_frequency,
    }


def repetition_rms_matrix(
    record: NinaProRecord,
    movement: int,
) -> tuple[np.ndarray, np.ndarray]:
    repetitions = sorted(
        int(value)
        for value in np.unique(record.repetitions[record.labels == movement])
        if value > 0
    )
    rows = []
    for repetition in repetitions:
        trial = select_trial(
            record, movement=movement, repetition=repetition
        )
        segment = record.emg[trial.start_sample : trial.stop_sample]
        rows.append(np.sqrt(np.mean(np.square(segment), axis=0)))
    return np.asarray(repetitions), np.vstack(rows)


def state_window_rms(
    record: NinaProRecord,
    *,
    window_ms: float = 200.0,
    step_ms: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    window_samples = int(round(window_ms * record.sampling_rate_hz / 1000.0))
    step_samples = int(round(step_ms * record.sampling_rate_hz / 1000.0))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window and step durations must be positive")

    rest_rows = []
    active_rows = []
    excluded_windows = 0
    last_start = record.emg.shape[0] - window_samples
    for start in range(0, last_start + 1, step_samples):
        stop = start + window_samples
        window_labels = record.labels[start:stop]
        if np.all(window_labels == 0):
            destination = rest_rows
        elif window_labels[0] > 0 and np.all(window_labels == window_labels[0]):
            destination = active_rows
        else:
            excluded_windows += 1
            continue
        window = np.asarray(record.emg[start:stop], dtype=np.float64)
        destination.append(np.sqrt(np.mean(np.square(window), axis=0)))

    channel_count = record.emg.shape[1]
    rest = (
        np.vstack(rest_rows)
        if rest_rows
        else np.empty((0, channel_count), dtype=np.float64)
    )
    active = (
        np.vstack(active_rows)
        if active_rows
        else np.empty((0, channel_count), dtype=np.float64)
    )
    return rest, active, excluded_windows


def gesture_channel_rms(
    record: NinaProRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = {}
    for segment in contiguous_trials(record.labels, record.repetitions):
        start = segment["start_sample"]
        stop = segment["stop_sample"]
        signal = np.asarray(record.emg[start:stop], dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(signal), axis=0))
        grouped.setdefault(segment["movement"], []).append(rms)

    movements = np.asarray(sorted(grouped), dtype=np.int32)
    mean_rms = np.vstack(
        [np.mean(np.vstack(grouped[movement]), axis=0) for movement in movements]
    )
    standard_deviation = np.vstack(
        [np.std(np.vstack(grouped[movement]), axis=0) for movement in movements]
    )
    return movements, mean_rms, standard_deviation


def trial_feature_matrix(
    record: NinaProRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    feature_rows = []
    movement_labels = []
    repetition_labels = []
    for segment in contiguous_trials(record.labels, record.repetitions):
        start = segment["start_sample"]
        stop = segment["stop_sample"]
        signal = np.asarray(record.emg[start:stop], dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(signal), axis=0))
        mean_waveform_length = np.mean(np.abs(np.diff(signal, axis=0)), axis=0)
        feature_rows.append(
            np.concatenate(
                (
                    np.log10(np.maximum(rms, np.finfo(float).tiny)),
                    np.log10(
                        np.maximum(
                            mean_waveform_length, np.finfo(float).tiny
                        )
                    ),
                )
            )
        )
        movement_labels.append(segment["movement"])
        repetition_labels.append(segment["repetition"])

    channel_count = record.emg.shape[1]
    names = tuple(
        [f"log_rms_ch{channel}" for channel in range(1, channel_count + 1)]
        + [
            f"log_mean_wl_ch{channel}"
            for channel in range(1, channel_count + 1)
        ]
    )
    return (
        np.vstack(feature_rows),
        np.asarray(movement_labels, dtype=np.int32),
        np.asarray(repetition_labels, dtype=np.int32),
        names,
    )


def principal_component_projection(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("features must contain at least two observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")

    mean = np.mean(values, axis=0)
    standard_deviation = np.std(values, axis=0, ddof=0)
    standard_deviation[standard_deviation == 0.0] = 1.0
    standardized = (values - mean) / standard_deviation
    _, singular_values, components = np.linalg.svd(
        standardized, full_matrices=False
    )
    scores = standardized @ components.T
    component_variance = np.square(singular_values) / (values.shape[0] - 1)
    explained_ratio = component_variance / np.sum(component_variance)
    return scores, explained_ratio, components


def movement_rms_profile(
    record: NinaProRecord,
    movement: int,
) -> np.ndarray:
    _, rms = repetition_rms_matrix(record, movement)
    return np.mean(rms, axis=0)


def plot_record_overview(
    record: NinaProRecord,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float = 12.0,
) -> plt.Figure:
    start = int(round(start_seconds * record.sampling_rate_hz))
    stop = min(
        record.emg.shape[0],
        start + int(round(duration_seconds * record.sampling_rate_hz)),
    )
    if stop <= start:
        raise ValueError("the requested overview interval is empty")

    segment = record.emg[start:stop]
    time = np.arange(start, stop) / record.sampling_rate_hz
    display, offsets = _stacked_display(segment)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 1]},
        constrained_layout=True,
    )
    axes[0].plot(time, display, color=BLUE, linewidth=0.55)
    axes[0].set_yticks(offsets)
    axes[0].set_yticklabels([f"Ch {index}" for index in range(1, 13)])
    axes[0].set_ylabel("Per-channel robust scale\n(vertically offset)")
    axes[0].set_title(
        f"NinaPro DB2 S{record.subject}, E{record.exercise}: multichannel overview"
    )
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].step(
        time,
        record.labels[start:stop],
        where="post",
        color=VERMILLION,
        linewidth=1.2,
    )
    axes[1].fill_between(
        time,
        0,
        record.labels[start:stop],
        step="post",
        color=VERMILLION,
        alpha=0.15,
    )
    axes[1].set_ylabel("Gesture\nlabel")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(axis="x", alpha=0.25)
    return figure


def plot_trial_channels(
    record: NinaProRecord,
    trial: TrialSelection,
) -> plt.Figure:
    segment = record.emg[trial.start_sample : trial.stop_sample]
    time = np.arange(segment.shape[0]) / record.sampling_rate_hz
    display, offsets = _stacked_display(segment)

    figure, axis = plt.subplots(figsize=(13, 6.6), constrained_layout=True)
    axis.plot(time, display, color=BLUE, linewidth=0.6)
    axis.set_yticks(offsets)
    axis.set_yticklabels([f"Ch {index}" for index in range(1, 13)])
    axis.set_xlabel("Time from refined trial onset (s)")
    axis.set_ylabel("Per-channel robust scale\n(vertically offset)")
    axis.set_title(
        f"Gesture {trial.movement}, repetition {trial.repetition}: "
        "channel-specific activation"
    )
    axis.grid(axis="x", alpha=0.25)
    return figure


def plot_channel_structure(
    record: NinaProRecord,
    trial: TrialSelection,
) -> plt.Figure:
    segment = np.asarray(
        record.emg[trial.start_sample : trial.stop_sample], dtype=np.float64
    )
    rms = np.sqrt(np.mean(np.square(segment), axis=0))
    correlation = np.corrcoef(segment, rowvar=False)

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True
    )
    channels = np.arange(1, segment.shape[1] + 1)
    axes[0].bar(channels, rms, color=GREEN, edgecolor=INK, linewidth=0.6)
    axes[0].set_xticks(channels)
    axes[0].set_xlabel("sEMG channel")
    axes[0].set_ylabel("RMS amplitude (recorded units)")
    axes[0].set_title("(a) Activation strength differs by electrode")
    axes[0].grid(axis="y", alpha=0.25)

    image = axes[1].imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_xticks(np.arange(12), labels=np.arange(1, 13))
    axes[1].set_yticks(np.arange(12), labels=np.arange(1, 13))
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Channel")
    axes[1].set_title("(b) Zero-lag cross-channel correlation")
    figure.colorbar(image, ax=axes[1], label="Pearson correlation")
    return figure


def plot_spectral_view(
    record: NinaProRecord,
    trial: TrialSelection,
    *,
    channel: int,
) -> tuple[plt.Figure, dict[str, float]]:
    if not 1 <= channel <= record.emg.shape[1]:
        raise ValueError("channel is one-based and must exist in the record")
    signal = record.emg[trial.start_sample : trial.stop_sample, channel - 1]
    frequencies, density, features = power_spectrum(
        signal, record.sampling_rate_hz
    )
    spec_frequency, spec_time, spec_power = spectrogram(
        signal,
        fs=record.sampling_rate_hz,
        nperseg=min(256, signal.size),
        noverlap=min(192, max(signal.size - 1, 0)),
        detrend="constant",
        scaling="density",
        mode="psd",
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 4.8), constrained_layout=True
    )
    valid = frequencies <= 500
    axes[0].semilogy(
        frequencies[valid],
        np.maximum(density[valid], np.finfo(float).tiny),
        color=GREEN,
        linewidth=1.4,
    )
    axes[0].axvline(
        features["median_frequency_hz"],
        color=PURPLE,
        linestyle="--",
        label=f"Median: {features['median_frequency_hz']:.1f} Hz",
    )
    axes[0].axvline(
        features["mean_frequency_hz"],
        color=ORANGE,
        linestyle=":",
        label=f"Mean: {features['mean_frequency_hz']:.1f} Hz",
    )
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power spectral density")
    axes[0].set_title(f"(a) Welch spectrum, channel {channel}")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    frequency_mask = spec_frequency <= 500
    image = axes[1].pcolormesh(
        spec_time,
        spec_frequency[frequency_mask],
        10
        * np.log10(
            np.maximum(
                spec_power[frequency_mask], np.finfo(float).tiny
            )
        ),
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("Time from trial onset (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"(b) Spectrogram, channel {channel}")
    figure.colorbar(image, ax=axes[1], label="PSD (dB/Hz)")
    return figure, features


def plot_activation_dynamics(
    record: NinaProRecord,
    trial: TrialSelection,
    *,
    channel: int,
    envelope_window_ms: float = 50.0,
) -> plt.Figure:
    signal = np.asarray(
        record.emg[trial.start_sample : trial.stop_sample, channel - 1],
        dtype=np.float64,
    )
    window_samples = int(
        round(envelope_window_ms * record.sampling_rate_hz / 1000.0)
    )
    envelope = moving_rms(signal, window_samples)
    energy = teager_kaiser_energy(signal)
    time = np.arange(signal.size) / record.sampling_rate_hz

    raw_display = _robust_normalise(signal - np.median(signal))
    envelope_display = _positive_normalise(envelope)
    energy_display = _robust_normalise(np.nan_to_num(energy, nan=0.0))

    figure, axes = plt.subplots(
        3, 1, figsize=(13, 7.2), sharex=True, constrained_layout=True
    )
    axes[0].plot(time, raw_display, color=BLUE, linewidth=0.6)
    axes[0].set_ylabel("Scaled raw")
    axes[0].set_title(f"Channel {channel}: three views of activation dynamics")
    axes[1].plot(time, envelope_display, color=ORANGE, linewidth=1.2)
    axes[1].fill_between(time, 0, envelope_display, color=ORANGE, alpha=0.2)
    axes[1].set_ylabel(f"{envelope_window_ms:g} ms\nRMS envelope")
    axes[2].plot(time, energy_display, color=PURPLE, linewidth=0.7)
    axes[2].set_ylabel("Scaled TKEO")
    axes[2].set_xlabel("Time from trial onset (s)")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    return figure


def plot_repetition_variability(
    record: NinaProRecord,
    *,
    movement: int,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    repetitions, rms = repetition_rms_matrix(record, movement)
    channel_median = np.median(rms, axis=0)
    channel_median[channel_median == 0.0] = 1.0
    relative_rms = rms / channel_median

    figure, axis = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    image = axis.imshow(
        relative_rms,
        cmap="cividis",
        aspect="auto",
        vmin=0.5,
        vmax=1.5,
    )
    axis.set_xticks(np.arange(12), labels=np.arange(1, 13))
    axis.set_yticks(np.arange(repetitions.size), labels=repetitions)
    axis.set_xlabel("sEMG channel")
    axis.set_ylabel("Repetition")
    axis.set_title(
        f"Gesture {movement}: RMS variability across six repetitions"
    )
    figure.colorbar(
        image,
        ax=axis,
        label="RMS / channel-specific median RMS",
    )
    return figure, repetitions, rms


def plot_rest_active_rms(
    rest_rms: np.ndarray,
    active_rms: np.ndarray,
    *,
    window_ms: float,
) -> plt.Figure:
    if rest_rms.ndim != 2 or active_rms.ndim != 2:
        raise ValueError("rest and active RMS arrays must be two-dimensional")
    if rest_rms.shape[1] != active_rms.shape[1]:
        raise ValueError("rest and active RMS arrays must have equal channel counts")

    channel_count = rest_rms.shape[1]
    channels = np.arange(1, channel_count + 1)
    figure, axis = plt.subplots(figsize=(13, 5.4), constrained_layout=True)
    rest_plot = axis.boxplot(
        [rest_rms[:, index] for index in range(channel_count)],
        positions=channels - 0.18,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    active_plot = axis.boxplot(
        [active_rms[:, index] for index in range(channel_count)],
        positions=channels + 0.18,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    _style_boxplot(rest_plot, BLUE)
    _style_boxplot(active_plot, ORANGE)
    axis.plot([], [], color=BLUE, linewidth=8, label="Rest-only windows")
    axis.plot([], [], color=ORANGE, linewidth=8, label="Single-gesture windows")
    axis.set_yscale("log")
    axis.set_xticks(channels)
    axis.set_xlabel("sEMG channel")
    axis.set_ylabel("Window RMS amplitude (recorded units, log scale)")
    axis.set_title(
        f"Rest versus active amplitude in transition-free {window_ms:g} ms windows"
    )
    axis.legend(frameon=False, ncols=2)
    axis.grid(axis="y", alpha=0.25)
    return figure


def plot_gesture_channel_atlas(
    movements: np.ndarray,
    mean_rms: np.ndarray,
) -> plt.Figure:
    values = np.asarray(mean_rms, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != movements.size:
        raise ValueError("movement labels and mean RMS rows must agree")
    relative = values / np.maximum(
        np.max(values, axis=1, keepdims=True), np.finfo(float).tiny
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 7.2), constrained_layout=True
    )
    absolute_image = axes[0].imshow(
        np.log10(np.maximum(values, np.finfo(float).tiny)),
        cmap="magma",
        aspect="auto",
    )
    axes[0].set_title("(a) Mean RMS magnitude")
    axes[0].set_xlabel("sEMG channel")
    axes[0].set_ylabel("Gesture label")
    figure.colorbar(
        absolute_image,
        ax=axes[0],
        label="log10 RMS (recorded units)",
    )

    relative_image = axes[1].imshow(
        relative,
        cmap="cividis",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("(b) Within-gesture channel profile")
    axes[1].set_xlabel("sEMG channel")
    axes[1].set_ylabel("Gesture label")
    figure.colorbar(
        relative_image,
        ax=axes[1],
        label="RMS / maximum channel RMS for that gesture",
    )
    for axis in axes:
        axis.set_xticks(np.arange(values.shape[1]), labels=np.arange(1, values.shape[1] + 1))
        axis.set_yticks(np.arange(movements.size), labels=movements)
    return figure


def plot_trial_pca(
    scores: np.ndarray,
    explained_ratio: np.ndarray,
    movement_labels: np.ndarray,
) -> plt.Figure:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("PCA scores must contain at least two components")
    if scores.shape[0] != movement_labels.size:
        raise ValueError("PCA observations and labels must agree")

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 5.4), constrained_layout=True
    )
    movement_values = np.unique(movement_labels)
    color_map = plt.get_cmap("turbo")
    colors = color_map(
        np.linspace(0.05, 0.95, movement_values.size)
    )
    for movement, color in zip(movement_values, colors, strict=True):
        mask = movement_labels == movement
        axes[0].scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=28,
            color=color,
            alpha=0.65,
            edgecolor=INK,
            linewidth=0.25,
        )
        centroid = np.mean(scores[mask, :2], axis=0)
        axes[0].text(
            centroid[0],
            centroid[1],
            str(int(movement)),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.18", "fc": "white", "alpha": 0.75},
        )
    axes[0].set_xlabel(f"PC1 ({100 * explained_ratio[0]:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({100 * explained_ratio[1]:.1f}% variance)")
    axes[0].set_title("(a) Trial features projected into two dimensions")
    axes[0].grid(alpha=0.2)

    component_count = min(10, explained_ratio.size)
    components = np.arange(1, component_count + 1)
    axes[1].bar(
        components,
        100 * explained_ratio[:component_count],
        color=PURPLE,
        edgecolor=INK,
        linewidth=0.5,
    )
    axes[1].plot(
        components,
        100 * np.cumsum(explained_ratio[:component_count]),
        color=ORANGE,
        marker="o",
        label="Cumulative variance",
    )
    axes[1].set_xticks(components)
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("Explained variance (%)")
    axes[1].set_title("(b) Information retained by each component")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    return figure


def plot_subject_channel_profiles(
    subject_ids: np.ndarray,
    profiles: np.ndarray,
    *,
    movement: int,
) -> plt.Figure:
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != subject_ids.size:
        raise ValueError("subject IDs and profile rows must agree")
    relative = values / np.maximum(
        np.max(values, axis=1, keepdims=True), np.finfo(float).tiny
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 4.8), constrained_layout=True
    )
    absolute_image = axes[0].imshow(
        np.log10(np.maximum(values, np.finfo(float).tiny)),
        cmap="magma",
        aspect="auto",
    )
    axes[0].set_title("(a) Absolute mean RMS")
    axes[0].set_xlabel("sEMG channel")
    axes[0].set_ylabel("DB2 subject")
    figure.colorbar(
        absolute_image,
        ax=axes[0],
        label="log10 RMS (recorded units)",
    )

    relative_image = axes[1].imshow(
        relative,
        cmap="cividis",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("(b) Within-subject channel profile")
    axes[1].set_xlabel("sEMG channel")
    axes[1].set_ylabel("DB2 subject")
    figure.colorbar(
        relative_image,
        ax=axes[1],
        label="RMS / subject-specific maximum channel RMS",
    )
    for axis in axes:
        axis.set_xticks(np.arange(values.shape[1]), labels=np.arange(1, values.shape[1] + 1))
        axis.set_yticks(np.arange(subject_ids.size), labels=subject_ids)
    figure.suptitle(
        f"Gesture {movement}: fixed-subset illustration of cross-subject variation"
    )
    return figure


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    dpi: int = 300,
    title: str | None = None,
    description: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    metadata = {}
    if title is not None:
        metadata["Title"] = title
    if description is not None:
        metadata["Description"] = description
    figure.savefig(svg_path, bbox_inches="tight", metadata=metadata)
    _embed_svg_accessibility(
        svg_path,
        title or stem.replace("_", " ").title(),
        description or "Data-linked exploratory sEMG figure.",
    )
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return svg_path, png_path


def _integer_vector(value: np.ndarray, variable: str) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            raise ValueError(f"{variable} must contain finite integer labels")
    return array.astype(np.int32, copy=False)


def _scalar_integer(value: np.ndarray, variable: str) -> int:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{variable} must contain one value")
    return int(array[0])


def _stacked_display(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.percentile(np.abs(segment), 99.0, axis=0)
    scale[scale == 0.0] = 1.0
    offsets = np.arange(segment.shape[1] - 1, -1, -1) * 3.0
    return segment / scale + offsets, offsets


def _robust_normalise(values: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(np.abs(values), 99.0))
    if scale == 0.0:
        return np.zeros_like(values)
    return values / scale


def _positive_normalise(values: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(values, 99.0))
    if scale == 0.0:
        return np.zeros_like(values)
    return values / scale


def _style_boxplot(boxplot: dict[str, list], color: str) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.65)
        box.set_edgecolor(INK)
    for median in boxplot["medians"]:
        median.set_color(INK)
        median.set_linewidth(1.2)
    for element in ("whiskers", "caps"):
        for artist in boxplot[element]:
            artist.set_color(INK)


def _embed_svg_accessibility(path: Path, title: str, description: str) -> None:
    svg = path.read_text(encoding="utf-8")
    svg_start = svg.find("<svg")
    tag_end = svg.find(">", svg_start)
    if svg_start < 0 or tag_end < 0:
        raise ValueError(f"could not locate the root SVG element in {path}")
    accessible_text = (
        f"\n <title>{escape(title)}</title>"
        f"\n <desc>{escape(description)}</desc>"
    )
    path.write_text(
        svg[: tag_end + 1] + accessible_text + svg[tag_end + 1 :],
        encoding="utf-8",
    )
