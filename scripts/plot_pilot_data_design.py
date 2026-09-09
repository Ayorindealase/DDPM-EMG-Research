#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch


ROLE_LABEL = {0: "S", 1: "V", 2: "T"}
ROLE_COLOR = ["#0072B2", "#CC79A7", "#E69F00"]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_box(ax, x, y, width, height, title, value, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="#17212B",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.64, value, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(x + width / 2, y + height * 0.28, title, ha="center", va="center", fontsize=8.5)


def build_figure(
    audit: dict[str, object],
    plan: dict[str, object],
    validation: dict[str, object],
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(11.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18])
    ax_flow = fig.add_subplot(grid[0, 0])
    ax_quality = fig.add_subplot(grid[0, 1])
    ax_roles = fig.add_subplot(grid[1, 0])
    ax_windows = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "NinaPro DB2 pilot data accounting and leakage-resistant design",
        fontsize=15,
        weight="bold",
    )

    ax_flow.set_title("(a) Audited data flow", loc="left", weight="bold")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    flow = [
        ("participants", f"{audit['subject_count']}", "#DDECF6"),
        ("MATLAB files", f"{audit['successfully_loaded_file_count']}", "#DDECF6"),
        ("active trial segments", f"{int(audit['trial_segment_count']):,}", "#E7F3EE"),
    ]
    for index, (title, value, color) in enumerate(flow):
        x = 0.03 + index * 0.325
        draw_box(ax_flow, x, 0.49, 0.25, 0.30, title, value, color)
        if index < len(flow) - 1:
            ax_flow.annotate(
                "",
                xy=(x + 0.315, 0.64),
                xytext=(x + 0.255, 0.64),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#17212B"},
            )
    ax_flow.text(
        0.03,
        0.26,
        f"Pilot: {len(plan['development_subjects'])} development participants",
        color="#0072B2",
        weight="bold",
    )
    ax_flow.text(
        0.03,
        0.13,
        f"Locked: {len(plan['confirmatory_subjects'])} confirmatory participants",
        color="#D55E00",
        weight="bold",
    )

    ax_quality.set_title("(b) Audit quality gate", loc="left", weight="bold")
    ax_quality.axis("off")
    quality_rows = [
        ("Files loaded / expected", f"{audit['successfully_loaded_file_count']} / {audit['expected_file_count']}"),
        ("Audit errors", str(audit["error_count"])),
        ("Non-finite EMG values", f"{int(audit['nonfinite_emg_values']):,}"),
        ("Constant channel records", str(audit["constant_channels"])),
        ("Potential saturation records", str(audit["potential_saturation_channels"])),
        ("Terminal-alignment warnings", str(audit["warning_count"])),
    ]
    for index, (label, value) in enumerate(quality_rows):
        y = 0.86 - index * 0.14
        ax_quality.text(0.02, y, label, transform=ax_quality.transAxes, va="center")
        ax_quality.text(
            0.96,
            y,
            value,
            transform=ax_quality.transAxes,
            va="center",
            ha="right",
            weight="bold",
        )
        ax_quality.plot([0.02, 0.96], [y - 0.065, y - 0.065], transform=ax_quality.transAxes, color="#D7DEE5", lw=0.8)
    ax_quality.text(
        0.02,
        0.015,
        "Warnings are retained for review; they are not automatic exclusions.",
        transform=ax_quality.transAxes,
        fontsize=8,
        color="#4B5563",
    )

    development_order = [int(value) for value in plan["development_subject_random_order"]]
    role_matrix = np.zeros((len(plan["folds"]), len(development_order)), dtype=int)
    for row_index, fold in enumerate(plan["folds"]):
        for subject in fold["source_validation_subjects"]:
            role_matrix[row_index, development_order.index(int(subject))] = 1
        for subject in fold["pseudo_target_subjects"]:
            role_matrix[row_index, development_order.index(int(subject))] = 2

    ax_roles.set_title("(c) Four outer pilot folds", loc="left", weight="bold")
    ax_roles.imshow(role_matrix, aspect="auto", cmap=ListedColormap(ROLE_COLOR), vmin=-0.5, vmax=2.5)
    ax_roles.set_xticks(np.arange(len(development_order)), [f"S{value}" for value in development_order], rotation=45, ha="right")
    ax_roles.set_yticks(np.arange(len(plan["folds"])), [f"Fold {fold['fold']}" for fold in plan["folds"]])
    for row in range(role_matrix.shape[0]):
        for column in range(role_matrix.shape[1]):
            code = int(role_matrix[row, column])
            ax_roles.text(column, row, ROLE_LABEL[code], ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax_roles.set_xlabel("Development participants in seeded random order")
    ax_roles.text(
        0.0,
        -0.30,
        "S = source training; V = source validation; T = pseudo-target",
        transform=ax_roles.transAxes,
        fontsize=8,
    )

    ax_windows.set_title("(d) Exercise-1 pilot windows", loc="left", weight="bold")
    type_counts = validation["window_type_counts"]
    labels = ["Generator\n1000 ms / 500 ms step", "Classifier\n200 ms / 100 ms step"]
    counts = [int(type_counts["generator"]), int(type_counts["classifier"])]
    bars = ax_windows.bar(labels, counts, color=["#009E73", "#56B4E9"], edgecolor="#17212B", linewidth=0.8, width=0.58)
    ax_windows.set_ylabel("Leakage-checked window count")
    ax_windows.spines[["top", "right"]].set_visible(False)
    ax_windows.grid(axis="y", color="#D7DEE5", linewidth=0.7)
    ax_windows.set_axisbelow(True)
    for bar, count in zip(bars, counts, strict=True):
        ax_windows.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", weight="bold")
    ax_windows.text(
        0.0,
        -0.18,
        "Split subjects before windowing; test repetitions 2 and 5 remain untouched.",
        transform=ax_windows.transAxes,
        fontsize=8,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure_title = "NinaPro DB2 pilot data accounting and leakage-resistant design"
    figure_description = (
        "Four-panel measured-data figure showing the complete DB2 audit, "
        "quality checks, the seeded four-fold participant design, and the "
        "Exercise-1 generator and classifier window counts."
    )
    fig.savefig(
        output_prefix.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Title": figure_title, "Description": figure_description},
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Title": figure_title, "Subject": figure_description},
    )
    fig.savefig(
        output_prefix.with_suffix(".png"),
        bbox_inches="tight",
        dpi=320,
        metadata={"Title": figure_title, "Description": figure_description},
    )
    plt.close(fig)

    caption = (
        "Data accounting and leakage-resistant design for the NinaPro DB2 "
        "proof-of-concept study. (a) All participants and files underwent the "
        "same metadata and signal-integrity audit before the seeded development/"
        "confirmation boundary was created. (b) Quality-gate outcomes retain "
        "alignment warnings for review rather than treating them as automatic "
        "exclusions. (c) The 16 development participants form four outer folds; "
        "each participant serves as a pseudo-target exactly once. (d) Exercise-1 "
        "windows were created only after participant and repetition roles were "
        "assigned. Generator and classifier windows never cross an active-trial "
        "boundary. Values are measured from the audited dataset; no model result "
        "is shown."
    )
    output_prefix.with_name(output_prefix.name + "_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the frozen DB2 pilot data design.")
    parser.add_argument("--audit-summary", type=Path, default=Path("outputs/data_audit/audit_summary.json"))
    parser.add_argument("--split-plan", type=Path, default=Path("data/splits/pilot_participants.json"))
    parser.add_argument("--split-validation", type=Path, default=Path("outputs/data_audit/split_validation.json"))
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/figures/data_audit/db2_pilot_data_design"))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    build_figure(
        load_json(args.audit_summary),
        load_json(args.split_plan),
        load_json(args.split_validation),
        args.output_prefix,
    )
    print(f"Saved pilot-design figure to {args.output_prefix}.[svg|pdf|png]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
