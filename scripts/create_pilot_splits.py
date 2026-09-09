#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emg_diffusion.data.splits import (
    PilotSplitConfig,
    build_pilot_split,
    confirmatory_lock,
    iter_window_manifest,
    sha256_file,
    validate_window_manifest,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the NinaPro DB2 pilot split and build a leakage-checked "
            "Exercise-1 window manifest from the completed data audit."
        )
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("outputs/data_audit")
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("outputs/data_audit/pilot_window_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--exercise", type=int, default=1)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    summary_path = args.audit_dir / "audit_summary.json"
    windows_path = args.audit_dir / "window_counts.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("audit_status") != "pass":
        raise RuntimeError("the dataset audit must pass before splits are created")
    if int(summary.get("subject_count", 0)) != 40:
        raise RuntimeError("the frozen split requires all 40 NinaPro DB2 participants")

    config = PilotSplitConfig(seed=args.seed, exercise=args.exercise)
    plan = build_pilot_split(config)
    plan["audit_provenance"] = {
        "audit_summary_sha256": sha256_file(summary_path),
        "window_counts_sha256": sha256_file(windows_path),
        "audited_file_count": int(summary["successfully_loaded_file_count"]),
        "audited_subject_count": int(summary["subject_count"]),
        "audit_error_count": int(summary["error_count"]),
        "audit_warning_count": int(summary["warning_count"]),
    }

    manifest_rows = list(iter_window_manifest(read_csv(windows_path), plan))
    validation = validate_window_manifest(manifest_rows, plan)
    validation["audit_summary_sha256"] = sha256_file(summary_path)
    validation["window_counts_sha256"] = sha256_file(windows_path)

    write_json(args.split_dir / "pilot_participants.json", plan)
    write_json(args.split_dir / "confirmatory_lock.json", confirmatory_lock(plan))
    write_json(
        args.split_dir / "calibration_allocations.json",
        {
            "schema_version": 1,
            "calibration_candidates": plan["repetition_policy"][
                "target_calibration_candidates"
            ],
            "test_repetitions": plan["repetition_policy"][
                "target_test_repetitions"
            ],
            "allocations": plan["calibration_allocations"],
        },
    )
    fold_rows = []
    for fold in plan["folds"]:
        for role_key in (
            "source_train_subjects",
            "source_validation_subjects",
            "pseudo_target_subjects",
        ):
            for subject in fold[role_key]:
                fold_rows.append(
                    {
                        "fold": fold["fold"],
                        "subject": subject,
                        "role": role_key.removesuffix("_subjects"),
                    }
                )
    write_csv(args.split_dir / "outer_folds.csv", fold_rows)
    write_csv(args.manifest_path, manifest_rows)
    write_json(args.audit_dir / "split_validation.json", validation)

    print(
        "Frozen pilot split created: "
        f"{len(plan['development_subjects'])} development, "
        f"{len(plan['confirmatory_subjects'])} confirmatory, "
        f"{validation['window_count']} Exercise-{args.exercise} windows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

