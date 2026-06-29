#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and convert wide qualitative labels into long web-app labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
LABELS_PATH = EXPERIMENT_DIR / "labels.csv"
TASKS_PATH = EXPERIMENT_DIR / "label_tasks.csv"
OUTPUT_PATH = EXPERIMENT_DIR / "labels_long.csv"

LABEL_FIELDS = [
    "hip_shoulder_separation",
    "lower_body_dominance",
    "direction",
    "shoulder_horizontal_abduction",
    "torso_velo_z",
    "hip_extension",
    "heel_connection",
    "drift",
]

FIELD_ALLOWED_VALUES = {
    "hip_shoulder_separation": {"good", "average", "bad", "unclear"},
    "lower_body_dominance": {"glute", "quad", "mixed", "unclear"},
    "direction": {"good", "bad", "unclear"},
    "shoulder_horizontal_abduction": {"good", "average", "bad", "unclear"},
    "torso_velo_z": {"fast", "slow", "unclear"},
    "hip_extension": {"good", "bad", "unclear"},
    "heel_connection": {"connected", "early_extension", "unclear"},
    "drift": {"good", "average", "bad", "unclear"},
}

OUTPUT_FIELDS = [
    "coach_id",
    "session_pitch",
    "item_name",
    "label_value",
    "view_used",
    "playback_speed",
    "skipped",
    "skip_reason",
    "notes",
    "created_at",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise RuntimeError(f"Invalid boolean value for skipped: {value!r}")


def validate_required_columns(rows: list[dict[str, str]], required: set[str], name: str) -> None:
    if not rows:
        raise RuntimeError(f"{name} has no rows.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise RuntimeError(f"{name} missing required columns: {sorted(missing)}")


def load_task_session_pitches() -> set[str]:
    task_rows = read_csv(TASKS_PATH)
    validate_required_columns(task_rows, {"session_pitch", "active"}, "label_tasks.csv")
    return {row["session_pitch"] for row in task_rows if row["active"].strip().lower() == "true"}


def convert_rows(label_rows: list[dict[str, str]], task_session_pitches: set[str]) -> list[dict[str, str]]:
    long_rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    invalid_values: list[str] = []
    missing_values: list[str] = []
    outside_tasks: list[str] = []

    for row_index, row in enumerate(label_rows, start=2):
        coach_id = row["rater_id"].strip()
        session_pitch = row["session_pitch"].strip()
        if session_pitch not in task_session_pitches:
            outside_tasks.append(f"row {row_index}: {session_pitch}")

        skipped = normalize_bool(row.get("skipped", ""))
        for field in LABEL_FIELDS:
            value = row.get(field, "").strip()
            key = (coach_id, session_pitch, field)
            if key in seen_keys:
                raise RuntimeError(f"Duplicate converted label key: {key}")
            seen_keys.add(key)

            if skipped:
                if value and value not in FIELD_ALLOWED_VALUES[field]:
                    invalid_values.append(f"row {row_index}: {field}={value!r}")
            else:
                if not value:
                    missing_values.append(f"row {row_index}: {field}")
                elif value not in FIELD_ALLOWED_VALUES[field]:
                    invalid_values.append(f"row {row_index}: {field}={value!r}")

            long_rows.append(
                {
                    "coach_id": coach_id,
                    "session_pitch": session_pitch,
                    "item_name": field,
                    "label_value": value,
                    "view_used": row.get("view_used", "").strip(),
                    "playback_speed": row.get("playback_speed", "").strip(),
                    "skipped": "true" if skipped else "false",
                    "skip_reason": row.get("skip_reason", "").strip(),
                    "notes": row.get("notes", "").strip(),
                    "created_at": row.get("saved_at_utc", "").strip(),
                }
            )

    if outside_tasks:
        raise RuntimeError("Label rows reference pitches outside active tasks: " + "; ".join(outside_tasks))
    if missing_values:
        raise RuntimeError("Missing label values: " + "; ".join(missing_values))
    if invalid_values:
        raise RuntimeError("Invalid label values: " + "; ".join(invalid_values))

    return long_rows


def write_rows(rows: list[dict[str, str]]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help=f"Write {OUTPUT_PATH.name}.")
    args = parser.parse_args()

    label_rows = read_csv(LABELS_PATH)
    required = {
        "saved_at_utc",
        "rater_id",
        "session_pitch",
        "view_used",
        "playback_speed",
        "skipped",
        "skip_reason",
        "notes",
        *LABEL_FIELDS,
    }
    validate_required_columns(label_rows, required, "labels.csv")

    task_session_pitches = load_task_session_pitches()
    long_rows = convert_rows(label_rows, task_session_pitches)
    skipped_rows = sum(1 for row in label_rows if normalize_bool(row.get("skipped", "")))

    print(f"source_rows={len(label_rows)}")
    print(f"unique_session_pitch={len({row['session_pitch'] for row in label_rows})}")
    print(f"active_task_session_pitch={len(task_session_pitches)}")
    print(f"label_fields={len(LABEL_FIELDS)}")
    print(f"long_rows={len(long_rows)}")
    print(f"skipped_source_rows={skipped_rows}")

    if args.write:
        write_rows(long_rows)
        print(f"wrote={OUTPUT_PATH}")
    else:
        print("write=false")


if __name__ == "__main__":
    main()

