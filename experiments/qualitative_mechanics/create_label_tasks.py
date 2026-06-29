#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the pitch-level task seed for the qualitative mechanics web app."""

from __future__ import annotations

import csv
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.csv"
TASKS_PATH = EXPERIMENT_DIR / "label_tasks.csv"

ACTIVE_LABEL_FIELDS = [
    "hip_shoulder_separation",
    "lower_body_dominance",
    "direction",
    "shoulder_horizontal_abduction",
    "torso_velo_z",
    "hip_extension",
    "heel_connection",
    "drift",
]

TASK_FIELDS = [
    "task_id",
    "display_order",
    "session_pitch",
    "pitcher_id",
    "p_throws",
    "filename_new",
    "c3d_path",
    "active_label_fields",
    "active",
]


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("manifest.csv has no rows.")

    required = {"order", "session_pitch", "pitcher_id", "p_throws", "filename_new", "c3d_path"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise RuntimeError(f"manifest.csv missing required columns: {sorted(missing)}")

    session_pitches = [row["session_pitch"] for row in rows]
    duplicate_pitches = sorted({pitch for pitch in session_pitches if session_pitches.count(pitch) > 1})
    if duplicate_pitches:
        raise RuntimeError(f"Duplicate session_pitch values in manifest.csv: {duplicate_pitches}")

    return rows


def build_tasks(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    tasks = []
    active_label_fields = ";".join(ACTIVE_LABEL_FIELDS)
    for row in manifest_rows:
        display_order = row["order"]
        tasks.append(
            {
                "task_id": f"pitch_{display_order}",
                "display_order": display_order,
                "session_pitch": row["session_pitch"],
                "pitcher_id": row["pitcher_id"],
                "p_throws": row["p_throws"],
                "filename_new": row["filename_new"],
                "c3d_path": row["c3d_path"],
                "active_label_fields": active_label_fields,
                "active": "true",
            }
        )
    return tasks


def write_tasks(tasks: list[dict[str, str]]) -> None:
    with TASKS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_FIELDS)
        writer.writeheader()
        writer.writerows(tasks)


def main() -> None:
    manifest_rows = read_manifest()
    tasks = build_tasks(manifest_rows)
    write_tasks(tasks)
    print(f"manifest_rows={len(manifest_rows)}")
    print(f"task_rows={len(tasks)}")
    print(f"active_label_fields={len(ACTIVE_LABEL_FIELDS)}")
    print(f"wrote={TASKS_PATH}")


if __name__ == "__main__":
    main()

