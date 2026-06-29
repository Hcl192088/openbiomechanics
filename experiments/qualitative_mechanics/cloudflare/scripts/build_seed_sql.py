#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Cloudflare D1 seed SQL from the qualitative mechanics CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


CLOUDFLARE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = CLOUDFLARE_DIR.parent
TASKS_PATH = EXPERIMENT_DIR / "label_tasks.csv"
LABELS_LONG_PATH = EXPERIMENT_DIR / "labels_long.csv"
POI_PATH = CLOUDFLARE_DIR.parents[2] / "baseball_pitching" / "data" / "poi" / "poi_metrics.csv"
OUTPUT_PATH = CLOUDFLARE_DIR / "seed.sql"
PBKDF2_ITERATIONS = 100_000
POI_METRICS = [
    "pitch_speed_mph",
    "max_rotation_hip_shoulder_separation",
    "rotation_hip_shoulder_separation_fp",
    "shoulder_horizontal_abduction_fp",
    "max_shoulder_horizontal_abduction",
    "max_torso_rotational_velo",
    "pelvis_rotation_fp",
    "cog_velo_pkh",
    "stride_length",
    "stride_angle",
    "max_rear_hip_flexion",
    "max_rear_hip_internal_rotation_velo",
    "rear_hip_transfer_pkh_fp",
    "rear_hip_generation_pkh_fp",
    "rear_hip_absorption_pkh_fp",
    "lead_hip_transfer_fp_br",
    "lead_hip_generation_fp_br",
    "lead_hip_absorption_fp_br",
    "lead_knee_extension_from_fp_to_br",
    "lead_knee_extension_angular_velo_fp",
    "lead_grf_x_max",
    "lead_grf_y_max",
    "lead_grf_z_max",
    "rear_grf_x_max",
    "rear_grf_y_max",
    "rear_grf_z_max",
    "max_cog_velo_x",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{path.name} has no rows.")
    return rows


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_real(value: str) -> str:
    if value == "":
        return "NULL"
    return str(float(value))


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def insert(table: str, columns: list[str], values: list[str]) -> str:
    cols = ", ".join(columns)
    vals = ", ".join(values)
    return f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({vals});"


def build_seed() -> str:
    task_rows = read_csv(TASKS_PATH)
    label_rows = read_csv(LABELS_LONG_PATH)
    poi_rows = read_csv(POI_PATH)
    seed_coaches = [
        ("1", "hcl", "0000", 1, 1, "cloudflare-preview-salt-hcl"),
        ("2", "ayung", "0000", 0, 1, "cloudflare-preview-salt-ayung"),
    ]
    lines = []
    for coach_id, name, password, is_admin, must_change_password, salt in seed_coaches:
        lines.append(
            "INSERT OR IGNORE INTO coaches (id, name, password_hash, is_admin, must_change_password, created_at) VALUES "
            + "("
            + ", ".join(
                [
                    sql_string(coach_id),
                    sql_string(name),
                    sql_string(hash_password(password, salt)),
                    str(is_admin),
                    str(must_change_password),
                    sql_string("2026-06-29T00:00:00+00:00"),
                ]
            )
            + ");"
        )
    for row in task_rows:
        lines.append(
            insert(
                "label_tasks",
                [
                    "id",
                    "session_pitch",
                    "display_order",
                    "pitcher_id",
                    "p_throws",
                    "filename_new",
                    "c3d_path",
                    "active_label_fields",
                    "active",
                ],
                [
                    sql_string(row["task_id"]),
                    sql_string(row["session_pitch"]),
                    row["display_order"],
                    sql_string(row["pitcher_id"]),
                    sql_string(row["p_throws"]),
                    sql_string(row["filename_new"]),
                    sql_string(row["c3d_path"]),
                    sql_string(row["active_label_fields"]),
                    "1" if row["active"].strip().lower() == "true" else "0",
                ],
            )
        )
    task_session_pitches = {row["session_pitch"] for row in task_rows}
    for row in poi_rows:
        if row["session_pitch"] not in task_session_pitches:
            continue
        lines.append(
            insert(
                "poi_metrics",
                ["session_pitch", *POI_METRICS],
                [sql_string(row["session_pitch"]), *[sql_real(row[metric]) for metric in POI_METRICS]],
            )
        )
    for row in label_rows:
        lines.append(
            insert(
                "labels",
                [
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
                ],
                [
                    sql_string(row["coach_id"]),
                    sql_string(row["session_pitch"]),
                    sql_string(row["item_name"]),
                    sql_string(row["label_value"]),
                    sql_string(row["view_used"]),
                    sql_string(row["playback_speed"]),
                    "1" if row["skipped"].strip().lower() == "true" else "0",
                    sql_string(row["skip_reason"]),
                    sql_string(row["notes"]),
                    sql_string(row["created_at"]),
                ],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help=f"Write {OUTPUT_PATH}.")
    args = parser.parse_args()
    sql = build_seed()
    print(f"tasks={len(read_csv(TASKS_PATH))}")
    print(f"labels={len(read_csv(LABELS_LONG_PATH))}")
    print(f"poi_metrics={sum(1 for row in read_csv(POI_PATH) if row['session_pitch'] in {task['session_pitch'] for task in read_csv(TASKS_PATH)})}")
    print(f"bytes={len(sql.encode('utf-8'))}")
    if args.write:
        OUTPUT_PATH.write_text(sql, encoding="utf-8")
        print(f"wrote={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
