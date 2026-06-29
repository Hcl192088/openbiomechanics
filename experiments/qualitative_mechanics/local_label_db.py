#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local SQLite prototype for qualitative mechanics web-app data flow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import secrets
import sqlite3
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DB_PATH = EXPERIMENT_DIR / "qualitative_mechanics.sqlite"
TASKS_PATH = EXPERIMENT_DIR / "label_tasks.csv"
LABELS_LONG_PATH = EXPERIMENT_DIR / "labels_long.csv"

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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return f"pbkdf2_sha256$200000${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    algorithm, iterations, salt, expected = stored_hash.split("$", 3)
    if algorithm != "pbkdf2_sha256":
        raise RuntimeError(f"Unsupported password hash algorithm: {algorithm}")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations))
    return secrets.compare_digest(digest.hex(), expected)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS labels;
        DROP TABLE IF EXISTS label_tasks;
        DROP TABLE IF EXISTS coaches;

        CREATE TABLE coaches (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE label_tasks (
            id TEXT PRIMARY KEY,
            session_pitch TEXT NOT NULL UNIQUE,
            display_order INTEGER NOT NULL UNIQUE,
            pitcher_id TEXT NOT NULL,
            p_throws TEXT NOT NULL,
            filename_new TEXT NOT NULL,
            c3d_path TEXT NOT NULL,
            active_label_fields TEXT NOT NULL,
            active INTEGER NOT NULL
        );

        CREATE TABLE labels (
            coach_id TEXT NOT NULL,
            session_pitch TEXT NOT NULL,
            item_name TEXT NOT NULL,
            label_value TEXT NOT NULL,
            view_used TEXT NOT NULL,
            playback_speed TEXT NOT NULL,
            skipped INTEGER NOT NULL,
            skip_reason TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (coach_id, session_pitch, item_name),
            FOREIGN KEY (coach_id) REFERENCES coaches(id),
            FOREIGN KEY (session_pitch) REFERENCES label_tasks(session_pitch)
        );
        """
    )


def seed_coaches(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO coaches (id, name, password_hash, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("1", "pilot_coach_1", hash_password("local-only-test-password"), 1, "2026-06-29T00:00:00+00:00"),
    )


def seed_tasks(conn: sqlite3.Connection) -> int:
    rows = read_csv(TASKS_PATH)
    if not rows:
        raise RuntimeError("label_tasks.csv has no rows.")
    for row in rows:
        conn.execute(
            """
            INSERT INTO label_tasks (
                id, session_pitch, display_order, pitcher_id, p_throws,
                filename_new, c3d_path, active_label_fields, active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["task_id"],
                row["session_pitch"],
                int(row["display_order"]),
                row["pitcher_id"],
                row["p_throws"],
                row["filename_new"],
                row["c3d_path"],
                row["active_label_fields"],
                1 if row["active"].strip().lower() == "true" else 0,
            ),
        )
    return len(rows)


def validate_label_row(row: dict[str, str]) -> None:
    item_name = row["item_name"].strip()
    label_value = row["label_value"].strip()
    skipped = row["skipped"].strip().lower() == "true"
    if item_name not in ACTIVE_LABEL_FIELDS:
        raise RuntimeError(f"Unsupported item_name: {item_name}")
    if not skipped and label_value not in FIELD_ALLOWED_VALUES[item_name]:
        raise RuntimeError(f"Invalid label value for {item_name}: {label_value}")


def seed_labels(conn: sqlite3.Connection) -> int:
    rows = read_csv(LABELS_LONG_PATH)
    if not rows:
        raise RuntimeError("labels_long.csv has no rows.")
    for row in rows:
        validate_label_row(row)
        conn.execute(
            """
            INSERT INTO labels (
                coach_id, session_pitch, item_name, label_value, view_used,
                playback_speed, skipped, skip_reason, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["coach_id"],
                row["session_pitch"],
                row["item_name"],
                row["label_value"],
                row["view_used"],
                row["playback_speed"],
                1 if row["skipped"].strip().lower() == "true" else 0,
                row["skip_reason"],
                row["notes"],
                row["created_at"],
            ),
        )
    return len(rows)


def pending_tasks(conn: sqlite3.Connection, coach_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT t.*
        FROM label_tasks t
        WHERE t.active = 1
          AND (
              SELECT COUNT(DISTINCT l.item_name)
              FROM labels l
              WHERE l.coach_id = ?
                AND l.session_pitch = t.session_pitch
          ) < ?
        ORDER BY t.display_order
        """,
        (coach_id, len(ACTIVE_LABEL_FIELDS)),
    ).fetchall()


def completed_task_count(conn: sqlite3.Connection, coach_id: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT session_pitch) AS n
        FROM labels
        WHERE coach_id = ?
        """,
        (coach_id,),
    ).fetchone()
    return int(row["n"])


def login(conn: sqlite3.Connection, name: str, password: str) -> str:
    row = conn.execute(
        "SELECT id, password_hash FROM coaches WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        raise RuntimeError("Invalid login.")
    return str(row["id"])


def save_label(
    conn: sqlite3.Connection,
    coach_id: str,
    session_pitch: str,
    item_name: str,
    label_value: str,
    created_at: str,
) -> None:
    if item_name not in FIELD_ALLOWED_VALUES:
        raise RuntimeError(f"Unsupported item_name: {item_name}")
    if label_value not in FIELD_ALLOWED_VALUES[item_name]:
        raise RuntimeError(f"Invalid label value for {item_name}: {label_value}")
    conn.execute(
        """
        INSERT INTO labels (
            coach_id, session_pitch, item_name, label_value, view_used,
            playback_speed, skipped, skip_reason, notes, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 0, '', '', ?)
        """,
        (coach_id, session_pitch, item_name, label_value, "test", "1", created_at),
    )


def init_db() -> None:
    with connect() as conn:
        init_schema(conn)
        seed_coaches(conn)
        task_count = seed_tasks(conn)
        label_count = seed_labels(conn)
        print(f"db={DB_PATH}")
        print(f"coaches=1")
        print(f"tasks={task_count}")
        print(f"labels={label_count}")


def smoke_test() -> None:
    with connect() as conn:
        coach_id = login(conn, "pilot_coach_1", "local-only-test-password")
        pending = pending_tasks(conn, coach_id)
        completed = completed_task_count(conn, coach_id)
        if completed != 30:
            raise RuntimeError(f"Expected 30 completed tasks for coach 1, got {completed}.")
        if len(pending) != 28:
            raise RuntimeError(f"Expected 28 pending tasks for coach 1, got {len(pending)}.")
        before_label_count = conn.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
        task = pending[0]
        save_label(
            conn,
            coach_id,
            task["session_pitch"],
            "direction",
            "good",
            "2026-06-29T00:00:00+00:00",
        )
        after_label_count = conn.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
        if int(after_label_count) != int(before_label_count) + 1:
            raise RuntimeError("Saving one label did not create exactly one row.")
        pending_after = pending_tasks(conn, coach_id)
        if len(pending_after) != len(pending):
            raise RuntimeError("Partially completed pitch-level task should remain pending.")
        try:
            save_label(
                conn,
                coach_id,
                task["session_pitch"],
                "direction",
                "bad",
                "2026-06-29T00:00:01+00:00",
            )
        except sqlite3.IntegrityError:
            duplicate_blocked = True
        else:
            duplicate_blocked = False
        if not duplicate_blocked:
            raise RuntimeError("Duplicate label insert was not blocked.")
        remaining_fields = [field for field in ACTIVE_LABEL_FIELDS if field != "direction"]
        for field in remaining_fields:
            value = sorted(FIELD_ALLOWED_VALUES[field])[0]
            save_label(
                conn,
                coach_id,
                task["session_pitch"],
                field,
                value,
                "2026-06-29T00:00:02+00:00",
            )
        pending_after_complete = pending_tasks(conn, coach_id)
        if len(pending_after_complete) != len(pending) - 1:
            raise RuntimeError("Completed pitch-level task still appears pending.")
        print(f"login_coach_id={coach_id}")
        print(f"completed_tasks_before={completed}")
        print(f"pending_tasks_before={len(pending)}")
        print(f"labels_before={before_label_count}")
        print(f"labels_after_one_save={after_label_count}")
        print(f"pending_tasks_after_one_save={len(pending_after)}")
        print(f"pending_tasks_after_complete={len(pending_after_complete)}")
        print("duplicate_insert=blocked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Rebuild and seed the local SQLite database.")
    parser.add_argument("--smoke-test", action="store_true", help="Run local login, pending, save, and duplicate checks.")
    args = parser.parse_args()
    if args.init:
        init_db()
    if args.smoke_test:
        smoke_test()
    if not args.init and not args.smoke_test:
        parser.error("Choose --init, --smoke-test, or both.")


if __name__ == "__main__":
    main()
