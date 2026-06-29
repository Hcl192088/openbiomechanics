#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export pitch skeleton motion payloads to static JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qualitative_mechanics_experiment import load_motion_data


EXPERIMENT_DIR = Path(__file__).resolve().parent
TASKS_PATH = EXPERIMENT_DIR / "label_tasks.csv"
OUTPUT_DIR = EXPERIMENT_DIR / "web_motion"
MANIFEST_PATH = EXPERIMENT_DIR / "web_motion_manifest.csv"
DEFAULT_FRAME_STEP = 3


def read_tasks() -> list[dict[str, str]]:
    if not TASKS_PATH.exists():
        raise FileNotFoundError(f"Missing label tasks: {TASKS_PATH}")
    with TASKS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("label_tasks.csv has no rows.")
    required = {"session_pitch", "c3d_path", "active"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise RuntimeError(f"label_tasks.csv missing required columns: {sorted(missing)}")
    active_rows = [row for row in rows if row["active"].strip().lower() == "true"]
    if not active_rows:
        raise RuntimeError("label_tasks.csv has no active rows.")
    return active_rows


def motion_path(session_pitch: str) -> Path:
    safe_name = session_pitch.replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / f"{safe_name}.json"


def export_motion(tasks: list[dict[str, str]], frame_step: int, limit: int | None, write: bool) -> dict[str, object]:
    selected = tasks[:limit] if limit is not None else tasks
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = []
    total_bytes = 0
    for row in selected:
        session_pitch = row["session_pitch"]
        c3d_path = Path(row["c3d_path"])
        if not c3d_path.exists():
            raise FileNotFoundError(f"Missing C3D for {session_pitch}: {c3d_path}")
        payload = load_motion_data(c3d_path, frame_step)
        payload["metadata"]["session_pitch"] = session_pitch  # type: ignore[index]
        payload["metadata"]["frame_step"] = frame_step  # type: ignore[index]
        out_path = motion_path(session_pitch)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        total_bytes += len(encoded)
        if write:
            out_path.write_bytes(encoded)
        exported.append(
            {
                "session_pitch": session_pitch,
                "frames": len(payload["frames"]),  # type: ignore[arg-type]
                "bytes": len(encoded),
                "path": str(out_path.relative_to(EXPERIMENT_DIR)).replace("\\", "/"),
            }
        )

    return {
        "task_rows": len(tasks),
        "selected_rows": len(selected),
        "frame_step": frame_step,
        "total_bytes": total_bytes,
        "write": write,
        "exported": exported,
    }


def write_manifest(exported: list[dict[str, object]]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_pitch", "motion_path", "frames", "bytes"])
        writer.writeheader()
        for row in exported:
            writer.writerow(
                {
                    "session_pitch": row["session_pitch"],
                    "motion_path": row["path"],
                    "frames": row["frames"],
                    "bytes": row["bytes"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write", action="store_true", help=f"Write JSON files under {OUTPUT_DIR.name}.")
    args = parser.parse_args()
    if args.frame_step <= 0:
        raise RuntimeError("--frame-step must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be positive when provided.")

    summary = export_motion(read_tasks(), args.frame_step, args.limit, args.write)
    if args.write:
        write_manifest(summary["exported"])  # type: ignore[arg-type]
        summary["manifest_path"] = str(MANIFEST_PATH.relative_to(EXPERIMENT_DIR)).replace("\\", "/")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
