#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute minimal agreement and gated metric summaries from local labels."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DB_PATH = EXPERIMENT_DIR / "qualitative_mechanics.sqlite"
POI_PATH = REPO_ROOT / "baseball_pitching" / "data" / "poi" / "poi_metrics.csv"
OUTPUT_PATH = EXPERIMENT_DIR / "label_analysis_summary.json"

AGREEMENT_THRESHOLD = 0.70
MIN_SHARED_TASKS = 5
MIN_COACHES = 2

METRICS = [
    "pitch_speed_mph",
    "max_cog_velo_x",
    "pelvis_rotation_fp",
    "rotation_hip_shoulder_separation_fp",
    "max_rotation_hip_shoulder_separation",
    "shoulder_horizontal_abduction_fp",
    "max_shoulder_horizontal_abduction",
    "max_torso_rotational_velo",
    "torso_rotation_fp",
]


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing local SQLite database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def read_poi_metrics() -> dict[str, dict[str, float]]:
    if not POI_PATH.exists():
        raise FileNotFoundError(f"Missing POI metrics: {POI_PATH}")
    with POI_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("poi_metrics.csv has no header.")
        required = {"session_pitch", *METRICS}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(f"poi_metrics.csv missing required columns: {sorted(missing)}")
        metrics_by_pitch: dict[str, dict[str, float]] = {}
        for row in reader:
            session_pitch = row["session_pitch"]
            metrics_by_pitch[session_pitch] = {metric: parse_float(row[metric], metric, session_pitch) for metric in METRICS}
    return metrics_by_pitch


def parse_float(value: str, metric: str, session_pitch: str) -> float:
    if value == "":
        raise RuntimeError(f"Missing {metric} for session_pitch={session_pitch}")
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {metric} for session_pitch={session_pitch}: {value!r}") from exc


def load_labels(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT coach_id, session_pitch, item_name, label_value, skipped
        FROM labels
        ORDER BY item_name, session_pitch, coach_id
        """
    ).fetchall()


def label_distribution(rows: list[sqlite3.Row]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = "<skipped>" if int(row["skipped"]) else row["label_value"]
        counts[row["coach_id"]][value] += 1
    return {coach_id: dict(counter) for coach_id, counter in sorted(counts.items())}


def item_agreement(item_rows: list[sqlite3.Row]) -> dict[str, object]:
    coaches = sorted({row["coach_id"] for row in item_rows})
    unclear_count = sum(1 for row in item_rows if row["label_value"] == "unclear")
    skipped_count = sum(1 for row in item_rows if int(row["skipped"]))
    by_pitch: dict[str, dict[str, str]] = defaultdict(dict)
    for row in item_rows:
        if int(row["skipped"]):
            continue
        by_pitch[row["session_pitch"]][row["coach_id"]] = row["label_value"]

    compared_pairs = 0
    exact_matches = 0
    shared_tasks = 0
    for coach_labels in by_pitch.values():
        pitch_coaches = sorted(coach_labels)
        if len(pitch_coaches) < 2:
            continue
        shared_tasks += 1
        for index, coach_a in enumerate(pitch_coaches):
            for coach_b in pitch_coaches[index + 1 :]:
                compared_pairs += 1
                if coach_labels[coach_a] == coach_labels[coach_b]:
                    exact_matches += 1

    exact_agreement_rate = None if compared_pairs == 0 else exact_matches / compared_pairs
    if len(coaches) < MIN_COACHES:
        gate_reason = "fewer_than_two_coaches"
    elif shared_tasks < MIN_SHARED_TASKS:
        gate_reason = "not_enough_shared_tasks"
    elif exact_agreement_rate is None or exact_agreement_rate < AGREEMENT_THRESHOLD:
        gate_reason = "below_agreement_threshold"
    else:
        gate_reason = "pass"

    return {
        "coach_count": len(coaches),
        "shared_tasks": shared_tasks,
        "compared_pairs": compared_pairs,
        "exact_matches": exact_matches,
        "exact_agreement_rate": exact_agreement_rate,
        "unclear_count": unclear_count,
        "skipped_count": skipped_count,
        "label_distribution_by_coach": label_distribution(item_rows),
        "pooled_analysis_enabled": gate_reason == "pass",
        "pooled_analysis_gate_reason": gate_reason,
    }


def pooled_metric_summary(item_rows: list[sqlite3.Row], metrics_by_pitch: dict[str, dict[str, float]]) -> dict[str, object]:
    grouped_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in item_rows:
        if int(row["skipped"]) or row["label_value"] == "unclear":
            continue
        session_pitch = row["session_pitch"]
        if session_pitch not in metrics_by_pitch:
            raise RuntimeError(f"Missing POI metrics for labeled session_pitch={session_pitch}")
        for metric in METRICS:
            grouped_values[row["label_value"]][metric].append(metrics_by_pitch[session_pitch][metric])

    summary: dict[str, object] = {}
    for label_value, metric_values in sorted(grouped_values.items()):
        summary[label_value] = {
            metric: {
                "n": len(values),
                "mean": mean(values),
            }
            for metric, values in sorted(metric_values.items())
        }
    return summary


def build_summary() -> dict[str, object]:
    metrics_by_pitch = read_poi_metrics()
    with connect() as conn:
        rows = load_labels(conn)
    if not rows:
        raise RuntimeError("No labels in local database.")

    rows_by_item: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        rows_by_item[row["item_name"]].append(row)

    item_summaries = {}
    for item_name, item_rows in sorted(rows_by_item.items()):
        agreement = item_agreement(item_rows)
        item_summary = {"agreement": agreement}
        if agreement["pooled_analysis_enabled"]:
            item_summary["pooled_metric_summary"] = pooled_metric_summary(item_rows, metrics_by_pitch)
        else:
            item_summary["pooled_metric_summary"] = None
        item_summaries[item_name] = item_summary

    return {
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "min_shared_tasks": MIN_SHARED_TASKS,
        "min_coaches": MIN_COACHES,
        "metric_columns": METRICS,
        "item_summaries": item_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help=f"Write {OUTPUT_PATH.name}.")
    args = parser.parse_args()

    summary = build_summary()
    item_count = len(summary["item_summaries"])
    enabled_count = sum(
        1
        for item in summary["item_summaries"].values()
        if item["agreement"]["pooled_analysis_enabled"]
    )
    print(f"items={item_count}")
    print(f"pooled_analysis_enabled_items={enabled_count}")
    for item_name, item in summary["item_summaries"].items():
        agreement = item["agreement"]
        print(
            f"{item_name}: coaches={agreement['coach_count']} "
            f"shared_tasks={agreement['shared_tasks']} "
            f"agreement={agreement['exact_agreement_rate']} "
            f"gate={agreement['pooled_analysis_gate_reason']}"
        )
    if args.write:
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote={OUTPUT_PATH}")
    else:
        print("write=false")


if __name__ == "__main__":
    main()

