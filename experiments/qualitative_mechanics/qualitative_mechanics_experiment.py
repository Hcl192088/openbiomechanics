#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local qualitative pitching-mechanics labeling experiment.

This tool serves a browser UI for labeling C3D skeleton motion clips. It writes
only experiment artifacts in this folder: manifest.csv and labels.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT / "baseball_pitching"
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.csv"
LABELS_PATH = EXPERIMENT_DIR / "labels.csv"

SKELETON_CONNECTIONS = [
    ("C7", "CLAV"), ("CLAV", "STRN"), ("STRN", "T10"),
    ("T10", "LASI"), ("T10", "RASI"), ("LASI", "RASI"),
    ("LASI", "LPSI"), ("RASI", "RPSI"), ("LPSI", "RPSI"),
    ("C7", "RBAK"), ("LFHD", "RFHD"), ("LBHD", "RBHD"),
    ("LFHD", "LBHD"), ("RFHD", "RBHD"),
    ("CLAV", "RSHO"), ("RSHO", "LSHO"), ("RSHO", "RUPA"), ("RUPA", "RELB"),
    ("RELB", "RMELB"), ("RMELB", "RFRM"), ("RFRM", "RWRA"),
    ("RFRM", "RWRB"), ("RWRA", "RWRB"), ("RWRA", "RFIN"),
    ("CLAV", "LSHO"), ("LSHO", "LUPA"), ("LUPA", "LELB"),
    ("LELB", "LMELB"), ("LMELB", "LFRM"), ("LFRM", "LWRA"),
    ("LFRM", "LWRB"), ("LWRA", "LWRB"), ("LWRA", "LFIN"),
    ("RASI", "RTHI"), ("RTHI", "RKNE"), ("RKNE", "RMKNE"),
    ("RMKNE", "RTIB"), ("RTIB", "RANK"), ("RANK", "RMANK"),
    ("RMANK", "RHEE"), ("RANK", "RTOE"), ("RHEE", "RTOE"),
    ("LASI", "LTHI"), ("LTHI", "LKNE"), ("LKNE", "LMKNE"),
    ("LMKNE", "LTIB"), ("LTIB", "LANK"), ("LANK", "LMANK"),
    ("LMANK", "LHEE"), ("LANK", "LTOE"), ("LHEE", "LTOE"),
]

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

FIELD_VALUE_ORDER = {
    "hip_shoulder_separation": ["good", "average", "bad", "unclear"],
    "lower_body_dominance": ["glute", "mixed", "quad", "unclear"],
    "direction": ["good", "bad", "unclear"],
    "shoulder_horizontal_abduction": ["good", "average", "bad", "unclear"],
    "torso_velo_z": ["fast", "slow", "unclear"],
    "hip_extension": ["good", "bad", "unclear"],
    "heel_connection": ["connected", "early_extension", "unclear"],
    "drift": ["good", "average", "bad", "unclear"],
}

PILOT_FIELD_METRICS = {
    "hip_shoulder_separation": [
        "pitch_speed_mph",
        "max_rotation_hip_shoulder_separation",
        "rotation_hip_shoulder_separation_fp",
    ],
    "shoulder_horizontal_abduction": [
        "pitch_speed_mph",
        "shoulder_horizontal_abduction_fp",
        "max_shoulder_horizontal_abduction",
    ],
    "torso_velo_z": ["pitch_speed_mph", "max_torso_rotational_velo"],
    "hip_extension": [
        "pitch_speed_mph",
        "pelvis_rotation_fp",
        "rotation_hip_shoulder_separation_fp",
        "max_rotation_hip_shoulder_separation",
        "max_torso_rotational_velo",
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
    ],
    "direction": ["pitch_speed_mph", "stride_length", "stride_angle", "max_cog_velo_x"],
    "heel_connection": [
        "pitch_speed_mph",
        "lead_knee_extension_from_fp_to_br",
        "lead_knee_extension_angular_velo_fp",
        "lead_grf_z_max",
    ],
    "drift": ["pitch_speed_mph", "cog_velo_pkh", "max_cog_velo_x", "stride_angle"],
}

PILOT_INTERPRETATION = {
    "hip_shoulder_separation": (
        "Highest-priority validation candidate: visual groups align with direct "
        "hip-shoulder separation POI metrics and pitch speed."
    ),
    "shoulder_horizontal_abduction": (
        "Promising but sample-limited: direct shoulder horizontal abduction "
        "metrics move in the expected direction, but the bad group is small."
    ),
    "torso_velo_z": (
        "Visual fast/slow maps better to torso rotational velocity than to pitch speed."
    ),
    "hip_extension": (
        "Pilot-positive but indirect: good/bad groups separate on speed and several "
        "transfer/lead-leg metrics, but POI still lacks direct FP hip extension angles."
    ),
    "direction": (
        "Current good/bad rubric likely mixes open stride and cross-fire into one "
        "bad group; angle-based categories should be split before strong interpretation."
    ),
    "heel_connection": (
        "Contested label: pitch-speed separation should not be treated as a clean "
        "mechanism until the rubric is tightened and matched to POI metrics."
    ),
    "drift": (
        "More consistent with center-of-mass velocity at PKH than with max COM "
        "velocity or pitch speed."
    ),
}

LABEL_COLUMNS = [
    "saved_at_utc",
    "rater_id",
    "session_pitch",
    "pitcher_id",
    "p_throws",
    "order",
    "view_used",
    "playback_speed",
    *LABEL_FIELDS,
    "skipped",
    "skip_reason",
    "notes",
]

_CFG: dict[str, object] = {}


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_manifest(rows: list[dict[str, object]]) -> None:
    fields = ["order", "session_pitch", "pitcher_id", "p_throws", "filename_new", "c3d_path"]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_c3d_path(row: pd.Series) -> Path:
    pitcher_id = f"{int(row['user']):06d}"
    return DATA_ROOT / "c3d" / pitcher_id / str(row["filename_new"])


def ensure_manifest(seed: int, pitchers: int, pitches_per_pitcher: int, rebuild: bool) -> list[dict[str, str]]:
    if MANIFEST_PATH.exists() and not rebuild:
        return _read_csv_records(MANIFEST_PATH)

    meta_path = DATA_ROOT / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    meta = pd.read_csv(meta_path)
    required = {"session_pitch", "user", "filename_new"}
    missing = required.difference(meta.columns)
    if missing:
        raise RuntimeError(f"metadata.csv missing required columns: {sorted(missing)}")
    if "p_throws" not in meta.columns:
        poi_path = DATA_ROOT / "poi" / "poi_metrics.csv"
        if not poi_path.exists():
            raise FileNotFoundError(f"Missing POI metrics for p_throws: {poi_path}")
        poi = pd.read_csv(poi_path, usecols=["session_pitch", "p_throws"])
        meta = meta.merge(poi.drop_duplicates("session_pitch"), on="session_pitch", how="left")
        if meta["p_throws"].isna().any():
            missing_throws = meta.loc[meta["p_throws"].isna(), "session_pitch"].head(10).tolist()
            raise RuntimeError(f"Missing p_throws after POI merge for session_pitch values: {missing_throws}")

    rows: list[dict[str, object]] = []
    rng = random.Random(seed)
    grouped = list(meta.groupby("user", sort=False))
    rng.shuffle(grouped)

    for user, group in grouped[:pitchers]:
        candidates = []
        for _, row in group.iterrows():
            c3d_path = build_c3d_path(row)
            if c3d_path.exists():
                candidates.append((row, c3d_path))
        rng.shuffle(candidates)
        for row, c3d_path in candidates[:pitches_per_pitcher]:
            rows.append({
                "session_pitch": str(row["session_pitch"]),
                "pitcher_id": f"{int(user):06d}",
                "p_throws": str(row["p_throws"]),
                "filename_new": str(row["filename_new"]),
                "c3d_path": str(c3d_path),
            })

    if not rows:
        raise RuntimeError("No displayable C3D files found for manifest generation.")

    rng.shuffle(rows)
    for idx, row in enumerate(rows, 1):
        row["order"] = idx
    _write_manifest(rows)
    return _read_csv_records(MANIFEST_PATH)


def ensure_labels_file() -> None:
    if not LABELS_PATH.exists():
        with LABELS_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
        return

    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        rows = list(reader)
    if existing_fields == LABEL_COLUMNS:
        return
    extra_fields = [field for field in existing_fields if field not in LABEL_COLUMNS]
    output_fields = [*LABEL_COLUMNS, *extra_fields]
    with LABELS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in output_fields})


def build_patch_manifest(base_manifest: list[dict[str, str]], patch_fields: list[str]) -> list[dict[str, str]]:
    if not patch_fields:
        return base_manifest
    for field in patch_fields:
        if field not in LABEL_FIELDS:
            raise RuntimeError(f"Unsupported patch field: {field}")
    labels = _read_csv_records(LABELS_PATH)
    manifest_by_pitch = {row["session_pitch"]: row for row in base_manifest}
    patch_rows = []
    for row in labels:
        needs_patch = False
        for field in patch_fields:
            value = row.get(field, "")
            allowed = FIELD_ALLOWED_VALUES[field]
            if value not in allowed:
                needs_patch = True
                break
        if not needs_patch:
            continue
        manifest_row = manifest_by_pitch.get(row.get("session_pitch", ""))
        if not manifest_row:
            raise RuntimeError(f"Label row references pitch outside manifest: {row.get('session_pitch', '')}")
        patch_rows.append({**manifest_row, "patch_rater_id": row.get("rater_id", "")})
    if not patch_rows:
        raise RuntimeError(f"No existing labels need patching for: {', '.join(patch_fields)}")
    return patch_rows


def load_motion_data(c3d_path: Path, frame_step: int) -> dict[str, object]:
    import ezc3d

    c3d = ezc3d.c3d(str(c3d_path))
    pts = c3d["data"]["points"]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    marker_labels = [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in labels]
    units_param = c3d["parameters"]["POINT"]["UNITS"]["value"]
    unit_str = units_param[0].decode().strip() if isinstance(units_param[0], bytes) else str(units_param[0]).strip()
    unit_key = unit_str.lower()
    if unit_key == "m":
        scale = 1000.0
    elif unit_key in {"mm", "millimeter", "millimeters"}:
        scale = 1.0
    else:
        raise RuntimeError(f"Unsupported C3D point unit: {unit_str}")

    needed = {marker for pair in SKELETON_CONNECTIONS for marker in pair}
    indices = {label: idx for idx, label in enumerate(marker_labels) if label in needed}
    fps = float(c3d["header"]["points"]["frame_rate"])
    first_frame = c3d["header"]["points"]["first_frame"]
    n_frames = pts.shape[2]
    frame_numbers = list(range(0, n_frames, frame_step))
    frames = []

    for frame in frame_numbers:
        frame_data = {}
        for label, idx in indices.items():
            val = pts[:3, idx, frame]
            if not np.any(val == 0) and not np.any(np.isnan(val)):
                frame_data[label] = [float(val[0]) * scale, float(val[2]) * scale, -float(val[1]) * scale]
        frames.append(frame_data)

    if not frames:
        raise RuntimeError("No valid marker frames found in C3D.")

    effective_fps = fps / frame_step
    times = [(first_frame + frame - 1) / fps for frame in frame_numbers]
    return {
        "fps": effective_fps,
        "connections": SKELETON_CONNECTIONS,
        "frames": frames,
        "metadata": {
            "fps": effective_fps,
            "n_frames": len(frames),
            "source_n_frames": n_frames,
            "first_frame": first_frame,
            "times": times,
            "connections": SKELETON_CONNECTIONS,
        },
    }


def _ordered_values(field: str, values: list[str]) -> list[str]:
    preferred = FIELD_VALUE_ORDER.get(field, [])
    seen = set(values)
    ordered = [value for value in preferred if value in seen]
    ordered.extend(sorted(value for value in values if value not in set(ordered)))
    return ordered


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if not pooled:
        return None
    return float((a.mean() - b.mean()) / pooled)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def compute_pilot_stats() -> dict[str, object]:
    from itertools import combinations
    from scipy import stats

    labels = pd.read_csv(LABELS_PATH, dtype={"session_pitch": str, "pitcher_id": str})
    poi_path = DATA_ROOT / "poi" / "poi_metrics.csv"
    if not poi_path.exists():
        raise FileNotFoundError(f"Missing POI metrics: {poi_path}")
    poi = pd.read_csv(poi_path, dtype={"session_pitch": str})
    joined = labels.merge(poi, on="session_pitch", how="left", suffixes=("", "_poi"), indicator="_poi_merge")

    session_counts = labels["session_pitch"].value_counts()
    duplicates = session_counts[session_counts > 1].to_dict()
    pitch_speed_missing = int(joined["pitch_speed_mph"].isna().sum()) if "pitch_speed_mph" in joined else len(joined)
    missing_poi_rows = int((joined["_poi_merge"] == "left_only").sum())

    result: dict[str, object] = {
        "qc": {
            "labeled_rows": int(len(labels)),
            "unique_session_pitch": int(labels["session_pitch"].nunique()),
            "duplicate_session_pitch_count": int(sum(count - 1 for count in duplicates.values())),
            "duplicate_session_pitch_values": duplicates,
            "missing_poi_rows": missing_poi_rows,
            "missing_pitch_speed_mph": pitch_speed_missing,
            "unique_pitchers": int(labels["pitcher_id"].nunique()) if "pitcher_id" in labels else None,
            "throws": labels["p_throws"].value_counts().sort_index().to_dict() if "p_throws" in labels else {},
        },
        "label_distributions": {},
        "fields": [],
        "notes": {
            "scope": "Exploratory pilot screen only; not a defensible inference test.",
            "tests": {
                "anova": "Parametric comparison of means across three or more groups.",
                "kruskal": "Non-parametric comparison across three or more groups.",
                "welch": "Two-group mean comparison allowing unequal variance.",
                "mwu": "Two-group non-parametric Mann-Whitney U comparison.",
            },
        },
    }

    distributions: dict[str, dict[str, int]] = {}
    for field in LABEL_FIELDS:
        if field not in labels:
            continue
        counts = labels[field].fillna("").replace("", "<blank>").value_counts().to_dict()
        values = _ordered_values(field, list(counts))
        distributions[field] = {value: int(counts[value]) for value in values}
    result["label_distributions"] = distributions

    fields_out = []
    for field, metrics in PILOT_FIELD_METRICS.items():
        if field not in joined:
            continue
        field_rows = joined[joined[field].notna() & ~joined[field].isin(["", "unclear"])]
        group_values = _ordered_values(field, sorted(field_rows[field].dropna().unique().tolist()))
        metric_rows = []
        for metric in metrics:
            if metric not in field_rows:
                metric_rows.append({"metric": metric, "missing_metric": True})
                continue
            groups = []
            group_summaries = []
            for value in group_values:
                series = pd.to_numeric(field_rows.loc[field_rows[field] == value, metric], errors="coerce").dropna()
                arr = series.to_numpy(dtype=float)
                if len(arr) == 0:
                    continue
                groups.append((value, arr))
                group_summaries.append({
                    "value": value,
                    "n": int(len(arr)),
                    "mean": _round_or_none(float(arr.mean()), 4),
                    "sd": _round_or_none(float(arr.std(ddof=1)), 4) if len(arr) > 1 else None,
                })
            tests: dict[str, object] = {}
            pairs = []
            if len(groups) == 2:
                left_name, left = groups[0]
                right_name, right = groups[1]
                tests["welch_p"] = _round_or_none(stats.ttest_ind(left, right, equal_var=False).pvalue)
                tests["mwu_p"] = _round_or_none(stats.mannwhitneyu(left, right, alternative="two-sided").pvalue)
                tests["cohen_d"] = _round_or_none(_cohen_d(left, right))
                tests["cohen_d_order"] = f"{left_name}-{right_name}"
            elif len(groups) > 2:
                arrays = [arr for _, arr in groups]
                tests["anova_p"] = _round_or_none(stats.f_oneway(*arrays).pvalue)
                tests["kruskal_p"] = _round_or_none(stats.kruskal(*arrays).pvalue)
                for i, j in combinations(range(len(groups)), 2):
                    left_name, left = groups[i]
                    right_name, right = groups[j]
                    pairs.append({
                        "left": left_name,
                        "right": right_name,
                        "welch_p": _round_or_none(stats.ttest_ind(left, right, equal_var=False).pvalue),
                        "mwu_p": _round_or_none(stats.mannwhitneyu(left, right, alternative="two-sided").pvalue),
                        "mean_diff": _round_or_none(float(left.mean() - right.mean()), 4),
                        "cohen_d": _round_or_none(_cohen_d(left, right)),
                    })
            metric_rows.append({
                "metric": metric,
                "groups": group_summaries,
                "tests": tests,
                "pairs": pairs,
            })
        fields_out.append({
            "field": field,
            "interpretation": PILOT_INTERPRETATION.get(field, ""),
            "metrics": metric_rows,
        })
    result["fields"] = fields_out
    return result


def html_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qualitative Mechanics Experiment</title>
<style>
body { margin:0; background:#111418; color:#e8eaed; font-family:Segoe UI, Arial, sans-serif; }
.app { display:grid; grid-template-columns:minmax(0,1fr) 360px; height:100vh; }
#viewer { position:relative; min-width:0; }
.side { border-left:1px solid #2c3138; padding:16px; overflow:auto; background:#171b21; }
.row { display:flex; gap:8px; align-items:center; margin:8px 0; flex-wrap:wrap; }
button, select, input, textarea { background:#242a33; color:#f2f4f8; border:1px solid #3a424d; border-radius:6px; padding:8px; }
button { cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button.warn { background:#6e2c2c; border-color:#a54242; }
label { display:block; margin-top:12px; font-size:13px; color:#bdc7d5; }
select, input, textarea { width:100%; box-sizing:border-box; }
input[type="range"] { width:100%; }
textarea { min-height:64px; resize:vertical; }
.status { color:#9fb0c4; font-size:13px; line-height:1.4; }
.pill { padding:3px 8px; border-radius:999px; background:#2d333b; font-size:12px; }
#loading { position:absolute; top:16px; left:16px; background:#171b21; padding:8px 10px; border:1px solid #3a424d; border-radius:6px; }
</style>
</head>
<body>
<div class="app">
  <div id="viewer"><div id="loading">Loading...</div></div>
  <div class="side">
    <h2>Qualitative Mechanics</h2>
    <div class="status" id="itemStatus"></div>
    <label>Rater ID<input id="raterId" placeholder="required"></label>
    <div class="row">
      <button onclick="prevItem()">Prev</button>
      <button onclick="nextItem()">Next</button>
      <button onclick="setView('home')">Home</button>
      <button onclick="setView('side')">Open side</button>
      <button onclick="setView('second')">Second base</button>
      <button onclick="setView('free')">Free</button>
      <button onclick="toggleReveal()">Reveal IDs</button>
      <button onclick="openDashboard()">Dashboard</button>
    </div>
    <div class="row">
      <button onclick="togglePlay()">Play/Pause</button>
      <span class="pill" id="viewPill">view: home</span>
      <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
    </div>
    <div class="row">
      <input id="frameSlider" type="range" min="0" max="0" value="0" oninput="seekFrame(this.value)">
      <span class="pill" id="frameText">0 / 0</span>
    </div>
    <form id="labelForm">
      <label>Hip-shoulder separation<select name="hip_shoulder_separation"><option>unclear</option><option>good</option><option>average</option><option>bad</option></select></label>
      <label>Glute / quad dominance<select name="lower_body_dominance"><option>unclear</option><option>glute</option><option>quad</option><option>mixed</option></select></label>
      <label>Direction<select name="direction"><option>unclear</option><option>good</option><option>bad</option></select></label>
      <label>Shoulder horizontal abduction<select name="shoulder_horizontal_abduction"><option>unclear</option><option>good</option><option>average</option><option>bad</option></select></label>
      <label>Torso Velo Z<select name="torso_velo_z"><option>unclear</option><option>fast</option><option>slow</option></select></label>
      <label>Hip Extension<select name="hip_extension"><option>unclear</option><option>good</option><option>bad</option></select></label>
      <label>Heel connection<select name="heel_connection"><option>unclear</option><option>connected</option><option>early_extension</option></select></label>
      <label>Drift<select name="drift"><option>unclear</option><option>good</option><option>average</option><option>bad</option></select></label>
      <label>Skip reason<input name="skip_reason" placeholder="only when skipped"></label>
      <label>Notes<textarea name="notes"></textarea></label>
    </form>
    <div class="row">
      <button class="primary" onclick="saveLabel(false)">Save label</button>
      <button class="warn" onclick="saveLabel(true)">Skip / bad display</button>
    </div>
    <div class="status" id="saveStatus"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
let manifest = [];
let idx = 0;
let motion = null;
let scene, camera, renderer, controls, lines = [];
let currentFrame = 0, playing = true, lastTime = 0, currentView = 'home';
let revealIds = false;
const PATCH_FIELDS = __PATCH_FIELDS__;

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function initThree() {
  const el = document.getElementById('viewer');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f141a);
  camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 1, 100000);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(renderer.domElement);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.GridHelper(6000, 20, 0x3a424d, 0x222830));
  window.addEventListener('resize', () => {
    camera.aspect = el.clientWidth / el.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(el.clientWidth, el.clientHeight);
  });
}

function clearLines() {
  for (const line of lines) scene.remove(line);
  lines = [];
}

function buildSkeleton() {
  clearLines();
  const mat = new THREE.LineBasicMaterial({color:0xe8eaed, linewidth:2});
  for (const pair of motion.connections) {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const line = new THREE.Line(geom, mat);
    line.userData = {a:pair[0], b:pair[1]};
    scene.add(line);
    lines.push(line);
  }
}

function updateSkeleton() {
  if (!motion) return;
  const frame = motion.frames[currentFrame] || {};
  for (const line of lines) {
    const a = frame[line.userData.a], b = frame[line.userData.b];
    line.visible = !!(a && b);
    if (line.visible) {
      line.geometry.setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
    }
  }
  document.getElementById('frameSlider').value = currentFrame;
  document.getElementById('frameText').textContent = currentFrame + ' / ' + (motion.frames.length - 1);
}

function centerOfFrame() {
  const frame = motion.frames[0] || {};
  const vals = Object.values(frame);
  const avg = vals.reduce((acc, p) => [acc[0]+p[0], acc[1]+p[1], acc[2]+p[2]], [0,0,0]);
  return vals.length ? new THREE.Vector3(avg[0]/vals.length, avg[1]/vals.length, avg[2]/vals.length) : new THREE.Vector3();
}

function setView(view) {
  currentView = view;
  document.getElementById('viewPill').textContent = 'view: ' + view;
  if (!motion) return;
  const target = centerOfFrame();
  controls.target.copy(target);
  controls.enableRotate = view === 'free';
  controls.enablePan = view === 'free';
  const item = manifest[idx] || {};
  if (view === 'home') {
    camera.position.set(target.x + 8000, target.y + 1000, target.z);
  } else if (view === 'side') {
    const lateral = item.p_throws === 'L' ? -5000 : 5000;
    camera.position.set(target.x, target.y + 1000, target.z + lateral);
  } else if (view === 'second') {
    camera.position.set(target.x - 8000, target.y + 1000, target.z);
  }
  camera.lookAt(target);
  controls.update();
}

function updateItemStatus() {
  const item = manifest[idx] || {};
  const mode = PATCH_FIELDS.length ? `Patch ${PATCH_FIELDS.join(', ')} | ` : '';
  const base = `${mode}Item ${idx + 1}/${manifest.length}`;
  const detail = revealIds ? ` | pitch ${item.session_pitch} | pitcher ${item.pitcher_id} | throws ${item.p_throws}` : '';
  document.getElementById('itemStatus').textContent = base + detail;
}

function toggleReveal() {
  revealIds = !revealIds;
  updateItemStatus();
}

function openDashboard() {
  window.open('/dashboard', '_blank');
}

async function loadItem() {
  const item = manifest[idx];
  document.getElementById('labelForm').reset();
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = revealIds ? ('Loading ' + item.session_pitch + '...') : 'Loading...';
  if (PATCH_FIELDS.length && item.patch_rater_id) document.getElementById('raterId').value = item.patch_rater_id;
  updateItemStatus();
  try {
    motion = await api('/api/motion?session_pitch=' + encodeURIComponent(item.session_pitch));
    currentFrame = 0;
    document.getElementById('frameSlider').max = motion.frames.length - 1;
    document.getElementById('frameSlider').value = 0;
    buildSkeleton();
    updateSkeleton();
    setView(currentView);
    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    motion = null;
    clearLines();
    document.getElementById('loading').textContent = err.message;
  }
}

function nextItem() { if (idx < manifest.length - 1) { idx++; loadItem(); } }
function prevItem() { if (idx > 0) { idx--; loadItem(); } }
function togglePlay() { playing = !playing; }
function seekFrame(frame) {
  if (!motion) return;
  currentFrame = parseInt(frame, 10);
  playing = false;
  updateSkeleton();
}

async function saveLabel(skipped) {
  const rater = document.getElementById('raterId').value.trim();
  if (!rater) { document.getElementById('saveStatus').textContent = 'Rater ID is required.'; return; }
  const form = new FormData(document.getElementById('labelForm'));
  const payload = Object.fromEntries(form.entries());
  Object.assign(payload, {
    rater_id: rater,
    session_pitch: manifest[idx].session_pitch,
    view_used: currentView,
    playback_speed: document.getElementById('speed').value,
    skipped: skipped ? 'true' : 'false',
    patch_fields: PATCH_FIELDS,
  });
  try {
    await api('/api/label', {method:'POST', body:JSON.stringify(payload)});
    document.getElementById('saveStatus').textContent = 'Saved.';
    if (idx < manifest.length - 1) nextItem();
  } catch (err) {
    document.getElementById('saveStatus').textContent = err.message;
  }
}

function animate(t) {
  requestAnimationFrame(animate);
  if (motion && playing && t - lastTime > (1000 / motion.fps) / parseFloat(document.getElementById('speed').value)) {
    currentFrame = (currentFrame + 1) % motion.frames.length;
    updateSkeleton();
    lastTime = t;
  }
  if (controls) controls.update();
  if (renderer) renderer.render(scene, camera);
}

async function main() {
  initThree();
  manifest = await api('/api/manifest');
  await loadItem();
  animate(0);
}
main().catch(err => document.getElementById('loading').textContent = err.message);
</script>
</body>
</html>"""


def dashboard_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qualitative Mechanics Pilot Dashboard</title>
<style>
body { margin:0; background:#111418; color:#e8eaed; font-family:Segoe UI, Arial, sans-serif; }
main { max-width:1280px; margin:0 auto; padding:24px; }
h1 { margin:0 0 8px; font-size:24px; }
h2 { margin:30px 0 10px; font-size:18px; color:#f2f4f8; }
h3 { margin:18px 0 8px; font-size:14px; color:#9fb0c4; text-transform:uppercase; letter-spacing:0; }
.muted { color:#9fb0c4; font-size:13px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:8px; margin:16px 0; }
.stat { border:1px solid #2c3138; background:#171b21; border-radius:6px; padding:10px; }
.stat b { display:block; font-size:20px; margin-top:4px; }
.field { border-top:2px solid #3a424d; padding-top:18px; margin-top:28px; }
.field > h2 { font-size:22px; margin-top:0; }
.metric { margin:18px 0 22px; padding-left:12px; border-left:3px solid #2c3138; }
.metric h3 { margin-top:0; }
.note { color:#bdc7d5; margin:4px 0 12px; }
table { width:100%; border-collapse:collapse; margin:8px 0 18px; font-size:13px; }
th, td { border-bottom:1px solid #2c3138; padding:7px 8px; text-align:right; vertical-align:top; }
th:first-child, td:first-child { text-align:left; }
th { color:#bdc7d5; font-weight:600; background:#171b21; position:sticky; top:0; }
.sig { color:#7ee787; font-weight:600; }
.warn { color:#ffcf7a; }
.pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#242a33; color:#bdc7d5; margin-right:4px; }
.toolbar { display:flex; gap:10px; align-items:center; margin:14px 0; flex-wrap:wrap; }
button { background:#242a33; color:#f2f4f8; border:1px solid #3a424d; border-radius:6px; padding:8px 10px; cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button:disabled { opacity:.55; cursor:default; }
a { color:#7cb7ff; }
</style>
</head>
<body>
<main>
  <h1>Qualitative Mechanics Pilot Dashboard</h1>
  <div class="muted">Exploratory screen only. Click refresh after a labeling batch to recompute.</div>
  <div class="toolbar">
    <button class="primary" id="refreshBtn" onclick="refreshStats()">Refresh results</button>
    <span class="muted" id="refreshStatus">Not loaded yet.</span>
  </div>
  <div id="qc" class="grid"><div class="stat"><span class="muted">Status</span><b>Waiting</b></div></div>
  <h2>Label Distributions</h2>
  <div id="distributions"></div>
  <h2>Pilot Statistics</h2>
  <div id="fields"></div>
</main>
<script>
function fmt(v, digits=4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(digits);
  return String(v);
}
function pcell(v) {
  if (v === null || v === undefined) return '';
  const cls = v < 0.05 ? 'sig' : (v < 0.10 ? 'warn' : '');
  return `<span class="${cls}">${fmt(v, 4)}</span>`;
}
function table(headers, rows) {
  return `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderStats(data) {
  const loadedAt = new Date();
  const qc = data.qc;
  document.getElementById('qc').innerHTML = [
    ['Rows', qc.labeled_rows],
    ['Unique pitches', qc.unique_session_pitch],
    ['Duplicate pitches', qc.duplicate_session_pitch_count],
    ['Missing POI', qc.missing_poi_rows],
    ['Missing speed', qc.missing_pitch_speed_mph],
    ['Pitchers', qc.unique_pitchers],
    ['Throws', Object.entries(qc.throws).map(([k,v]) => `${k} ${v}`).join(', ')],
  ].map(([k,v]) => `<div class="stat"><span class="muted">${k}</span><b>${fmt(v)}</b></div>`).join('');

  document.getElementById('distributions').innerHTML = Object.entries(data.label_distributions).map(([field, counts]) => {
    const rows = Object.entries(counts).map(([value, n]) => [value, n]);
    return `<h3>${field}</h3>` + table(['Value', 'n'], rows);
  }).join('');

  document.getElementById('fields').innerHTML = data.fields.map(field => {
    const metrics = field.metrics.map(metric => {
      if (metric.missing_metric) return `<section class="metric"><h3>${metric.metric}</h3><div class="warn">Metric missing from joined data.</div></section>`;
      const groupRows = metric.groups.map(g => [g.value, g.n, fmt(g.mean, 4), fmt(g.sd, 4)]);
      const tests = metric.tests || {};
      const testBits = Object.entries(tests).map(([k,v]) => `<span class="pill">${k}: ${k.endsWith('_p') ? pcell(v) : fmt(v, 4)}</span>`).join(' ');
      const pairRows = (metric.pairs || []).map(p => [
        `${p.left} vs ${p.right}`,
        pcell(p.welch_p),
        pcell(p.mwu_p),
        fmt(p.mean_diff, 4),
        fmt(p.cohen_d, 4),
      ]);
      return `<section class="metric"><h3>${metric.metric}</h3>` +
        table(['Group', 'n', 'mean', 'sd'], groupRows) +
        `<div>${testBits}</div>` +
        (pairRows.length ? table(['Pair', 'Welch p', 'MWU p', 'mean diff', 'Cohen d'], pairRows) : '') +
        `</section>`;
    }).join('');
    return `<section class="field"><h2>${field.field}</h2><div class="note">${field.interpretation}</div>${metrics}</section>`;
  }).join('');
  document.getElementById('refreshStatus').textContent = `Last updated ${loadedAt.toLocaleTimeString()} | rows ${qc.labeled_rows}`;
}
async function refreshStats() {
  const btn = document.getElementById('refreshBtn');
  const status = document.getElementById('refreshStatus');
  btn.disabled = true;
  status.textContent = 'Refreshing...';
  try {
    const res = await fetch('/api/pilot-stats?ts=' + Date.now(), {cache:'no-store'});
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || res.statusText);
    renderStats(data);
  } catch (err) {
    status.textContent = 'Refresh failed: ' + err.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(fmt % args + "\n")

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = html_page().replace("__PATCH_FIELDS__", json.dumps(_CFG.get("patch_fields", [])))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/dashboard":
            body = dashboard_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/manifest":
            self._json(_CFG["manifest"])
            return
        if parsed.path == "/api/pilot-stats":
            try:
                self._json(compute_pilot_stats())
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/motion":
            qs = parse_qs(parsed.query)
            session_pitch = qs.get("session_pitch", [""])[0]
            item = _CFG["c3d_map"].get(session_pitch)  # type: ignore[index]
            if not item:
                self._json({"error": f"No manifest item for {session_pitch}"}, 404)
                return
            try:
                self._json(load_motion_data(Path(item["c3d_path"]), int(_CFG["frame_step"])))  # type: ignore[index]
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            item = _CFG["c3d_map"].get(str(payload.get("session_pitch", "")))  # type: ignore[index]
            if not item:
                self._json({"error": "Label references a pitch outside the manifest."}, 400)
                return
            patch_fields = payload.get("patch_fields") or []
            if patch_fields:
                self._patch_existing_label(payload, patch_fields)
                self._json({"ok": True, "patched": True})
                return

            row = {col: "" for col in LABEL_COLUMNS}
            row.update({
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "pitcher_id": item["pitcher_id"],
                "p_throws": item["p_throws"],
                "order": item["order"],
            })
            for col in LABEL_COLUMNS:
                if col in payload:
                    row[col] = str(payload[col])
            if row["skipped"] == "true" and not row["skip_reason"].strip():
                self._json({"error": "Skip reason is required for skipped items."}, 400)
                return
            with LABELS_PATH.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
                writer.writerow(row)
            self._json({"ok": True})
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _patch_existing_label(self, payload: dict[str, object], patch_fields: list[str]) -> None:
        for field in patch_fields:
            if field not in LABEL_FIELDS:
                raise RuntimeError(f"Unsupported patch field: {field}")
        rows = _read_csv_records(LABELS_PATH)
        session_pitch = str(payload.get("session_pitch", ""))
        rater_id = str(payload.get("rater_id", ""))
        matches = [idx for idx, row in enumerate(rows) if row.get("session_pitch") == session_pitch and row.get("rater_id") == rater_id]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one label row for rater={rater_id} session_pitch={session_pitch}, found {len(matches)}")
        row = rows[matches[0]]
        for field in patch_fields:
            value = str(payload.get(field, ""))
            if value not in FIELD_ALLOWED_VALUES[field]:
                raise RuntimeError(f"Invalid value for {field}: {value}")
            row[field] = value
        row["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
        with LABELS_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the qualitative mechanics labeling experiment.")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--pitchers", type=int, default=20)
    parser.add_argument("--pitches-per-pitcher", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--check-first-load", action="store_true")
    parser.add_argument("--patch-field", action="append", default=[])
    parser.add_argument("--pilot-stats", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_labels_file()
    base_manifest = ensure_manifest(args.seed, args.pitchers, args.pitches_per_pitcher, args.rebuild_manifest)
    manifest = build_patch_manifest(base_manifest, args.patch_field)

    _CFG["manifest"] = manifest
    _CFG["c3d_map"] = {row["session_pitch"]: row for row in manifest}
    _CFG["frame_step"] = args.frame_step
    _CFG["patch_fields"] = args.patch_field

    if args.pilot_stats:
        print(json.dumps(compute_pilot_stats(), ensure_ascii=False, indent=2))
        return

    if args.check_first_load:
        first = manifest[0]
        motion = load_motion_data(Path(first["c3d_path"]), args.frame_step)
        print(
            "Loaded first manifest pitch:",
            first["session_pitch"],
            f"frames={len(motion['frames'])}",
            f"fps={motion['fps']}",
        )
        return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Qualitative Mechanics Experiment -> {url}")
    print(f"Pilot dashboard -> {url}/dashboard")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Labels:   {LABELS_PATH}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
