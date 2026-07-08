#!/usr/bin/env python3
"""
Recalculate paper-style non-throwing shoulder SJM from raw C3D markers.
Uses LSHO/RSHO from fp_poi_time to BR_time.
"""

from __future__ import annotations

import json
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POI_PATH = DATA_DIR / "poi" / "poi_metrics.csv"
META_PATH = DATA_DIR / "metadata.csv"
OUT_DIR = ROOT / "code" / "py" / "nonthrowing_shoulder_sjm_c3d_outputs"


def decode_label(label: object) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8").strip()
    return str(label).strip()


def marker_index(labels: list[object], name: str) -> int | None:
    for i, label in enumerate(labels):
        if decode_label(label) == name:
            return i
    return None


def load_input_table() -> pd.DataFrame:
    poi = pd.read_csv(
        POI_PATH,
        usecols=["session_pitch", "p_throws", "pitch_speed_mph", "fp_poi_time"],
    )
    meta = pd.read_csv(
        META_PATH,
        usecols=["user", "session_pitch", "session_height_m", "filename_new"],
    )
    df = poi.merge(meta, on="session_pitch", how="inner")
    df["user_dir"] = df["user"].astype(int).map(lambda x: f"{x:06d}")
    df["c3d_path"] = df.apply(
        lambda r: ROOT / "data" / "c3d" / r["user_dir"] / str(r["filename_new"]),
        axis=1,
    )
    return df


def choose_marker(p_throws: str) -> str:
    if p_throws == "R":
        return "LSHO"
    if p_throws == "L":
        return "RSHO"
    raise ValueError(f"Unexpected p_throws={p_throws!r}")


def c3d_time_vector(c3d: ezc3d.c3d) -> np.ndarray:
    fps = float(c3d["header"]["points"]["frame_rate"])
    first_frame = int(c3d["header"]["points"]["first_frame"])
    n_frames = int(c3d["data"]["points"].shape[2])
    return (np.arange(n_frames) + first_frame - 1) / fps


def point_scale(c3d: ezc3d.c3d) -> float:
    try:
        units = c3d["parameters"]["POINT"]["UNITS"]["value"][0]
        unit_str = decode_label(units).lower()
        if unit_str == "m":
            return 1.0
        if unit_str == "mm":
            return 0.001
    except Exception:
        pass
    return 0.001


def analyze_row(row: pd.Series) -> dict[str, float] | None:
    c3d_path = Path(row["c3d_path"])
    if not c3d_path.exists():
        return None

    fp_time = row["fp_poi_time"]
    height_m = row["session_height_m"]
    pitch_speed = row["pitch_speed_mph"]
    if pd.isna(fp_time) or pd.isna(height_m) or pd.isna(pitch_speed):
        return None

    c3d = ezc3d.c3d(str(c3d_path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    target = choose_marker(str(row["p_throws"]))
    idx = marker_index(labels, target)
    if idx is None:
        return None

    points = c3d["data"]["points"][:3, idx, :].T.astype(float)
    scale = point_scale(c3d)
    points *= scale
    times = c3d_time_vector(c3d)

    br_time = float(times[-1])
    if br_time <= fp_time:
        return None

    valid_mask = np.isfinite(points).all(axis=1) & ~(points == 0).all(axis=1)
    window_mask = valid_mask & (times >= fp_time) & (times <= br_time)
    phase = points[window_mask]
    phase_times = times[window_mask]
    if len(phase) < 5:
        return None

    centered = phase - phase.mean(axis=0)
    sjm = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))) / height_m)
    sjm_x = float(np.sqrt(np.mean(centered[:, 0] ** 2)) / height_m)
    sjm_y = float(np.sqrt(np.mean(centered[:, 1] ** 2)) / height_m)
    sjm_z = float(np.sqrt(np.mean(centered[:, 2] ** 2)) / height_m)

    return {
        "session_pitch": row["session_pitch"],
        "marker_used": target,
        "pitch_speed_mph": float(pitch_speed),
        "session_height_m": float(height_m),
        "fp_poi_time": float(fp_time),
        "br_time_c3d_end": br_time,
        "n_frames": int(len(phase)),
        "sjm": sjm,
        "sjm_x": sjm_x,
        "sjm_y": sjm_y,
        "sjm_z": sjm_z,
    }


def summarize(df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {"n_pitches": int(len(df))}
    for col in ["sjm", "sjm_x", "sjm_y", "sjm_z"]:
        valid = df[[col, "pitch_speed_mph"]].dropna()
        if len(valid) >= 3:
            r, p = stats.pearsonr(valid[col], valid["pitch_speed_mph"])
            out[f"{col}_vs_speed"] = {"r": float(r), "p_value": float(p), "n": int(len(valid))}
            out[f"{col}_mean_x1e3"] = float(valid[col].mean() * 1000.0)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_input_table()
    results = []
    for _, row in rows.iterrows():
        item = analyze_row(row)
        if item is not None:
            results.append(item)

    if not results:
        raise RuntimeError("No valid C3D marker SJM results were produced.")

    df = pd.DataFrame(results).sort_values("session_pitch")
    metrics_path = OUT_DIR / "nonthrowing_shoulder_sjm_c3d_marker_metrics.csv"
    summary_path = OUT_DIR / "summary.json"
    df.to_csv(metrics_path, index=False)
    summary = summarize(df)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
