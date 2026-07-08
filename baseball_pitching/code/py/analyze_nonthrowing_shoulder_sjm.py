#!/usr/bin/env python3
"""
Replicate non-throwing shoulder movement (SJM) from fp_poi_time to BR_time
and classify post-FP glove-shoulder trajectory patterns.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FULL_SIG_DIR = DATA_DIR / "full_sig"
POI_PATH = DATA_DIR / "poi" / "poi_metrics.csv"
META_PATH = DATA_DIR / "metadata.csv"
OUT_DIR = ROOT / "code" / "py" / "nonthrowing_shoulder_sjm_outputs"
IMG_DIR = ROOT / "imgs"

LANDMARK_COLS = [
    "session_pitch",
    "time",
    "BR_time",
    "glove_shoulder_jc_x",
    "glove_shoulder_jc_y",
    "glove_shoulder_jc_z",
    "shoulder_jc_x",
    "shoulder_jc_y",
    "shoulder_jc_z",
    "rear_hip_x",
    "rear_hip_y",
    "rear_hip_z",
    "lead_hip_x",
    "lead_hip_y",
    "lead_hip_z",
]
EVENT_COLS = ["session_pitch", "fp_poi_time", "pitch_speed_mph", "p_throws"]
META_COLS = ["session_pitch", "session_height_m"]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_landmarks() -> pd.DataFrame:
    with zipfile.ZipFile(FULL_SIG_DIR / "landmarks.zip", "r") as zf:
        with zf.open("landmarks.csv") as fh:
            return pd.read_csv(fh, usecols=LANDMARK_COLS)


def load_inputs() -> pd.DataFrame:
    landmarks = load_landmarks()
    poi = pd.read_csv(POI_PATH, usecols=EVENT_COLS)
    meta = pd.read_csv(META_PATH, usecols=META_COLS)
    merged = landmarks.merge(poi, on="session_pitch", how="inner")
    merged = merged.merge(meta, on="session_pitch", how="inner")
    return merged.sort_values(["session_pitch", "time"]).reset_index(drop=True)


def unit(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0:
        return None
    return vec / norm


def torso_axes(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    glove_shoulder = row[["glove_shoulder_jc_x", "glove_shoulder_jc_y", "glove_shoulder_jc_z"]].to_numpy(dtype=float)
    throw_shoulder = row[["shoulder_jc_x", "shoulder_jc_y", "shoulder_jc_z"]].to_numpy(dtype=float)
    rear_hip = row[["rear_hip_x", "rear_hip_y", "rear_hip_z"]].to_numpy(dtype=float)
    lead_hip = row[["lead_hip_x", "lead_hip_y", "lead_hip_z"]].to_numpy(dtype=float)

    shoulder_center = (glove_shoulder + throw_shoulder) / 2.0
    pelvis_center = (rear_hip + lead_hip) / 2.0

    u_z = unit(shoulder_center - pelvis_center)
    if u_z is None:
        return None

    shoulder_line = glove_shoulder - throw_shoulder
    lateral = shoulder_line - np.dot(shoulder_line, u_z) * u_z
    u_y = unit(lateral)
    if u_y is None:
        return None

    if row["p_throws"] == "L":
        u_y = -u_y

    u_x = unit(np.cross(u_y, u_z))
    if u_x is None:
        return None

    return u_x, u_y, u_z


def glove_local_position(row: pd.Series) -> np.ndarray | None:
    axes = torso_axes(row)
    if axes is None:
        return None

    glove_shoulder = row[["glove_shoulder_jc_x", "glove_shoulder_jc_y", "glove_shoulder_jc_z"]].to_numpy(dtype=float)
    rear_hip = row[["rear_hip_x", "rear_hip_y", "rear_hip_z"]].to_numpy(dtype=float)
    lead_hip = row[["lead_hip_x", "lead_hip_y", "lead_hip_z"]].to_numpy(dtype=float)
    pelvis_center = (rear_hip + lead_hip) / 2.0
    rel = glove_shoulder - pelvis_center
    u_x, u_y, u_z = axes
    return np.array([np.dot(rel, u_x), np.dot(rel, u_y), np.dot(rel, u_z)], dtype=float)


def interpolate_local_curve(times: np.ndarray, values: np.ndarray, n_points: int = 101) -> np.ndarray:
    tau = (times - times[0]) / (times[-1] - times[0])
    grid = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, values.shape[1]), dtype=float)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(grid, tau, values[:, dim])
    return out


def classify_pattern(displacement: np.ndarray, tau: np.ndarray, height_m: float) -> tuple[str, dict[str, float]]:
    peak_idx = int(np.nanargmax(displacement))
    peak_disp = float(displacement[peak_idx])
    peak_tau = float(tau[peak_idx])
    final_disp = float(displacement[-1])

    max_disp_norm = peak_disp / height_m
    final_disp_norm = final_disp / height_m
    late_extra_norm = max(final_disp - displacement[len(displacement) // 2], 0.0) / height_m
    recovery_ratio = (peak_disp - final_disp) / peak_disp if peak_disp > 0 else 0.0

    small_threshold = 0.012
    late_threshold = 0.008

    if max_disp_norm <= small_threshold:
        label = "always_small"
    elif peak_tau >= 0.60 and late_extra_norm >= late_threshold:
        label = "late_pull"
    elif peak_tau <= 0.45 and recovery_ratio >= 0.45 and final_disp_norm <= small_threshold:
        label = "early_large_then_stable"
    else:
        label = "mixed"

    return label, {
        "peak_disp_m": peak_disp,
        "peak_tau": peak_tau,
        "final_disp_m": final_disp,
        "max_disp_norm_height": max_disp_norm,
        "final_disp_norm_height": final_disp_norm,
        "late_extra_norm_height": late_extra_norm,
        "recovery_ratio": recovery_ratio,
    }


def analyze_pitch(group: pd.DataFrame) -> dict[str, object] | None:
    fp_time = group["fp_poi_time"].iloc[0]
    br_time = group["BR_time"].iloc[0]
    height_m = group["session_height_m"].iloc[0]
    pitch_speed = group["pitch_speed_mph"].iloc[0]

    if any(pd.isna(v) for v in [fp_time, br_time, height_m, pitch_speed]):
        return None
    if br_time <= fp_time:
        return None

    phase = group[(group["time"] >= fp_time) & (group["time"] <= br_time)].copy()
    if len(phase) < 5:
        return None

    local_positions = []
    keep_index = []
    for idx, row in phase.iterrows():
        local = glove_local_position(row)
        if local is not None and np.all(np.isfinite(local)):
            local_positions.append(local)
            keep_index.append(idx)

    if len(local_positions) < 5:
        return None

    phase = phase.loc[keep_index].copy()
    phase["local_x"] = [p[0] for p in local_positions]
    phase["local_y"] = [p[1] for p in local_positions]
    phase["local_z"] = [p[2] for p in local_positions]

    times = phase["time"].to_numpy(dtype=float)
    local_xyz = phase[["local_x", "local_y", "local_z"]].to_numpy(dtype=float)
    global_xyz = phase[["glove_shoulder_jc_x", "glove_shoulder_jc_y", "glove_shoulder_jc_z"]].to_numpy(dtype=float)
    global_center = global_xyz.mean(axis=0)
    global_centered = global_xyz - global_center

    sjm = float(np.sqrt(np.mean(np.sum(global_centered**2, axis=1))) / height_m)
    sjm_x = float(np.sqrt(np.mean(global_centered[:, 0] ** 2)) / height_m)
    sjm_y = float(np.sqrt(np.mean(global_centered[:, 1] ** 2)) / height_m)
    sjm_z = float(np.sqrt(np.mean(global_centered[:, 2] ** 2)) / height_m)

    disp_from_fp = np.linalg.norm(local_xyz - local_xyz[0], axis=1)
    tau = (times - times[0]) / (times[-1] - times[0])
    pattern, pattern_metrics = classify_pattern(disp_from_fp, tau, height_m)
    curve = interpolate_local_curve(times, local_xyz)

    result = {
        "session_pitch": group["session_pitch"].iloc[0],
        "pitch_speed_mph": float(pitch_speed),
        "session_height_m": float(height_m),
        "p_throws": group["p_throws"].iloc[0],
        "fp_poi_time": float(fp_time),
        "BR_time": float(br_time),
        "n_frames": int(len(phase)),
        "sjm": sjm,
        "sjm_x": sjm_x,
        "sjm_y": sjm_y,
        "sjm_z": sjm_z,
        "fp_local_x": float(local_xyz[0, 0]),
        "fp_local_y": float(local_xyz[0, 1]),
        "fp_local_z": float(local_xyz[0, 2]),
        "br_local_x": float(local_xyz[-1, 0]),
        "br_local_y": float(local_xyz[-1, 1]),
        "br_local_z": float(local_xyz[-1, 2]),
        "pattern": pattern,
        **pattern_metrics,
    }
    return {"metrics": result, "curve": curve}


def one_way_anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict[str, float] | None:
    groups = [g[value_col].dropna().to_numpy(dtype=float) for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    stat, p_value = stats.f_oneway(*groups)
    return {"f_stat": float(stat), "p_value": float(p_value)}


def save_plot_curves(curves: dict[str, list[np.ndarray]]) -> Path:
    labels = ["X local", "Y local", "Z local"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    grid = np.linspace(0.0, 100.0, 101)

    for ax, dim, label in zip(axes, range(3), labels):
        for pattern, pattern_curves in curves.items():
            if not pattern_curves:
                continue
            stack = np.stack(pattern_curves)
            mean = stack[:, :, dim].mean(axis=0)
            std = stack[:, :, dim].std(axis=0)
            ax.plot(grid, mean, label=f"{pattern} (n={len(pattern_curves)})")
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        ax.set_ylabel(f"{label} (m)")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(100, color="black", linewidth=0.8)
        ax.legend()

    axes[-1].set_xlabel("FP to BR normalized time (%)")
    fig.suptitle("Non-throwing shoulder local trajectory by pattern")
    fig.tight_layout()
    out_path = IMG_DIR / "nonthrowing_shoulder_local_trajectory_patterns.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.7)
    valid = df[[x_col, y_col]].dropna()
    r, p_value = stats.pearsonr(valid[x_col], valid[y_col])
    ax.set_title(f"{title}\nr={r:.3f}, p={p_value:.4g}, n={len(valid)}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    out_path = IMG_DIR / out_name
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_pitches": int(len(df)),
        "pattern_counts": df["pattern"].value_counts().to_dict(),
        "pattern_speed_means": df.groupby("pattern")["pitch_speed_mph"].mean().round(3).to_dict(),
    }

    for col in [
        "sjm",
        "sjm_x",
        "sjm_y",
        "sjm_z",
        "fp_local_x",
        "fp_local_y",
        "fp_local_z",
        "br_local_x",
        "br_local_y",
        "br_local_z",
        "peak_disp_m",
        "final_disp_m",
    ]:
        valid = df[[col, "pitch_speed_mph"]].dropna()
        if len(valid) >= 3:
            r, p_value = stats.pearsonr(valid[col], valid["pitch_speed_mph"])
            summary[f"{col}_vs_speed"] = {"r": float(r), "p_value": float(p_value), "n": int(len(valid))}

    anova = one_way_anova(df[df["pattern"] != "mixed"], "pitch_speed_mph", "pattern")
    if anova is not None:
        summary["pattern_speed_anova_excluding_mixed"] = anova

    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_inputs()
    analyses: list[dict[str, object]] = []
    curves: dict[str, list[np.ndarray]] = {
        "always_small": [],
        "late_pull": [],
        "early_large_then_stable": [],
        "mixed": [],
    }

    for _, group in df.groupby("session_pitch", sort=True):
        analyzed = analyze_pitch(group)
        if analyzed is None:
            continue
        analyses.append(analyzed["metrics"])
        curves[analyzed["metrics"]["pattern"]].append(analyzed["curve"])

    metrics_df = pd.DataFrame(analyses).sort_values("session_pitch")
    if metrics_df.empty:
        raise RuntimeError("No valid pitches were analyzed.")

    metrics_path = OUT_DIR / "nonthrowing_shoulder_sjm_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = build_summary(metrics_df)
    summary["curve_plot"] = str(save_plot_curves(curves))
    summary["sjm_scatter"] = str(
        save_scatter(
            metrics_df,
            "sjm",
            "pitch_speed_mph",
            "SJM vs pitch speed",
            "nonthrowing_shoulder_sjm_vs_speed.png",
        )
    )
    summary["late_scatter"] = str(
        save_scatter(
            metrics_df,
            "final_disp_m",
            "pitch_speed_mph",
            "Final displacement from FP vs pitch speed",
            "nonthrowing_shoulder_final_disp_vs_speed.png",
        )
    )

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
