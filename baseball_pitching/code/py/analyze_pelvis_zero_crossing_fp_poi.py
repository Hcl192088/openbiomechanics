from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "baseball_pitching" / "data"
ANGLE_PATH = DATA_DIR / "full_sig" / "joint_angles.csv"
POI_PATH = DATA_DIR / "poi" / "poi_metrics.csv"
OUT_DIR = ROOT / "baseball_pitching" / "code" / "py" / "pelvis_zero_crossing_fp_poi_outputs"


def unique_event(group: pd.DataFrame, column: str, session_pitch: str) -> float:
    values = group[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{session_pitch}: expected one {column}, found {len(values)}")
    return float(values[0])


def crossing_time(t0: float, y0: float, t1: float, y1: float) -> float:
    if y1 == y0:
        raise ValueError("cannot interpolate a crossing between equal angle values")
    return float(t0 + (-y0) * (t1 - t0) / (y1 - y0))


def regression(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    complete = pd.DataFrame({"x": x, "y": y}).dropna()
    fit = stats.linregress(complete["x"], complete["y"])
    return {
        "n": len(complete),
        "r": fit.rvalue,
        "r2": fit.rvalue**2,
        "slope": fit.slope,
        "p": fit.pvalue,
    }


def main() -> None:
    angle_columns = [
        "session_pitch",
        "time",
        "pkh_time",
        "fp_poi_time",
        "pelvis_angle_z",
    ]
    angles = pd.read_csv(ANGLE_PATH, usecols=angle_columns)
    poi = pd.read_csv(POI_PATH, usecols=["session_pitch", "pitch_speed_mph"])
    speed = poi.dropna().drop_duplicates("session_pitch")

    rows: list[dict[str, float | int | str]] = []
    for session_pitch, group in angles.groupby("session_pitch", sort=False):
        group = group.sort_values("time")
        pkh = unique_event(group, "pkh_time", str(session_pitch))
        fp = unique_event(group, "fp_poi_time", str(session_pitch))
        if not pkh < fp:
            raise ValueError(f"{session_pitch}: expected pkh_time < fp_poi_time")

        window = group.loc[
            (group["time"] >= pkh) & (group["time"] <= fp),
            ["time", "pelvis_angle_z"],
        ].dropna()
        if len(window) < 2:
            raise ValueError(f"{session_pitch}: fewer than two PKH-to-FP angle samples")

        times = window["time"].to_numpy()
        angles_z = window["pelvis_angle_z"].to_numpy()
        upward = np.flatnonzero((angles_z[:-1] < 0) & (angles_z[1:] >= 0))
        if len(upward) == 0:
            raise ValueError(f"{session_pitch}: no negative-to-nonnegative 0-degree crossing")
        crossings = [
            crossing_time(times[i], angles_z[i], times[i + 1], angles_z[i + 1])
            for i in upward
        ]

        pkh_angle = float(np.interp(pkh, times, angles_z))
        fp_angle = float(np.interp(fp, times, angles_z))
        rows.append(
            {
                "session_pitch": session_pitch,
                "pitcher": str(session_pitch).split("_")[0],
                "pkh_time": pkh,
                "fp_poi_time": fp,
                "pkh_fp_duration": fp - pkh,
                "pelvis_angle_z_pkh": pkh_angle,
                "pelvis_angle_z_fp": fp_angle,
                "zero_crossing_count": len(crossings),
                "first_zero_crossing_from_fp": crossings[0] - fp,
                "stable_zero_crossing_from_fp": crossings[-1] - fp,
                "stable_zero_crossing_phase_pct": (crossings[-1] - pkh) / (fp - pkh) * 100,
            }
        )

    metrics = pd.DataFrame(rows).merge(speed, on="session_pitch", how="inner", validate="one_to_one")
    if len(metrics) != len(rows):
        raise ValueError("pitch-speed merge did not retain every analyzed pitch")

    repeated = metrics[metrics["zero_crossing_count"] > 1]
    single = metrics[metrics["zero_crossing_count"] == 1]
    q25, q75 = metrics["pitch_speed_mph"].quantile([0.25, 0.75])
    low = metrics[metrics["pitch_speed_mph"] <= q25]
    high = metrics[metrics["pitch_speed_mph"] >= q75]
    stable_test = stats.ttest_ind(
        high["stable_zero_crossing_from_fp"],
        low["stable_zero_crossing_from_fp"],
        equal_var=False,
    )
    single_low = single[single["pitch_speed_mph"] <= q25]
    single_high = single[single["pitch_speed_mph"] >= q75]
    first_single_test = stats.ttest_ind(
        single_high["first_zero_crossing_from_fp"],
        single_low["first_zero_crossing_from_fp"],
        equal_var=False,
    )

    pitcher_means = metrics.groupby("pitcher", as_index=False).agg(
        stable_zero_crossing_from_fp=("stable_zero_crossing_from_fp", "mean"),
        pitch_speed_mph=("pitch_speed_mph", "mean"),
    )

    duration_reg = regression(metrics["pkh_fp_duration"], metrics["pitch_speed_mph"])
    stable_reg = regression(metrics["stable_zero_crossing_from_fp"], metrics["pitch_speed_mph"])
    pitcher_reg = regression(
        pitcher_means["stable_zero_crossing_from_fp"], pitcher_means["pitch_speed_mph"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "pelvis_zero_crossing_metrics.csv", index=False, encoding="utf-8-sig")
    repeated.to_csv(OUT_DIR / "repeated_zero_crossings.csv", index=False, encoding="utf-8-sig")

    report = f"""# Pelvis 0-degree crossing analysis

FP definition: `fp_poi_time` only.

Stable crossing definition: the last negative-to-nonnegative crossing of `pelvis_angle_z = 0` between `pkh_time` and `fp_poi_time`, linearly interpolated between adjacent 360 Hz samples. This is an orientation threshold, not the physical onset of rotation.

## Coverage and validation

- Analyzed pitches: {len(metrics)}
- Unique pitchers inferred from the `session_pitch` prefix: {metrics['pitcher'].nunique()}
- PKH angle below 0 degrees: {(metrics['pelvis_angle_z_pkh'] < 0).sum()} / {len(metrics)}
- FP angle at or above 0 degrees: {(metrics['pelvis_angle_z_fp'] >= 0).sum()} / {len(metrics)}
- Pitches with exactly one upward crossing: {len(single)}
- Pitches with repeated upward crossings: {len(repeated)}
- Missing crossings: 0

## Stable 0-degree crossing

- All-pitch median relative to FP: {metrics['stable_zero_crossing_from_fp'].median() * 1000:.1f} ms
- Low quartile (<= {q25:.1f} mph): n = {len(low)}, mean = {low['stable_zero_crossing_from_fp'].mean() * 1000:.1f} ms
- High quartile (>= {q75:.1f} mph): n = {len(high)}, mean = {high['stable_zero_crossing_from_fp'].mean() * 1000:.1f} ms
- High-minus-low mean difference: {(high['stable_zero_crossing_from_fp'].mean() - low['stable_zero_crossing_from_fp'].mean()) * 1000:.1f} ms; Welch p = {stable_test.pvalue:.6f}
- Pitch-level regression: n = {stable_reg['n']}, r = {stable_reg['r']:.4f}, R2 = {stable_reg['r2']:.4f}, p = {stable_reg['p']:.6f}
- Pitcher-mean sensitivity: n = {pitcher_reg['n']}, r = {pitcher_reg['r']:.4f}, R2 = {pitcher_reg['r2']:.4f}, p = {pitcher_reg['p']:.6f}

## First-crossing sensitivity

- After excluding the {len(repeated)} repeated-crossing pitches, low-quartile mean = {single_low['first_zero_crossing_from_fp'].mean() * 1000:.1f} ms and high-quartile mean = {single_high['first_zero_crossing_from_fp'].mean() * 1000:.1f} ms.
- High-minus-low difference = {(single_high['first_zero_crossing_from_fp'].mean() - single_low['first_zero_crossing_from_fp'].mean()) * 1000:.1f} ms; Welch p = {first_single_test.pvalue:.6f}.
- Therefore the apparent first-crossing association is driven by the repeated-crossing cases and is not retained as the main result.

## PKH-to-FP duration recheck

- Definition: `fp_poi_time - pkh_time`.
- n = {duration_reg['n']}, r = {duration_reg['r']:.4f}, R2 = {duration_reg['r2']:.6f}, slope = {duration_reg['slope']:.4f} mph/s, p = {duration_reg['p']:.6f}.
- This recheck does not support a relationship between PKH-to-FP duration and pitch speed.
"""
    (OUT_DIR / "pelvis_zero_crossing_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
