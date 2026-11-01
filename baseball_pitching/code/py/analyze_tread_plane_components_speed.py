"""Tread Athletics "plane of shoulder rotation" components at FP, MER and BR vs pitch speed.

Tread definition (Brewster 2019): plane through the shoulder line, perpendicular to the spine;
the elbow should be in line with the shoulders (~90 deg abduction) from landing through release,
and not drift in front of the shoulder line at MER.

Per event (FP = fp_poi_time, MER = MER_time, BR = BR_time), linear interpolation:
  shoulder_angle_y   shoulder abduction (+), validated against POI shoulder_abduction_fp
  shoulder_angle_x   horizontal abduction (+) / adduction (-)
  elbow_angle_x      elbow flexion (+) / extension (-)
  upperarm_out_of_plane  upper arm (shoulder->elbow) angle off the plane perpendicular to the
                         trunk axis that contains the shoulder line; + = head side
  upperarm_in_plane      projected upper arm angle from the extended shoulder line;
                         + = ahead of the line in its rotation direction (elbow in front)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "baseball_pitching" / "data"
OUTPUT = Path(__file__).with_name("tread_plane_components_metrics.csv")

EVENTS = {"fp": "fp_poi_time", "mer": "MER_time", "br": "BR_time"}
POINTS = ["shoulder_jc", "glove_shoulder_jc", "elbow_jc", "thorax_prox", "thorax_dist"]
JOINTS = {"shoulder_angle_y": "shoulder_abd", "shoulder_angle_x": "shoulder_hab", "elbow_angle_x": "elbow_flex"}
LABELS = {
    "shoulder_abd": "Shoulder Angle Y (Abduction (+))",
    "shoulder_hab": "Shoulder Angle X (Horizontal Ab (+)/Adduction (-))",
    "elbow_flex": "Elbow Angle X (Flexion (+)/Extension (-))",
    "upperarm_out_of_plane": "Upper arm off shoulder plane (Head side (+))",
    "upperarm_in_plane": "Upper arm vs shoulder line in plane (Ahead (+)/Behind (-))",
}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def interp(time: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, time, values[:, k]) for k in range(values.shape[1])])


def landmark_metrics(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time")
    time = group["time"].to_numpy()
    xyz = {p: group[[f"{p}_x", f"{p}_y", f"{p}_z"]].to_numpy() for p in POINTS}
    line = xyz["shoulder_jc"] - xyz["glove_shoulder_jc"]
    line_rate = np.gradient(line, time, axis=0)
    out = {}
    for event, col in EVENTS.items():
        t = group[col].iloc[0]
        if pd.isna(t) or t < time[0] or t > time[-1]:
            raise RuntimeError(f"{col} missing or outside signal")
        s = unit(interp(time, line, t))
        ds = interp(time, line_rate, t)
        trunk = interp(time, xyz["thorax_prox"] - xyz["thorax_dist"], t)
        n = unit(trunk - (trunk @ s) * s)
        ahead = np.cross(n, s)
        if ahead @ ds < 0:
            ahead = -ahead
        arm = unit(interp(time, xyz["elbow_jc"] - xyz["shoulder_jc"], t))
        out[f"upperarm_out_of_plane_{event}"] = float(np.degrees(np.arcsin(np.clip(arm @ n, -1, 1))))
        out[f"upperarm_in_plane_{event}"] = float(np.degrees(np.arctan2(arm @ ahead, arm @ s)))
    return pd.Series(out)


def joint_metrics(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time")
    time = group["time"].to_numpy()
    out = {}
    for event, col in EVENTS.items():
        t = group[col].iloc[0]
        for src, name in JOINTS.items():
            out[f"{name}_{event}"] = float(np.interp(t, time, group[src]))
    return pd.Series(out)


def per_pitch(df: pd.DataFrame, func) -> pd.DataFrame:
    result = df.groupby("session_pitch", sort=False).apply(func, include_groups=False)
    if isinstance(result, pd.Series):
        result = result.unstack()
    return result.reset_index()


def fit(formula: str, data: pd.DataFrame, term: str) -> tuple[float, float, float, float]:
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["session"]})
    lo, hi = model.conf_int().loc[term]
    return model.params[term], lo, hi, model.pvalues[term]


def fmt(res: tuple[float, float, float, float]) -> str:
    b, lo, hi, p = res
    return f"{b:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3g}"


def main() -> None:
    event_cols = list(EVENTS.values())
    lm_cols = ["session_pitch", "time"] + event_cols + [f"{p}_{a}" for p in POINTS for a in "xyz"]
    geo = per_pitch(pd.read_csv(DATA / "full_sig" / "landmarks.csv", usecols=lm_cols), landmark_metrics)
    ja_cols = ["session_pitch", "time"] + event_cols + list(JOINTS)
    ang = per_pitch(pd.read_csv(DATA / "full_sig" / "joint_angles.csv", usecols=ja_cols), joint_metrics)

    poi = pd.read_csv(DATA / "poi" / "poi_metrics.csv",
                      usecols=["session_pitch", "session", "pitch_speed_mph", "shoulder_abduction_fp", "elbow_flexion_fp"])
    meta = pd.read_csv(DATA / "metadata.csv", usecols=["session_pitch", "session_mass_kg", "session_height_m"])
    data = (poi.merge(ang, on="session_pitch", validate="one_to_one")
               .merge(geo, on="session_pitch", validate="one_to_one")
               .merge(meta, on="session_pitch", how="left", validate="one_to_one"))
    if len(data) != 411 or data.isna().any().any():
        raise RuntimeError(f"Expected 411 complete pitches, found {len(data)} / NaN present")
    for joint, poi_col in [("shoulder_abd_fp", "shoulder_abduction_fp"), ("elbow_flex_fp", "elbow_flexion_fp")]:
        diff = (data[joint] - data[poi_col]).abs().max()
        print(f"check {joint} vs POI {poi_col}: max |diff|={diff:.3f} deg")
    data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"n pitches={len(data)}, n sessions={data['session'].nunique()}")

    rows = []
    for name, label in LABELS.items():
        for event in EVENTS:
            c = f"{name}_{event}"
            x = data[c]
            pm = data.groupby("session")[[c, "pitch_speed_mph"]].mean()
            rows.append({
                "metric": label, "event": event.upper(),
                "median": x.median(), "q1": x.quantile(.25), "q3": x.quantile(.75), "sd": x.std(),
                "r_pitch": stats.pearsonr(x, data["pitch_speed_mph"]).statistic,
                "r_pitcher_mean": stats.pearsonr(pm[c], pm["pitch_speed_mph"]).statistic,
                "clustered": fmt(fit(f"pitch_speed_mph ~ {c}", data, c)),
                "adj_height_mass": fmt(fit(f"pitch_speed_mph ~ {c} + session_height_m + session_mass_kg", data, c)),
                "within_pitcher": fmt(fit(f"pitch_speed_mph ~ {c} + C(session)", data, c)),
            })
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(table.round(3).to_string(index=False))

    # Tread's target: elbow in line with the shoulders, i.e. ~90 deg abduction and 0 deg off plane.
    print("\nDistance from Tread target (|value - target|), mph per deg:")
    for event in EVENTS:
        for c, target in [(f"shoulder_abd_{event}", 90.0), (f"upperarm_out_of_plane_{event}", 0.0)]:
            data["_dist"] = (data[c] - target).abs()
            print(f"  |{c} - {target:g}|: clustered {fmt(fit('pitch_speed_mph ~ _dist', data, '_dist'))}; "
                  f"within {fmt(fit('pitch_speed_mph ~ _dist + C(session)', data, '_dist'))}")

    print("\nJoint model at BR (abduction + horizontal abduction + elbow flexion):")
    joint_model = "pitch_speed_mph ~ shoulder_abd_br + shoulder_hab_br + elbow_flex_br"
    for c in ["shoulder_abd_br", "shoulder_hab_br", "elbow_flex_br"]:
        print(f"  {c}: clustered {fmt(fit(joint_model, data, c))}; "
              f"within {fmt(fit(joint_model + ' + C(session)', data, c))}")


if __name__ == "__main__":
    main()
