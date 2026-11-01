"""Arm position relative to the shoulder plane of rotation at ball release (BR) vs pitch speed.

Definitions (all evaluated at BR_time, landmarks interpolated linearly):
  s  = shoulder_jc - glove_shoulder_jc            (shoulder line, toward throwing side)
  t  = thorax_prox - thorax_dist                  (trunk long axis, upward)
  Plane A ("anatomical"): contains s, normal = t made orthogonal to s.
  Plane B ("kinematic"):  plane swept by the shoulder line, normal = s x ds/dt
                          (oriented so normal . t > 0).
  Arm vectors from throwing shoulder: upper arm (elbow_jc), shoulder->hand (hand_jc).
  out-of-plane angle: asin(arm . n); + = toward head side of the plane.
  in-plane angle: angle of the projected arm from the extended shoulder line;
                  + = ahead of the line in the shoulder line's rotation direction.
  line deviation: unsigned 3D angle between arm and the extended shoulder line.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "baseball_pitching" / "data"
OUTPUT = Path(__file__).with_name("arm_plane_of_rotation_br_metrics.csv")

POINTS = ["shoulder_jc", "glove_shoulder_jc", "elbow_jc", "hand_jc", "thorax_prox", "thorax_dist"]
ARMS = {"upperarm": "elbow_jc", "hand": "hand_jc"}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def pitch_metrics(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time")
    br = group["BR_time"].iloc[0]
    time = group["time"].to_numpy()
    if pd.isna(br) or br < time[0] or br > time[-1]:
        return pd.Series(dtype=float)

    def at_br(values: np.ndarray) -> np.ndarray:
        return np.array([np.interp(br, time, values[:, k]) for k in range(3)])

    xyz = {p: group[[f"{p}_x", f"{p}_y", f"{p}_z"]].to_numpy() for p in POINTS}
    line_series = xyz["shoulder_jc"] - xyz["glove_shoulder_jc"]
    line_rate_series = np.gradient(line_series, time, axis=0)

    pos = {p: at_br(v) for p, v in xyz.items()}
    s = unit(at_br(line_series))
    ds = at_br(line_rate_series)
    trunk = pos["thorax_prox"] - pos["thorax_dist"]

    normal_a = unit(trunk - (trunk @ s) * s)
    normal_b = unit(np.cross(s, ds))
    if normal_b @ trunk < 0:
        normal_b = -normal_b

    out = {"plane_tilt_ab_deg": float(np.degrees(np.arccos(np.clip(normal_a @ normal_b, -1, 1))))}
    for plane, n in {"A": normal_a, "B": normal_b}.items():
        ahead = np.cross(n, s)
        if ahead @ ds < 0:
            ahead = -ahead
        for arm, point in ARMS.items():
            a = unit(pos[point] - pos["shoulder_jc"])
            out[f"{arm}_out_of_plane_{plane}_deg"] = float(np.degrees(np.arcsin(np.clip(a @ n, -1, 1))))
            out[f"{arm}_in_plane_{plane}_deg"] = float(np.degrees(np.arctan2(a @ ahead, a @ s)))
    for arm, point in ARMS.items():
        a = unit(pos[point] - pos["shoulder_jc"])
        out[f"{arm}_line_deviation_deg"] = float(np.degrees(np.arccos(np.clip(a @ s, -1, 1))))
    return pd.Series(out)


def fit(formula: str, data: pd.DataFrame, term: str) -> str:
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["session"]})
    lo, hi = model.conf_int().loc[term]
    return f"{model.params[term]:+.3f} mph/deg [{lo:+.3f}, {hi:+.3f}], p={model.pvalues[term]:.3g}"


def main() -> None:
    cols = ["session_pitch", "time", "BR_time"] + [f"{p}_{a}" for p in POINTS for a in "xyz"]
    landmarks = pd.read_csv(DATA / "full_sig" / "landmarks.csv", usecols=cols)
    metrics = landmarks.groupby("session_pitch", sort=False).apply(pitch_metrics, include_groups=False)
    if isinstance(metrics, pd.Series):
        metrics = metrics.unstack()
    metrics = metrics.reset_index()

    joint = pd.read_csv(DATA / "full_sig" / "joint_angles.csv",
                        usecols=["session_pitch", "time", "BR_time", "shoulder_angle_x", "shoulder_angle_y"])
    def joint_at_br(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        br = g["BR_time"].iloc[0]
        return pd.Series({c + "_br": np.interp(br, g["time"], g[c]) for c in ["shoulder_angle_x", "shoulder_angle_y"]})
    joint_br = joint.groupby("session_pitch", sort=False).apply(joint_at_br, include_groups=False).reset_index()

    poi = pd.read_csv(DATA / "poi" / "poi_metrics.csv", usecols=["session_pitch", "session", "pitch_speed_mph"])
    meta = pd.read_csv(DATA / "metadata.csv", usecols=["session_pitch", "session_mass_kg", "session_height_m"])
    data = (poi.merge(metrics, on="session_pitch", how="inner", validate="one_to_one")
               .merge(joint_br, on="session_pitch", how="left", validate="one_to_one")
               .merge(meta, on="session_pitch", how="left", validate="one_to_one")
               .dropna())
    data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"n pitches={len(data)}, n sessions={data['session'].nunique()}")
    print(f"plane A vs B tilt at BR: median={data['plane_tilt_ab_deg'].median():.1f} deg, "
          f"IQR=[{data['plane_tilt_ab_deg'].quantile(.25):.1f}, {data['plane_tilt_ab_deg'].quantile(.75):.1f}]")
    print("Validation vs joint angles (Pearson r):")
    for c in ["upperarm_out_of_plane_A_deg", "upperarm_in_plane_A_deg"]:
        for j in ["shoulder_angle_x_br", "shoulder_angle_y_br"]:
            print(f"  {c} vs {j}: r={stats.pearsonr(data[c], data[j]).statistic:+.3f}")

    metric_cols = [c for c in metrics.columns if c not in ("session_pitch", "plane_tilt_ab_deg")]
    pitcher_means = data.groupby("session")[metric_cols + ["pitch_speed_mph"]].mean()
    for c in metric_cols:
        x = data[c]
        print(f"\n== {c}: mean={x.mean():+.1f}, SD={x.std():.1f}, median={x.median():+.1f}, "
              f"IQR=[{x.quantile(.25):+.1f}, {x.quantile(.75):+.1f}]")
        print(f"  pitch-level r={stats.pearsonr(x, data['pitch_speed_mph']).statistic:+.3f}; "
              f"pitcher-mean r={stats.pearsonr(pitcher_means[c], pitcher_means['pitch_speed_mph']).statistic:+.3f}")
        print(f"  clustered:           {fit(f'pitch_speed_mph ~ {c}', data, c)}")
        print(f"  + height/mass:       {fit(f'pitch_speed_mph ~ {c} + session_height_m + session_mass_kg', data, c)}")
        print(f"  within pitcher (FE): {fit(f'pitch_speed_mph ~ {c} + C(session)', data, c)}")
        data["_abs"] = x.abs()
        print(f"  |angle| clustered:   {fit('pitch_speed_mph ~ _abs', data, '_abs')}")
        print(f"  |angle| within (FE): {fit('pitch_speed_mph ~ _abs + C(session)', data, '_abs')}")


if __name__ == "__main__":
    main()
