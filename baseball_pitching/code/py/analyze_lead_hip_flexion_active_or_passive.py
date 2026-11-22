"""Is the pelvis tilting forward over the lead femur (lead hip flexion, FP->BR) driven by
the lead hip muscles (concentric hip-flexor work) or resisted by them (eccentric
hip-extensor work while something else drives the motion)?

Window: start = fp_poi_time, end = BR_time.

Per-axis lead hip joint power (thigh frame; Visual3D LT/RT_HIP_ANGULAR_VELOCITY is
thigh-relative-to-pelvis resolved in the thigh frame, as is lead_hip_lead_thigh_moment):
    P_axis = s_axis * M_axis * omega_axis (rad/s)
Signs s chosen so that sum over axes matches official `lead_hip_energy_generated`
(validated: R throwers s = (-1, +1, -1), L throwers s = (-1, -1, -1); y differs because
left-handed data are mirrored). The x axis (flexion/extension) sign is the same for both.
P_x > 0: moment and flexion/extension velocity in the same direction (muscles drive);
P_x < 0: moment opposes the motion (muscles brake).
Flexion direction of omega_x is identified from its correlation with d(lead_hip_angle_x)/dt
(lead_hip_angle_x: flexion negative).

Per pitch: positive / negative sagittal work FP->BR, work during hip-flexing frames only,
share of hip-flexing frames with a flexor-direction moment.
Athlete level (100): relation to the pelvis-tilt JFP part
(forward_lean_pelvis_vs_lumbar_outputs/per_pitch.csv) and pitch speed, partial r on mass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "lead_hip_flexion_active_or_passive_outputs"
SIGNS = {"R": np.array([-1.0, 1.0, -1.0]), "L": np.array([-1.0, -1.0, -1.0])}


def residualize(df: pd.DataFrame, col: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df["session_mass_kg"].to_numpy(float)])
    y = df[col].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fs = ROOT / "data" / "full_sig"
    e = pd.read_csv(fs / "energy_flow.csv", usecols=["session_pitch", "time", "fp_poi_time", "BR_time",
                                                     "lead_hip_energy_generated"])
    f = pd.read_csv(fs / "forces_moments.csv", usecols=["session_pitch", "time",
                                                        *[f"lead_hip_lead_thigh_moment_{a}" for a in "xyz"]])
    v = pd.read_csv(fs / "joint_velos.csv", usecols=["session_pitch", "time", *[f"lead_hip_velo_{a}" for a in "xyz"]])
    a = pd.read_csv(fs / "joint_angles.csv", usecols=["session_pitch", "time", "lead_hip_angle_x"])
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "p_throws", "pitch_speed_mph"])
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    d = (e.merge(f, on=["session_pitch", "time"], validate="one_to_one")
          .merge(v, on=["session_pitch", "time"], validate="one_to_one")
          .merge(a, on=["session_pitch", "time"], validate="one_to_one")
          .merge(poi[["session_pitch", "p_throws"]], on="session_pitch", validate="many_to_one"))
    if len(d) != len(e):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    d = d.sort_values(["session_pitch", "time"])
    d["hip_x_rate"] = d.groupby("session_pitch")["lead_hip_angle_x"].transform(
        lambda s: np.gradient(s.to_numpy(float), d.loc[s.index, "time"].to_numpy(float)))
    w = d[(d["time"] >= d["fp_poi_time"]) & (d["time"] <= d["BR_time"])].copy()
    n_window = len(w)
    w = w.dropna()
    M = w[[f"lead_hip_lead_thigh_moment_{x}" for x in "xyz"]].to_numpy(float)
    W = np.radians(w[[f"lead_hip_velo_{x}" for x in "xyz"]].to_numpy(float))
    S = np.vstack([SIGNS[h] for h in w["p_throws"]])
    P_axes = S * M * W
    w["P_x"] = P_axes[:, 0]
    P_sum = P_axes.sum(axis=1)

    # flexion direction of omega_x: lead_hip_angle_x decreases with flexion
    r_dir = float(np.corrcoef(W[:, 0], w["hip_x_rate"])[0, 1])
    flex_sign = -1.0 if r_dir > 0 else 1.0     # omega_x * flex_sign > 0  <=> flexing
    w["flexing"] = (W[:, 0] * flex_sign) > 0

    checks = pd.Series({
        "window_rows": n_window, "rows_after_dropna": len(w),
        "pitches_after_dropna": w["session_pitch"].nunique(),
        "r_axis_sum_vs_official_power": float(np.corrcoef(P_sum, w["lead_hip_energy_generated"])[0, 1]),
        "mae_axis_sum_vs_official_w": float(np.mean(np.abs(P_sum - w["lead_hip_energy_generated"]))),
        "r_omega_x_vs_hip_angle_x_rate": r_dir,
        "share_frames_flexing": float(w["flexing"].mean()),
    })

    rows = []
    for sp, g in w.groupby("session_pitch", sort=False):
        t = g["time"].to_numpy(float)
        px = g["P_x"].to_numpy(float)
        flex = g["flexing"].to_numpy(bool)
        integ = lambda vals: float(np.trapezoid(vals, t))
        rows.append({
            "session_pitch": sp,
            "sag_work_pos_j": integ(np.maximum(px, 0)),
            "sag_work_neg_j": integ(np.minimum(px, 0)),
            "sag_work_net_j": integ(px),
            "flexing_work_pos_j": integ(np.where(flex, np.maximum(px, 0), 0)),
            "flexing_work_neg_j": integ(np.where(flex, np.minimum(px, 0), 0)),
            "share_flexing_frames_driving": float((px[flex] > 0).mean()) if flex.any() else np.nan,
            "official_hip_work_net_j": integ(g["lead_hip_energy_generated"].to_numpy(float)),
        })
    pitch = pd.DataFrame(rows).merge(meta, on="session_pitch").merge(
        poi[["session_pitch", "pitch_speed_mph"]], on="session_pitch")
    tilt = pd.read_csv(HERE / "forward_lean_pelvis_vs_lumbar_outputs/per_pitch.csv",
                       usecols=["session_pitch", "jfp_pelvis_tilt", "vx_pelvis_tilt", "lead_hip_angle_x_change_fp_br"])
    pitch = pitch.merge(tilt, on="session_pitch", how="left", validate="one_to_one")
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    ath = pitch.groupby("session").mean(numeric_only=True)
    ath.to_csv(OUT / "per_athlete.csv")

    targets = ["jfp_pelvis_tilt", "vx_pelvis_tilt", "pitch_speed_mph"]
    res_t = {y: residualize(ath, y) for y in targets}
    assoc = []
    for x in ["sag_work_pos_j", "sag_work_neg_j", "sag_work_net_j", "flexing_work_pos_j",
              "flexing_work_neg_j", "share_flexing_frames_driving"]:
        row = {"metric": x, "athlete_mean": ath[x].mean(), "athlete_sd": ath[x].std()}
        xr = residualize(ath, x)
        for y in targets:
            row[f"partial_r_{y}"] = float(np.corrcoef(xr, res_t[y])[0, 1])
        assoc.append(row)
    assoc = pd.DataFrame(assoc)
    checks.to_csv(OUT / "checks.csv", header=["value"])
    assoc.to_csv(OUT / "sagittal_hip_work.csv", index=False)
    pd.set_option("display.width", 250)
    print(checks.round(4).to_string())
    print()
    print(assoc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
