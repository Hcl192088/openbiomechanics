"""Rear-leg posture (more rear hip flexion, sinking lower) vs pitch speed.

Window: start = pkh_time, end = fp_poi_time (stride/drop phase), plus values at fp_poi_time.
Metrics per pitch:
  rear_hip_flex_max_pkh_fp : max rear hip flexion PKH->FP, deg (rear_hip_angle_x, flexion negative -> sign flipped)
  rear_hip_flex_fp         : rear hip flexion at FP, deg
  rear_knee_flex_max_pkh_fp, rear_knee_flex_fp : rear_knee_angle_x (sign checked: flexion must be positive)
  pelvis_drop_pkh_fp       : (hip-midpoint z at PKH - min hip-midpoint z PKH->FP) / standing height
  com_drop_pkh_fp          : same for centerofmass_z
Also POI max_rear_hip_flexion (cross-check), rear_grf_mag_max, rear_grf_x_max.
Stats: athlete-level partial r with speed (controlling mass); mixed model
speed ~ x_within + x_between + mass + (1|session).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "rear_leg_posture_speed_outputs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fs = ROOT / "data/full_sig"
    a = pd.read_csv(fs / "joint_angles.csv", usecols=["session_pitch", "time", "pkh_time", "fp_poi_time",
                                                       "rear_hip_angle_x", "rear_knee_angle_x"])
    lm = pd.read_csv(fs / "landmarks.csv", usecols=["session_pitch", "time", "lead_hip_z", "rear_hip_z", "centerofmass_z"])
    d = a.merge(lm, on=["session_pitch", "time"], validate="one_to_one")
    if len(d) != len(a):
        raise ValueError("angle/landmark merge lost rows")
    d["pelvis_z"] = (d["lead_hip_z"] + d["rear_hip_z"]) / 2
    w = d[(d["time"] >= d["pkh_time"]) & (d["time"] <= d["fp_poi_time"])]
    if w["session_pitch"].nunique() != d["session_pitch"].nunique():
        raise ValueError("some pitches have no PKH->FP frames")

    rows = []
    for sp, g in w.groupby("session_pitch"):
        g = g.sort_values("time")
        fp = g.iloc[-1]
        pk = g.iloc[0]
        rows.append({"session_pitch": sp,
                     "rear_hip_flex_max_pkh_fp": -g["rear_hip_angle_x"].min(),
                     "rear_hip_flex_fp": -fp["rear_hip_angle_x"],
                     "rear_knee_flex_max_pkh_fp": g["rear_knee_angle_x"].max(),
                     "rear_knee_flex_fp": fp["rear_knee_angle_x"],
                     "pelvis_drop_m": pk["pelvis_z"] - g["pelvis_z"].min(),
                     "com_drop_m": pk["centerofmass_z"] - g["centerofmass_z"].min()})
    p = pd.DataFrame(rows)
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg", "session_height_m"])
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "pitch_speed_mph", "max_rear_hip_flexion",
                                                                 "rear_grf_mag_max", "rear_grf_x_max"])
    p = p.merge(meta, on="session_pitch", validate="one_to_one").merge(poi, on="session_pitch", validate="one_to_one")
    p["pelvis_drop_pkh_fp"] = p["pelvis_drop_m"] / p["session_height_m"]
    p["com_drop_pkh_fp"] = p["com_drop_m"] / p["session_height_m"]

    checks = {"pitches": len(p), "athletes": p["session"].nunique(),
              "median_rear_knee_flex_fp": p["rear_knee_flex_fp"].median(),
              "median_rear_hip_flex_max": p["rear_hip_flex_max_pkh_fp"].median(),
              "r_hipflex_vs_poi_max_rear_hip_flexion": p[["rear_hip_flex_max_pkh_fp", "max_rear_hip_flexion"]].corr().iloc[0, 1],
              "median_pelvis_drop_pct_height": 100 * p["pelvis_drop_pkh_fp"].median()}
    if checks["median_rear_knee_flex_fp"] <= 0:
        raise ValueError("rear_knee_angle_x flexion sign assumption failed")
    p.to_csv(OUT / "per_pitch.csv", index=False)

    xs = ["rear_hip_flex_max_pkh_fp", "rear_hip_flex_fp", "rear_knee_flex_max_pkh_fp", "rear_knee_flex_fp",
          "pelvis_drop_pkh_fp", "com_drop_pkh_fp", "rear_grf_mag_max", "rear_grf_x_max"]
    ath = p.groupby("session").mean(numeric_only=True)
    X = np.c_[np.ones(len(ath)), ath["session_mass_kg"]]
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    out = []
    for x in xs:
        dd = p.dropna(subset=[x]).copy()
        dd["xb"] = dd.groupby("session")[x].transform("mean")
        dd["xw"] = dd[x] - dd["xb"]
        fit = smf.mixedlm("pitch_speed_mph ~ xw + xb + session_mass_kg", dd, groups=dd["session"]).fit(reml=True)
        a_ = ath[[x, "pitch_speed_mph"]].dropna()
        Xa = np.c_[np.ones(len(a_)), ath.loc[a_.index, "session_mass_kg"]]
        ra = lambda v: v - Xa @ np.linalg.lstsq(Xa, v, rcond=None)[0]
        out.append({"metric": x,
                    "raw_r_pitch": dd[[x, "pitch_speed_mph"]].corr().iloc[0, 1],
                    "athlete_partial_r_mass": np.corrcoef(ra(a_[x].values), ra(a_["pitch_speed_mph"].values))[0, 1],
                    "within_mph_per_sd": fit.params["xw"] * dd["xw"].std(), "within_p": fit.pvalues["xw"]})
    out = pd.DataFrame(out)
    pd.Series(checks).to_csv(OUT / "checks.csv", header=["value"])
    out.to_csv(OUT / "rear_leg_vs_speed.csv", index=False)
    print({k: round(float(v), 3) for k, v in checks.items()})
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
