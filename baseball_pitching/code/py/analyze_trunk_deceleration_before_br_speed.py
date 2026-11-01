"""Does the trunk "successfully decelerate" before BR, and does that relate to pitch speed?

Window: start = fp_poi_time, end = BR_time. Trunk angular speed = |torso_velo_z| (axial rotation;
abs() so left- and right-handers share a sign, as in 肩部STP分解驗證).

Per pitch:
  omega_peak         peak |torso_velo_z| in FP-BR (deg/s)
  peak_to_br_ms      time from that peak to BR
  omega_br           |torso_velo_z| at BR
  decel_fraction     1 - omega_br / omega_peak  (share of peak speed lost by BR)
  max_decel          largest deceleration, max of -d|omega|/dt between peak and BR (deg/s^2)
  arm_peak_ms        time of peak |shoulder_velo_z| (internal rotation) relative to BR, searched
                     FP to BR + 50 ms; torso_to_arm_gap_ms = arm peak time - torso peak time
The prior between-pitcher result (能量轉移與軀幹旋轉: omega_BR^2 unrelated to speed after peak) is
re-used as context; this script adds the within-pitcher test, max deceleration rate under the
current FP definition, sequencing gap, and the link to BR torso rotation angle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "baseball_pitching" / "data"
OUTPUT = Path(__file__).with_name("trunk_deceleration_before_br_metrics.csv")
ARM_SEARCH_AFTER_BR_S = 0.050


def pitch_metrics(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("time")
    fp, br = g["fp_poi_time"].iloc[0], g["BR_time"].iloc[0]
    t = g["time"].to_numpy()
    w = np.abs(g["torso_velo_z"].to_numpy())
    arm = np.abs(g["shoulder_velo_z"].to_numpy())
    win = (t >= fp) & (t <= br)
    if win.sum() < 5:
        raise RuntimeError("FP-BR window too short")
    i_peak = np.flatnonzero(win)[np.argmax(w[win])]
    omega_peak = w[i_peak]
    omega_br = float(np.interp(br, t, w))
    rate = np.gradient(w, t)
    decel_win = (t >= t[i_peak]) & (t <= br)
    max_decel = float(np.max(-rate[decel_win])) if decel_win.sum() > 1 else np.nan
    arm_win = (t >= fp) & (t <= br + ARM_SEARCH_AFTER_BR_S)
    i_arm = np.flatnonzero(arm_win)[np.argmax(arm[arm_win])]
    return pd.Series({
        "omega_peak": omega_peak,
        "peak_to_br_ms": (br - t[i_peak]) * 1000,
        "omega_br": omega_br,
        "decel_fraction": 1 - omega_br / omega_peak,
        "max_decel": max_decel,
        "arm_peak_ms": (t[i_arm] - br) * 1000,
        "torso_to_arm_gap_ms": (t[i_arm] - t[i_peak]) * 1000,
    })


def fmt(model, term: str) -> str:
    lo, hi = model.conf_int().loc[term]
    return f"{model.params[term]:+.4g} [{lo:+.4g}, {hi:+.4g}] p={model.pvalues[term]:.3g}"


def main() -> None:
    cols = ["session_pitch", "time", "fp_poi_time", "BR_time", "torso_velo_z", "shoulder_velo_z"]
    velos = pd.read_csv(DATA / "full_sig" / "joint_velos.csv", usecols=cols)
    metrics = velos.groupby("session_pitch", sort=False).apply(pitch_metrics, include_groups=False)
    if isinstance(metrics, pd.Series):
        metrics = metrics.unstack()
    metrics = metrics.reset_index()
    poi = pd.read_csv(DATA / "poi" / "poi_metrics.csv",
                      usecols=["session_pitch", "session", "pitch_speed_mph", "torso_rotation_br",
                               "max_torso_rotational_velo"])
    meta = pd.read_csv(DATA / "metadata.csv", usecols=["session_pitch", "session_mass_kg", "session_height_m"])
    data = (poi.merge(metrics, on="session_pitch", validate="one_to_one")
               .merge(meta, on="session_pitch", validate="one_to_one"))
    if len(data) != 411 or data.isna().any().any():
        raise RuntimeError(f"Expected 411 complete pitches, found {len(data)} / NaN present")
    data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    r_check = stats.pearsonr(data["omega_peak"], data["max_torso_rotational_velo"]).statistic
    print(f"n={len(data)}, sessions={data['session'].nunique()}; "
          f"omega_peak vs POI max_torso_rotational_velo r={r_check:.3f}")

    names = ["omega_peak", "peak_to_br_ms", "omega_br", "decel_fraction", "max_decel",
             "arm_peak_ms", "torso_to_arm_gap_ms"]
    print("\nDescriptives (pitch level):")
    print(data[names].describe(percentiles=[.25, .5, .75]).T[["mean", "std", "25%", "50%", "75%"]].round(3).to_string())
    print(f"pitches with omega_br > 95% of peak (essentially no deceleration): "
          f"{int((data['decel_fraction'] < 0.05).sum())}")

    cl = {"cov_type": "cluster", "cov_kwds": {"groups": data["session"]}}
    pm = data.groupby("session")[names + ["pitch_speed_mph", "torso_rotation_br", "session_mass_kg",
                                         "session_height_m"]].mean()
    print("\nSpeed models (z-scored predictor; mph per SD):")
    for c in names:
        data["_z"] = (data[c] - data[c].mean()) / data[c].std()
        pm["_z"] = (pm[c] - pm[c].mean()) / pm[c].std()
        r = stats.pearsonr(pm[c], pm["pitch_speed_mph"])
        between = smf.ols("pitch_speed_mph ~ _z + session_mass_kg + session_height_m", data=pm).fit()
        line = (f"  {c:20s} pitcher-mean r={r.statistic:+.3f} p={r.pvalue:.3g} | "
                f"+mass/height {fmt(between, '_z')}")
        if c != "omega_peak":
            pm["_peak"] = pm["omega_peak"]
            ctrl = smf.ols("pitch_speed_mph ~ _z + _peak + session_mass_kg + session_height_m", data=pm).fit()
            line += f" | +peak {fmt(ctrl, '_z')}"
        within = smf.ols("pitch_speed_mph ~ _z + C(session)", data=data).fit(**cl)
        within_peak = smf.ols("pitch_speed_mph ~ _z + omega_peak + C(session)", data=data).fit(**cl)
        line += f" | within {fmt(within, '_z')}"
        if c != "omega_peak":
            line += f" | within+peak {fmt(within_peak, '_z')}"
        print(line)

    print("\nLink to BR torso rotation angle (does less rotation at BR = more deceleration?):")
    for c in ["decel_fraction", "max_decel", "peak_to_br_ms", "omega_br"]:
        r = stats.pearsonr(pm[c], pm["torso_rotation_br"])
        within = smf.ols(f"torso_rotation_br ~ {c} + C(session)", data=data).fit(**cl)
        print(f"  {c:16s} pitcher-mean r={r.statistic:+.3f} p={r.pvalue:.3g} | within {fmt(within, c)}")


if __name__ == "__main__":
    main()
