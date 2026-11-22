"""Decompose between-athlete variance of positive shoulder transfer (FP->BR) and
test whether thorax-limited STP energy tracks peak trunk speed or retained BR speed.

Window: start = fp_poi_time, end = BR_time.

Per frame, gate = (STP + JFP) > 0. The positive total integral of max(STP+JFP, 0)
is split exactly into gated components:
    jfp                     JFP on gated frames
    stp_thorax_limited      STP on gated frames where thorax/arm STP have opposite
                            sign, STP > 0, and |thorax_stp| <= |arm_stp|
    stp_arm_limited         same, but |arm_stp| < |thorax_stp|
    stp_other               remaining gated STP (sign condition not met)
    thorax_stp = thorax_dist_seg_pwr + JFP ; arm_stp = upper_arm_prox_seg_pwr - JFP

Trunk speed = |torso_velo_z| (deg/s): omega_peak = FP-BR max, omega_br = value at BR,
omega_thorax_limited = mean on thorax-limited frames.
Athlete level = pitches averaged within `session` (100 athletes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("shoulder_transfer_component_variance_outputs")
COMPONENTS = ["jfp", "stp_thorax_limited", "stp_arm_limited", "stp_other"]


def summarize(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("time")
    fp = g["fp_poi_time"].dropna().unique()
    br = g["BR_time"].dropna().unique()
    if len(fp) != 1 or len(br) != 1:
        raise ValueError(f"Ambiguous FP/BR for {g.name}")
    t_all = g["time"].to_numpy(float)
    w_all = np.abs(g["torso_velo_z"].to_numpy(float))
    win = (t_all >= fp[0]) & (t_all <= br[0])
    w = g[win]
    t = t_all[win]
    stp = w["shoulder_energy_transfer_stp"].to_numpy(float)
    jfp = w["shoulder_energy_transfer_jfp"].to_numpy(float)
    gate = (stp + jfp) > 0
    thorax_stp = w["thorax_dist_seg_pwr"].to_numpy(float) + jfp
    arm_stp = w["upper_arm_prox_seg_pwr"].to_numpy(float) - jfp
    valid = (thorax_stp * arm_stp < 0) & (stp > 0)
    thorax_lim = gate & valid & (np.abs(thorax_stp) <= np.abs(arm_stp))
    arm_lim = gate & valid & ~(np.abs(thorax_stp) <= np.abs(arm_stp))
    other = gate & ~valid

    def integ(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.trapezoid(np.where(mask, values, 0.0), t))

    omega = w_all[win]
    return pd.Series({
        "positive_total_j": float(np.trapezoid(np.maximum(stp + jfp, 0.0), t)),
        "jfp": integ(jfp, gate),
        "stp_thorax_limited": integ(stp, thorax_lim),
        "stp_arm_limited": integ(stp, arm_lim),
        "stp_other": integ(stp, other),
        "thorax_limited_duration_s": float(thorax_lim.sum() * np.median(np.diff(t))),
        "omega_peak": float(omega.max()),
        "omega_br": float(np.interp(br[0], t_all, w_all)),
        "omega_thorax_limited": float(omega[thorax_lim].mean()) if thorax_lim.any() else np.nan,
    })


def main() -> None:
    OUT.mkdir(exist_ok=True)
    e_cols = ["session_pitch", "time", "fp_poi_time", "BR_time",
              "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
              "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr"]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=e_cols)
    velo = pd.read_csv(ROOT / "data/full_sig/joint_velos.csv", usecols=["session_pitch", "time", "torso_velo_z"])
    frame = energy.merge(velo, on=["session_pitch", "time"], how="inner", validate="one_to_one")
    if len(frame) != len(energy):
        raise ValueError(f"Exact time-key merge lost rows: {len(energy)} -> {len(frame)}")

    pitch = frame.groupby("session_pitch", sort=False).apply(summarize, include_groups=False).reset_index()
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv",
                      usecols=["session_pitch", "shoulder_transfer_fp_br", "pitch_speed_mph"])
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    pitch = pitch.merge(poi, on="session_pitch", validate="one_to_one").merge(
        meta, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411 or pitch["session"].nunique() != 100:
        raise ValueError("Expected 411 pitches and 100 sessions")
    max_err = float((pitch[COMPONENTS].sum(axis=1) - pitch["positive_total_j"]).abs().max())
    official_r = float(pitch["positive_total_j"].corr(pitch["shoulder_transfer_fp_br"]))
    pitch.to_csv(OUT / "per_pitch.csv", index=False)

    a = pitch.groupby("session").mean(numeric_only=True)
    a.to_csv(OUT / "per_athlete.csv")

    design = np.column_stack([np.ones(len(a)), a["session_mass_kg"]])
    res = lambda v: np.asarray(v, float) - design @ np.linalg.lstsq(design, np.asarray(v, float), rcond=None)[0]
    total = a["positive_total_j"]
    total_res = res(total)
    speed_res = res(a["pitch_speed_mph"])
    var_rows = []
    for c in COMPONENTS:
        part_res = res(a[c])
        var_rows.append({
            "component": c,
            "mean_j": a[c].mean(),
            "share_of_mean_total": a[c].mean() / total.mean(),
            "share_of_total_variance": np.cov(a[c], total)[0, 1] / total.var(),
            "share_of_total_variance_mass_adjusted": np.cov(part_res, total_res)[0, 1] / total_res.var(ddof=1),
            "r_total": a[c].corr(total),
            "partial_r_total_mass": float(np.corrcoef(part_res, total_res)[0, 1]),
            "r_pitch_speed": a[c].corr(a["pitch_speed_mph"]),
            "partial_r_pitch_speed_mass": float(np.corrcoef(part_res, speed_res)[0, 1]),
        })
    variance = pd.DataFrame(var_rows)
    variance.to_csv(OUT / "component_variance.csv", index=False)

    z = a.copy()
    for c in ["omega_peak", "omega_br", "session_mass_kg"]:
        z[c] = (z[c] - z[c].mean()) / z[c].std()
    model_rows = []
    for y in ["stp_thorax_limited", "positive_total_j", "thorax_limited_duration_s", "omega_thorax_limited"]:
        fit = smf.ols(f"{y} ~ omega_peak + omega_br + session_mass_kg", z).fit(cov_type="HC3")
        for term in ["omega_peak", "omega_br", "session_mass_kg"]:
            model_rows.append({"outcome": y, "term": term,
                               "beta_per_sd": fit.params[term],
                               "std_beta": fit.params[term] / z[y].std(),
                               "p_hc3": fit.pvalues[term], "r2": fit.rsquared})
    models = pd.DataFrame(model_rows)
    models.to_csv(OUT / "trunk_speed_models.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"max decomposition error J: {max_err:.2e}; pitch-level r with official total: {official_r:.4f}")
    print(f"athlete r(omega_peak, omega_br) = {a['omega_peak'].corr(a['omega_br']):.3f}")
    print(variance.round(3).to_string(index=False))
    print()
    print(models.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
