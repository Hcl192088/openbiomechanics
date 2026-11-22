"""What drives the pelvis to tilt forward over the lead femur (FP->BR)?

Window: start = fp_poi_time, end = BR_time. "Tilting frames": d(pelvis_angle_x)/dt > 0
(anterior tilt increasing; pelvis_angle_x anterior-positive in both handedness because
left-handed angles are mirrored to right-handed convention).

A. Pelvis energy budget (official Visual3D segment powers, scalar, frame-invariant):
   pelvis_leadhip_seg_pwr, pelvis_rearhip_seg_pwr, pelvis_thorax_seg_pwr (power INTO the
   pelvis at each joint = JFP + STP). Integrated over tilting frames, split positive/negative.
B. Hip torque power on the pelvis tilt axis:
   pelvis-side STP at a hip = -M_hip(pelvis frame) . omega_pelvis(pelvis frame)
   (CMO.v3s: RPV_LHIP_STP = dot(-LTH::ProxEndTorque, RPV::AngVel); hip_pelvis_moment is
   LT/RT_HIP_RPV_MMT resolved in RPV; pelvis_velo is RPV vs lab resolved in RPV).
   Per-axis sign vector chosen by matching the 3-axis sum to
   (pelvis_<side>hip_seg_pwr - JFP_pelvis_side) per handedness; must reach r >= 0.95 or stop.
   JFP_pelvis_side = -<side>_hip_energy_transfer_jfp (sign also checked in the search).
   x component = torque power on pelvis tilt. Lumbar moment is not exported -> not split.
C. Momentum hypothesis: does forward COM velocity at FP (and GRF) predict the pelvis-tilt
   JFP part / forward velocity from tilt (forward_lean_pelvis_vs_lumbar_outputs)?
   Between- and within-athlete mixed models: y ~ x_within + x_between + (1|session).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "pelvis_tilt_drivers_outputs"


def fit_signs(M: np.ndarray, W: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    best = None
    for s in product([1.0, -1.0], repeat=3):
        c = (M * W * np.array(s)).sum(axis=1)
        mae = float(np.mean(np.abs(target - c)))
        if best is None or mae < best[2]:
            best = (np.array(s), float(np.corrcoef(target, c)[0, 1]), mae)
    return best[0], best[1]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fs = ROOT / "data" / "full_sig"
    e = pd.read_csv(fs / "energy_flow.csv", usecols=[
        "session_pitch", "time", "fp_poi_time", "BR_time",
        "pelvis_leadhip_seg_pwr", "pelvis_rearhip_seg_pwr", "pelvis_thorax_seg_pwr",
        "lead_hip_energy_transfer_jfp", "rear_hip_energy_transfer_jfp"])
    f = pd.read_csv(fs / "forces_moments.csv", usecols=[
        "session_pitch", "time", *[f"{s}_hip_pelvis_moment_{a}" for s in ["lead", "rear"] for a in "xyz"]])
    v = pd.read_csv(fs / "joint_velos.csv", usecols=["session_pitch", "time", *[f"pelvis_velo_{a}" for a in "xyz"]])
    a = pd.read_csv(fs / "joint_angles.csv", usecols=["session_pitch", "time", "pelvis_angle_x"])
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=[
        "session_pitch", "p_throws", "pitch_speed_mph", "lead_grf_x_max", "lead_grf_mag_max", "rear_grf_x_max"])
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    d = (e.merge(f, on=["session_pitch", "time"], validate="one_to_one")
          .merge(v, on=["session_pitch", "time"], validate="one_to_one")
          .merge(a, on=["session_pitch", "time"], validate="one_to_one")
          .merge(poi[["session_pitch", "p_throws"]], on="session_pitch", validate="many_to_one"))
    if len(d) != len(e):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    d = d.sort_values(["session_pitch", "time"])
    d["tilt_rate"] = d.groupby("session_pitch")["pelvis_angle_x"].transform(
        lambda s: np.gradient(s.to_numpy(float), d.loc[s.index, "time"].to_numpy(float)))
    w = d[(d["time"] >= d["fp_poi_time"]) & (d["time"] <= d["BR_time"])].dropna().copy()

    # ---- B: sign validation per handedness and side
    W = np.radians(w[[f"pelvis_velo_{x}" for x in "xyz"]].to_numpy(float))
    checks = {"rows": len(w), "pitches": w["session_pitch"].nunique()}
    for side in ["lead", "rear"]:
        M = w[[f"{side}_hip_pelvis_moment_{x}" for x in "xyz"]].to_numpy(float)
        sp = w[f"pelvis_{side}hip_seg_pwr"].to_numpy(float)
        jfp = w[f"{side}_hip_energy_transfer_jfp"].to_numpy(float)
        px = np.full(len(w), np.nan)
        for hand in ["R", "L"]:
            m = (w["p_throws"] == hand).to_numpy()
            best = None
            for jsign in (1.0, -1.0):          # pelvis-side JFP = jsign * transfer_jfp
                s, r = fit_signs(M[m], W[m], sp[m] - jsign * jfp[m])
                if best is None or r > best[1]:
                    best = (s, r, jsign)
            s, r, jsign = best
            checks[f"{side}_{hand}_signs"] = str(tuple(int(x) for x in s))
            checks[f"{side}_{hand}_jfp_sign"] = int(jsign)
            checks[f"{side}_{hand}_r"] = r
            if r < 0.95:
                raise ValueError(f"{side} hip STP reconstruction r = {r:.3f} < 0.95 for {hand}; stop")
            px[m] = s[0] * M[m, 0] * W[m, 0]
        w[f"{side}_hip_tilt_torque_pwr"] = px
    w["tilting"] = w["tilt_rate"] > 0
    checks["share_frames_tilting"] = float(w["tilting"].mean())
    checks["share_tilting_frames_L"] = float(w.loc[w["p_throws"] == "L", "tilting"].mean())
    checks["share_tilting_frames_R"] = float(w.loc[w["p_throws"] == "R", "tilting"].mean())

    # ---- per pitch integrals over tilting frames
    cols = {"lead_hip_sp": "pelvis_leadhip_seg_pwr", "rear_hip_sp": "pelvis_rearhip_seg_pwr",
            "lumbar_sp": "pelvis_thorax_seg_pwr", "lead_hip_tilt_torque": "lead_hip_tilt_torque_pwr",
            "rear_hip_tilt_torque": "rear_hip_tilt_torque_pwr"}
    rows = []
    for sp_id, g in w.groupby("session_pitch", sort=False):
        t = g["time"].to_numpy(float)
        tilt = g["tilting"].to_numpy(bool)
        row = {"session_pitch": sp_id}
        for k, c in cols.items():
            p = g[c].to_numpy(float)
            row[f"{k}_pos_j"] = float(np.trapezoid(np.where(tilt, np.maximum(p, 0), 0), t))
            row[f"{k}_neg_j"] = float(np.trapezoid(np.where(tilt, np.minimum(p, 0), 0), t))
            row[f"{k}_net_j"] = row[f"{k}_pos_j"] + row[f"{k}_neg_j"]
        rows.append(row)
    pitch = pd.DataFrame(rows).merge(meta, on="session_pitch").merge(poi.drop(columns="p_throws"), on="session_pitch")

    # COM forward velocity at FP and tilt outcomes from upstream outputs
    lean = pd.read_csv(HERE / "forward_lean_pelvis_vs_lumbar_outputs/per_pitch.csv",
                       usecols=["session_pitch", "jfp_pelvis_tilt", "vx_pelvis_tilt", "pelvis_angle_x_change_fp_br"])
    brake = pd.read_csv(HERE / "trunk_base_ahead_of_com_outputs/per_pitch.csv",
                        usecols=["session_pitch", "com_vx_fp", "pelvis_vx_fp", "pelvis_vx_loss_fp_br"])
    pitch = pitch.merge(lean, on="session_pitch", validate="one_to_one").merge(brake, on="session_pitch", validate="one_to_one")
    pitch.to_csv(OUT / "per_pitch.csv", index=False)

    budget = []
    for k in cols:
        budget.append({"source": k, **{f"mean_{s}_j": pitch.groupby("session")[f"{k}_{s}_j"].mean().mean()
                                       for s in ["pos", "neg", "net"]}})
    budget = pd.DataFrame(budget)

    # ---- C: mixed models (within / between)
    rows = []
    for x in ["com_vx_fp", "pelvis_vx_fp", "pelvis_vx_loss_fp_br", "lead_grf_x_max", "lead_grf_mag_max",
              "rear_grf_x_max", "lead_hip_tilt_torque_neg_j", "lumbar_sp_net_j", "rear_hip_sp_net_j"]:
        dd = pitch.dropna(subset=[x]).copy()
        dd["xb"] = dd.groupby("session")[x].transform("mean")
        dd["xw"] = dd[x] - dd["xb"]
        sd_w, sd_b = dd["xw"].std(), dd.groupby("session")[x].mean().std()
        for y in ["jfp_pelvis_tilt", "vx_pelvis_tilt"]:
            fit = smf.mixedlm(f"{y} ~ xw + xb + session_mass_kg", dd, groups=dd["session"]).fit(reml=True)
            rows.append({"predictor": x, "outcome": y, "n_pitches": len(dd),
                         "within_per_sd": fit.params["xw"] * sd_w, "within_p": fit.pvalues["xw"],
                         "between_per_sd": fit.params["xb"] * sd_b, "between_p": fit.pvalues["xb"]})
    models = pd.DataFrame(rows)

    pd.Series(checks).to_csv(OUT / "checks.csv", header=["value"])
    budget.to_csv(OUT / "pelvis_budget_tilting_frames.csv", index=False)
    models.to_csv(OUT / "tilt_driver_models.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.Series(checks).to_string())
    print()
    print(budget.round(2).to_string(index=False))
    print()
    print(models.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
