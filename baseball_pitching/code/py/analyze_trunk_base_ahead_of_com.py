"""What makes the trunk base move ahead of the whole-body COM (FP->BR)?

Follow-up of analyze_shoulder_speed_sources.py, where the term
(v_thorax_dist - v_com) . F_shoulder carried the strongest pitch-speed link.

Window: start = fp_poi_time, end = BR_time; frames gated by (STP + JFP) > 0.

1. Split  v_thorax_dist - v_com = (v_pelvis - v_com) + (v_thorax_dist - v_pelvis)
   pelvis = midpoint(rear_hip, lead_hip). Each part's JFP contribution
   = integral F_lab . part (gated), F_lab from the per-session offset
   (analyze_jfp_force_speed_alignment.py).
2. Braking check: forward (+lab x) velocity of pelvis, COM and thorax_dist;
   loss from fp_poi_time to BR_time (v_FP - v_BR, m/s).
   Lab +x is verified as forward per pitch (COM x-velocity at FP > 0).
3. Lead/rear leg GRF from POI (403 pitches with force plate data).
Athlete level = pitches averaged within `session`; partial r controls body mass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from analyze_jfp_force_speed_alignment import M0, fit_offset, prepare
from analyze_shoulder_speed_sources import basis, grad


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("trunk_base_ahead_of_com_outputs")
GRF = ["lead_grf_x_max", "lead_grf_z_max", "lead_grf_mag_max", "rear_grf_x_max", "rear_grf_mag_max"]


def pitch_metrics(g: pd.DataFrame, q: np.ndarray) -> dict:
    g = g.sort_values("time")
    t = g["time"].to_numpy(float)
    pt = lambda name: g[[f"{name}_{a}" for a in "xyz"]].to_numpy(float)
    P, com = pt("thorax_dist"), pt("centerofmass")
    pelvis = (pt("rear_hip") + pt("lead_hip")) / 2
    v_P, v_com, v_pel = grad(P, t), grad(com, t), grad(pelvis, t)
    fp, br = g["fp_poi_time"].iloc[0], g["BR_time"].iloc[0]
    vx_com_fp = float(np.interp(fp, t, v_com[:, 0]))
    if vx_com_fp <= 0:
        raise ValueError(f"{g.name}: COM x-velocity at FP <= 0; +x is not forward")

    win = (t >= fp) & (t <= br)
    jfp = g["shoulder_energy_transfer_jfp"].to_numpy(float)[win]
    stp = g["shoulder_energy_transfer_stp"].to_numpy(float)[win]
    gate = (jfp + stp) > 0
    tw = t[win]
    B = basis(pt("thorax_prox")[win], P[win], pt("thorax_ap")[win])
    Q = Rotation.from_rotvec(q).as_matrix() @ M0
    F_lab = np.einsum("nij,jk,nk->ni", B, Q, pt("shoulder_thorax_force")[win])

    def jfp_of(v: np.ndarray) -> float:
        return float(np.trapezoid(np.where(gate, np.einsum("ij,ij->i", F_lab, v[win]), 0.0), tw))

    out = {
        "jfp_trunk_base_rel_com": jfp_of(v_P - v_com),
        "jfp_pelvis_rel_com": jfp_of(v_pel - v_com),
        "jfp_trunk_base_rel_pelvis": jfp_of(v_P - v_pel),
    }
    for name, v in [("pelvis", v_pel), ("com", v_com), ("trunk_base", v_P)]:
        v_fp = float(np.interp(fp, t, v[:, 0]))
        v_br = float(np.interp(br, t, v[:, 0]))
        out[f"{name}_vx_fp"] = v_fp
        out[f"{name}_vx_br"] = v_br
        out[f"{name}_vx_loss_fp_br"] = v_fp - v_br
    return out


def partial_r(df: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    d = df[[x, y, "session_mass_kg"]].dropna()
    X = np.column_stack([np.ones(len(d)), d["session_mass_kg"]])
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return float(np.corrcoef(res(d[x].to_numpy(float)), res(d[y].to_numpy(float)))[0, 1]), len(d)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fs = ROOT / "data" / "full_sig"
    energy = pd.read_csv(fs / "energy_flow.csv", usecols=[
        "session_pitch", "time", "fp_poi_time", "BR_time",
        "shoulder_energy_transfer_jfp", "shoulder_energy_transfer_stp"])
    forces = pd.read_csv(fs / "forces_moments.csv", usecols=[
        "session_pitch", "time", *[f"shoulder_thorax_force_{a}" for a in "xyz"]])
    pts = ["shoulder_jc", "thorax_prox", "thorax_dist", "thorax_ap", "centerofmass", "rear_hip", "lead_hip"]
    landmarks = pd.read_csv(fs / "landmarks.csv", usecols=[
        "session_pitch", "time", *[f"{p}_{a}" for p in pts for a in "xyz"]])
    data = energy.merge(forces, on=["session_pitch", "time"], validate="one_to_one").merge(
        landmarks, on=["session_pitch", "time"], validate="one_to_one")
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "pitch_speed_mph", *GRF])
    session_of = meta.set_index("session_pitch")["session"].to_dict()

    groups = dict(tuple(data.groupby("session_pitch", sort=False)))
    sessions: dict = {}
    for sp in groups:
        sessions.setdefault(session_of[sp], []).append(sp)
    rows = []
    for session, sps in sessions.items():
        q = fit_offset([prepare(groups[sp]) for sp in sps])
        for sp in sps:
            row = pitch_metrics(groups[sp], q)
            row.update({"session_pitch": sp, "session": session})
            rows.append(row)
    pitch = pd.DataFrame(rows).merge(meta[["session_pitch", "session_mass_kg"]], on="session_pitch").merge(
        poi, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411:
        raise ValueError("Expected 411 pitches")
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    ath = pitch.groupby("session").mean(numeric_only=True)  # GRF means skip missing pitches
    ath.to_csv(OUT / "per_athlete.csv")

    # (1) split of the trunk-base-ahead term
    tot = ath["jfp_trunk_base_rel_com"]
    split = []
    for c in ["jfp_pelvis_rel_com", "jfp_trunk_base_rel_pelvis"]:
        r_sp, _ = partial_r(ath, c, "pitch_speed_mph")
        split.append({"part": c, "athlete_mean_j": ath[c].mean(),
                      "share_of_between_athlete_variance": np.cov(ath[c], tot)[0, 1] / tot.var(),
                      "partial_r_pitch_speed_mass": r_sp})
    split = pd.DataFrame(split)
    split.to_csv(OUT / "split_of_trunk_base_ahead.csv", index=False)

    # (2)(3) braking and GRF associations
    targets = ["jfp_trunk_base_rel_com", "jfp_pelvis_rel_com", "jfp_trunk_base_rel_pelvis", "pitch_speed_mph"]
    preds = [f"{n}_vx_{k}" for n in ["pelvis", "com", "trunk_base"] for k in ["fp", "br", "loss_fp_br"]] + GRF
    assoc = []
    for x in preds:
        row = {"predictor": x, "athlete_mean": ath[x].mean()}
        for y in targets:
            r, n = partial_r(ath, x, y)
            row[f"partial_r_{y}"] = r
            row["n_athletes"] = n
        assoc.append(row)
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(OUT / "braking_grf_associations.csv", index=False)

    pd.set_option("display.width", 250)
    print(split.round(3).to_string(index=False))
    print()
    print(assoc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
