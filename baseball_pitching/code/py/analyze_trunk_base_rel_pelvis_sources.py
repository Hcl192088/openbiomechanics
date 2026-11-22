"""Split 'trunk base moving ahead of the pelvis' (FP->BR) into lean, side-bend,
swing about vertical, and lengthening of the pelvis->trunk-base line.

Follow-up of analyze_trunk_base_ahead_of_com.py, where
(v_thorax_dist - v_pelvis) . F_shoulder carried the pitch-speed link.

Window: start = fp_poi_time, end = BR_time; frames gated by (STP + JFP) > 0.

Landmark-only decomposition (no Visual3D frame needed):
    r = thorax_dist - pelvis,  pelvis = midpoint(rear_hip, lead_hip)
    dr/dt = (r_hat . dr/dt) r_hat                      lengthening
          + (omega . e_y)(e_y x r)                     sagittal lean (lab y = medio-lateral)
          + (omega . e_x)(e_x x r)                     side bend   (lab x = toward plate)
          + (omega . e_z)(e_z x r)                     swing about vertical
    omega = r x dr/dt / |r|^2   (angular velocity of the line, perpendicular to r)
The four parts sum exactly to dr/dt. Lab axes checked: +x forward (COM x-velocity at
FP > 0, enforced in pitch loop), +z up (pelvis z > 0).
Each part: JFP contribution = integral F_lab . part (gated), F_lab from the
per-session offset (analyze_jfp_force_speed_alignment.py); and mean forward (x)
velocity on gated frames (m/s).
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
OUT = Path(__file__).resolve().with_name("trunk_base_rel_pelvis_sources_outputs")
PARTS = ["lengthening", "sagittal_lean", "side_bend", "swing_vertical"]
E = {"sagittal_lean": np.array([0.0, 1.0, 0.0]), "side_bend": np.array([1.0, 0.0, 0.0]),
     "swing_vertical": np.array([0.0, 0.0, 1.0])}


def pitch_metrics(g: pd.DataFrame, q: np.ndarray) -> dict:
    g = g.sort_values("time")
    t = g["time"].to_numpy(float)
    pt = lambda name: g[[f"{name}_{a}" for a in "xyz"]].to_numpy(float)
    P = pt("thorax_dist")
    pelvis = (pt("rear_hip") + pt("lead_hip")) / 2
    if (pelvis[:, 2] <= 0).any():
        raise ValueError(f"{g.name}: pelvis z <= 0; +z is not up")
    fp, br = g["fp_poi_time"].iloc[0], g["BR_time"].iloc[0]
    if np.interp(fp, t, grad(pt("centerofmass"), t)[:, 0]) <= 0:
        raise ValueError(f"{g.name}: COM x-velocity at FP <= 0; +x is not forward")

    r = P - pelvis
    dr = grad(P, t) - grad(pelvis, t)
    r2 = np.einsum("ij,ij->i", r, r)
    r_hat = r / np.sqrt(r2)[:, None]
    omega = np.cross(r, dr) / r2[:, None]
    parts = {"lengthening": np.einsum("ij,ij->i", r_hat, dr)[:, None] * r_hat}
    for k, e in E.items():
        parts[k] = (omega @ e)[:, None] * np.cross(np.broadcast_to(e, r.shape), r)
    recon_err = float(np.abs(sum(parts.values()) - dr).max())

    win = (t >= fp) & (t <= br)
    jfp = g["shoulder_energy_transfer_jfp"].to_numpy(float)[win]
    stp = g["shoulder_energy_transfer_stp"].to_numpy(float)[win]
    gate = (jfp + stp) > 0
    tw = t[win]
    B = basis(pt("thorax_prox")[win], P[win], pt("thorax_ap")[win])
    Q = Rotation.from_rotvec(q).as_matrix() @ M0
    F_lab = np.einsum("nij,jk,nk->ni", B, Q, pt("shoulder_thorax_force")[win])

    out = {"recon_error_m_s": recon_err,
           "line_length_m": float(np.sqrt(r2[win][gate]).mean()),
           "line_forward_lean_deg_fp": float(np.degrees(np.arctan2(
               np.interp(fp, t, r[:, 0]), np.interp(fp, t, r[:, 2])))),
           "line_forward_lean_deg_br": float(np.degrees(np.arctan2(
               np.interp(br, t, r[:, 0]), np.interp(br, t, r[:, 2]))))}
    for k, v in {**parts, "total": dr}.items():
        vw = v[win]
        out[f"jfp_{k}"] = float(np.trapezoid(np.where(gate, np.einsum("ij,ij->i", F_lab, vw), 0.0), tw))
        out[f"vx_{k}"] = float(vw[gate, 0].mean())
    return out


def residualize(df: pd.DataFrame, col: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df["session_mass_kg"].to_numpy(float)])
    y = df[col].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


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
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "pitch_speed_mph"])
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
    ath = pitch.groupby("session").mean(numeric_only=True)
    ath.to_csv(OUT / "per_athlete.csv")

    tot, tot_res, spd_res = ath["jfp_total"], residualize(ath, "jfp_total"), residualize(ath, "pitch_speed_mph")
    rows = []
    for k in PARTS + ["total"]:
        c, c_res = f"jfp_{k}", residualize(ath, f"jfp_{k}")
        v_res = residualize(ath, f"vx_{k}")
        rows.append({
            "part": k, "jfp_mean_j": ath[c].mean(),
            "share_of_between_athlete_variance": np.cov(ath[c], tot)[0, 1] / tot.var(),
            "share_mass_adjusted": np.cov(c_res, tot_res)[0, 1] / tot_res.var(ddof=1),
            "partial_r_jfp_part_pitch_speed": float(np.corrcoef(c_res, spd_res)[0, 1]),
            "forward_velocity_mean_m_s": ath[f"vx_{k}"].mean(),
            "partial_r_forward_velocity_pitch_speed": float(np.corrcoef(v_res, spd_res)[0, 1]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "trunk_base_rel_pelvis_sources.csv", index=False)
    extra = {}
    for c in ["line_forward_lean_deg_fp", "line_forward_lean_deg_br", "line_length_m"]:
        extra[c] = {"athlete_mean": ath[c].mean(),
                    "partial_r_pitch_speed": float(np.corrcoef(residualize(ath, c), spd_res)[0, 1])}
    extra = pd.DataFrame(extra).T
    extra.to_csv(OUT / "line_posture.csv")

    pd.set_option("display.width", 250)
    print(f"max reconstruction error (m/s): {pitch['recon_error_m_s'].max():.2e}")
    print(table.round(3).to_string(index=False))
    print()
    print(extra.round(3).to_string())


if __name__ == "__main__":
    main()
