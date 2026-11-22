"""Where does throwing-shoulder joint center speed (and JFP) come from?

Window: start = fp_poi_time, end = BR_time; frames gated by (STP + JFP) > 0,
same gate as analyze_jfp_force_speed_alignment.py.

Kinematic identity (landmark thorax basis B = [ml, ap, axial], P = thorax_dist,
shoulder position s = P + B @ l, l = shoulder position in thorax frame):

    v_s = v_com                      (1) whole-body centre of mass
        + (v_P - v_com)              (2) trunk base relative to COM
        + omega x (s - P)            (3) trunk angular motion, split by thorax axis:
                                         axial (rotation), ml (flexion/extension),
                                         ap (lateral tilt)
        + B @ dl/dt                  (4) shoulder JC moving within the thorax frame
        + residual                   (5) numerical differentiation mismatch

omega from the landmark basis: [omega]x = dB/dt @ B^T (skew part).
Each term is projected (a) onto the unit direction of v_s -> contributions sum to
|v_s|; (b) onto the lab shoulder force F_lab -> contributions sum to the
reconstructed JFP. F_lab uses the per-session offset validated in
analyze_jfp_force_speed_alignment.py (leave-one-pitch-out athlete r = 0.993).

Axis labels use the landmark thorax basis (axial = thorax_prox - thorax_dist),
not the Visual3D RTA frame; the axial axis is anatomically defined either way.
Athlete level = pitches averaged within `session` (100 athletes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_jfp_force_speed_alignment import M0, fit_offset, prepare
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("shoulder_speed_sources_outputs")
TERMS = ["com", "trunk_base_rel_com", "trunk_rotation_axial", "trunk_flexion_ml",
         "trunk_lateral_tilt_ap", "shoulder_rel_thorax", "residual"]


def basis(prox: np.ndarray, dist: np.ndarray, ap_point: np.ndarray) -> np.ndarray:
    axial = prox - dist
    axial /= np.linalg.norm(axial, axis=1, keepdims=True)
    ap = ap_point - (prox + dist) / 2
    ap -= np.einsum("ij,ij->i", ap, axial)[:, None] * axial
    ap /= np.linalg.norm(ap, axis=1, keepdims=True)
    ml = np.cross(ap, axial)
    return np.stack([ml, ap, axial], axis=2)


def grad(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(x, t, axis=0)


def pitch_terms(g: pd.DataFrame, q: np.ndarray) -> tuple[dict, dict]:
    g = g.sort_values("time")
    t = g["time"].to_numpy(float)
    pt = lambda name: g[[f"{name}_{a}" for a in "xyz"]].to_numpy(float)
    s, P, com = pt("shoulder_jc"), pt("thorax_dist"), pt("centerofmass")
    B = basis(pt("thorax_prox"), P, pt("thorax_ap"))
    l = np.einsum("nji,nj->ni", B, s - P)          # B^T (s - P)
    dB = grad(B, t)
    W = np.einsum("nij,nkj->nik", dB, B)           # dB/dt @ B^T
    W = 0.5 * (W - np.transpose(W, (0, 2, 1)))
    omega = np.column_stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]])
    r = s - P
    v_s, v_P, v_com = grad(s, t), grad(P, t), grad(com, t)
    omega_local = np.einsum("nji,nj->ni", B, omega)  # components on ml, ap, axial
    rot_parts = {
        "trunk_flexion_ml": np.cross(omega_local[:, [0]] * B[:, :, 0], r),
        "trunk_lateral_tilt_ap": np.cross(omega_local[:, [1]] * B[:, :, 1], r),
        "trunk_rotation_axial": np.cross(omega_local[:, [2]] * B[:, :, 2], r),
    }
    comps = {"com": v_com, "trunk_base_rel_com": v_P - v_com, **rot_parts,
             "shoulder_rel_thorax": np.einsum("nij,nj->ni", B, grad(l, t))}
    comps["residual"] = v_s - sum(comps.values())

    win = (t >= g["fp_poi_time"].iloc[0]) & (t <= g["BR_time"].iloc[0])
    jfp = g["shoulder_energy_transfer_jfp"].to_numpy(float)[win]
    stp = g["shoulder_energy_transfer_stp"].to_numpy(float)[win]
    gate = (jfp + stp) > 0
    tw = t[win]
    Q = Rotation.from_rotvec(q).as_matrix() @ M0
    F_lab = np.einsum("nij,jk,nk->ni", B[win], Q, pt("shoulder_thorax_force")[win])
    vs = v_s[win]
    speed = np.linalg.norm(vs, axis=1)
    unit = vs / speed[:, None]

    speed_part, jfp_part = {}, {}
    for k, c in comps.items():
        cw = c[win]
        speed_part[k] = float(np.einsum("ij,ij->i", cw, unit)[gate].mean())
        jfp_part[k] = float(np.trapezoid(np.where(gate, np.einsum("ij,ij->i", F_lab, cw), 0.0), tw))
    speed_part["total_speed"] = float(speed[gate].mean())
    jfp_part["total_jfp_reconstructed"] = float(np.trapezoid(np.where(gate, np.einsum("ij,ij->i", F_lab, vs), 0.0), tw))
    jfp_part["jfp_official"] = float(np.trapezoid(np.where(gate, jfp, 0.0), tw))
    return speed_part, jfp_part


def residualize(df: pd.DataFrame, col: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df["session_mass_kg"].to_numpy(float)])
    y = df[col].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def share_table(ath: pd.DataFrame, prefix: str, total: str) -> pd.DataFrame:
    tot, tot_res = ath[total], residualize(ath, total)
    spd_res = residualize(ath, "pitch_speed_mph")
    rows = []
    for k in TERMS:
        c = f"{prefix}{k}"
        c_res = residualize(ath, c)
        rows.append({
            "term": k, "athlete_mean": ath[c].mean(), "share_of_mean": ath[c].mean() / tot.mean(),
            "share_of_between_athlete_variance": np.cov(ath[c], tot)[0, 1] / tot.var(),
            "share_mass_adjusted": np.cov(c_res, tot_res)[0, 1] / tot_res.var(ddof=1),
            "r_pitch_speed": ath[c].corr(ath["pitch_speed_mph"]),
            "partial_r_pitch_speed_mass": float(np.corrcoef(c_res, spd_res)[0, 1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fs = ROOT / "data" / "full_sig"
    energy = pd.read_csv(fs / "energy_flow.csv", usecols=[
        "session_pitch", "time", "fp_poi_time", "BR_time",
        "shoulder_energy_transfer_jfp", "shoulder_energy_transfer_stp"])
    forces = pd.read_csv(fs / "forces_moments.csv", usecols=[
        "session_pitch", "time", *[f"shoulder_thorax_force_{a}" for a in "xyz"]])
    landmarks = pd.read_csv(fs / "landmarks.csv", usecols=[
        "session_pitch", "time",
        *[f"{p}_{a}" for p in ["shoulder_jc", "thorax_prox", "thorax_dist", "thorax_ap", "centerofmass"] for a in "xyz"]])
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
    if len(groups) != 411 or len(sessions) != 100:
        raise ValueError("Expected 411 pitches and 100 sessions")

    rows = []
    for session, sps in sessions.items():
        q = fit_offset([prepare(groups[sp]) for sp in sps])
        for sp in sps:
            speed_part, jfp_part = pitch_terms(groups[sp], q)
            row = {"session_pitch": sp, "session": session}
            row.update({f"speed_{k}": v for k, v in speed_part.items()})
            row.update({f"jfp_{k}": v for k, v in jfp_part.items()})
            rows.append(row)
    pitch = pd.DataFrame(rows).merge(meta[["session_pitch", "session_mass_kg"]], on="session_pitch").merge(
        poi, on="session_pitch", validate="one_to_one")
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    ath = pitch.groupby("session").mean(numeric_only=True)
    ath.to_csv(OUT / "per_athlete.csv")

    checks = pd.Series({
        "max_abs_speed_sum_error": float((pitch[[f"speed_{k}" for k in TERMS]].sum(axis=1) - pitch["speed_total_speed"]).abs().max()),
        "pitch_r_jfp_reconstructed_vs_official": float(pitch["jfp_total_jfp_reconstructed"].corr(pitch["jfp_jfp_official"])),
        "athlete_r_jfp_reconstructed_vs_official": float(ath["jfp_total_jfp_reconstructed"].corr(ath["jfp_jfp_official"])),
        "athlete_mean_residual_speed_m_s": float(ath["speed_residual"].mean()),
    })
    checks.to_csv(OUT / "checks.csv", header=["value"])
    speed_tab = share_table(ath, "speed_", "speed_total_speed")
    jfp_tab = share_table(ath, "jfp_", "jfp_total_jfp_reconstructed")
    speed_tab.to_csv(OUT / "speed_sources.csv", index=False)
    jfp_tab.to_csv(OUT / "jfp_sources.csv", index=False)

    pd.set_option("display.width", 250)
    print(checks.round(4).to_string())
    print("\nSHOULDER SPEED SOURCES (m/s)")
    print(speed_tab.round(3).to_string(index=False))
    print("\nJFP SOURCES (J)")
    print(jfp_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
