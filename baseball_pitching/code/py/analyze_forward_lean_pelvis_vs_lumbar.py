"""Is the forward lean over the hips pelvis tilt (lead hip flexion) or trunk-on-pelvis
(lumbar) flexion?

Follow-up of analyze_trunk_base_rel_pelvis_sources.py.
Window: start = fp_poi_time, end = BR_time; frames gated by (STP + JFP) > 0.

Pelvis frame: R_pel = intrinsic ZYX Cardan of (pelvis_angle_z, _y, _x), as defined in
code/v3d/CMO.v3s (PELVIS_ANGLE: RPV vs Virtual Lab, AXIS1=Z, AXIS3=X). Validated per
pitch: pelvis x-axis vs hip-to-hip landmark vector |median cos| >= 0.99. For 83 pitches
(22 sessions, all p_throws = L) the angles match only after mirroring about lab y (R -> M R M,
M = diag(1,-1,1)): left-handers' angles are mirrored to right-handed convention, landmarks are not.
Mirroring is chosen from POI p_throws; the landmark check must agree (else stop). Sign may differ
by handedness; recorded, not used).

Line r = thorax_dist - pelvis_center (hip midpoint), r_p = R_pel^T r.
    dr/dt = omega_pel x r                       carried by pelvis motion
          + R_pel d(r_p)/dt                      trunk base moving relative to pelvis
          + residual (numerical)
omega_pel (lab) from dR/dt R^T, split on pelvis axes:
    x = tilt (anterior/posterior), y = obliquity, z = pelvis rotation.
Relative motion split on pelvis axes via omega_rel = r_p x dr_p / |r_p|^2:
    about pelvis x = trunk flexion on pelvis (lumbar), about y = lateral bend,
    about z = twist of the line, plus lengthening along r_p.
Each part: JFP contribution (integral F_lab . part, gated; F_lab from the per-session
offset of analyze_jfp_force_speed_alignment.py) and mean forward (lab x) velocity.
Also: lead femur sagittal angle (lead_hip -> lead_knee_jc, lab) and lead_hip_angle_x
change FP->BR, to check whether pelvis tilt is hip flexion over a fixed femur.
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
OUT = Path(__file__).resolve().with_name("forward_lean_pelvis_vs_lumbar_outputs")
MIRROR_Y = np.diag([1.0, -1.0, 1.0])
PARTS = ["pelvis_tilt", "pelvis_obliquity", "pelvis_rotation",
         "trunk_flexion_on_pelvis", "trunk_lateral_bend_on_pelvis", "line_twist_on_pelvis",
         "line_lengthening", "residual"]


def axis_parts(omega_local: np.ndarray, axes: np.ndarray, r: np.ndarray, names: list[str]) -> dict:
    """omega_local: components on the three frame axes; axes: (n,3,3) columns = axes in lab."""
    return {name: np.cross(omega_local[:, [i]] * axes[:, :, i], r) for i, name in enumerate(names)}


def pitch_metrics(g: pd.DataFrame, q: np.ndarray, throws: str) -> dict:
    g = g.sort_values("time")
    t = g["time"].to_numpy(float)
    pt = lambda name: g[[f"{name}_{a}" for a in "xyz"]].to_numpy(float)
    R_raw = Rotation.from_euler("ZYX", g[["pelvis_angle_z", "pelvis_angle_y", "pelvis_angle_x"]].to_numpy(float),
                                degrees=True).as_matrix()
    R_mirror = np.einsum("ij,njk,kl->nil", MIRROR_Y, R_raw, MIRROR_Y)
    hip_axis = pt("lead_hip") - pt("rear_hip")
    hip_axis /= np.linalg.norm(hip_axis, axis=1, keepdims=True)
    fp, br = g["fp_poi_time"].iloc[0], g["BR_time"].iloc[0]
    win = (t >= fp) & (t <= br)
    cos_raw = float(np.median(np.einsum("ij,ij->i", R_raw[win, :, 0], hip_axis[win])))
    cos_mir = float(np.median(np.einsum("ij,ij->i", R_mirror[win, :, 0], hip_axis[win])))
    ok_raw, ok_mir = abs(cos_raw) >= 0.99, abs(cos_mir) >= 0.99
    if ok_raw == ok_mir:
        raise ValueError(f"{g['session_pitch'].iloc[0]}: pelvis frame ambiguous or unmatched "
                         f"(raw cos {cos_raw:.3f}, mirrored cos {cos_mir:.3f})")
    if throws not in ("L", "R"):
        raise ValueError(f"{g['session_pitch'].iloc[0]}: unexpected p_throws {throws!r}")
    if (throws == "L") != ok_mir:
        raise ValueError(f"{g['session_pitch'].iloc[0]}: p_throws={throws} disagrees with landmark check "
                         f"(raw cos {cos_raw:.3f}, mirrored cos {cos_mir:.3f})")
    R, hip_cos, mirrored = (R_mirror, cos_mir, True) if throws == "L" else (R_raw, cos_raw, False)
    if np.interp(fp, t, grad(pt("centerofmass"), t)[:, 0]) <= 0:
        raise ValueError(f"{g['session_pitch'].iloc[0]}: COM x-velocity at FP <= 0; +x is not forward")

    pelvis = (pt("rear_hip") + pt("lead_hip")) / 2
    P = pt("thorax_dist")
    r = P - pelvis
    dr = grad(P, t) - grad(pelvis, t)
    W = np.einsum("nij,nkj->nik", grad(R, t), R)
    W = 0.5 * (W - np.transpose(W, (0, 2, 1)))
    omega = np.column_stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]])
    omega_p = np.einsum("nji,nj->ni", R, omega)
    parts = axis_parts(omega_p, R, r, ["pelvis_tilt", "pelvis_obliquity", "pelvis_rotation"])

    r_p = np.einsum("nji,nj->ni", R, r)
    dr_p = grad(r_p, t)
    rel_lab = np.einsum("nij,nj->ni", R, dr_p)
    rp2 = np.einsum("ij,ij->i", r_p, r_p)
    omega_rel_p = np.cross(r_p, dr_p) / rp2[:, None]          # in pelvis coordinates
    for i, name in enumerate(["trunk_flexion_on_pelvis", "trunk_lateral_bend_on_pelvis", "line_twist_on_pelvis"]):
        e = np.zeros(3)
        e[i] = 1.0
        part_p = np.cross(omega_rel_p[:, [i]] * e, r_p)
        parts[name] = np.einsum("nij,nj->ni", R, part_p)
    rhat_p = r_p / np.sqrt(rp2)[:, None]
    parts["line_lengthening"] = np.einsum("nij,nj->ni", R, np.einsum("ij,ij->i", rhat_p, dr_p)[:, None] * rhat_p)
    rel_sum = sum(parts[k] for k in ["trunk_flexion_on_pelvis", "trunk_lateral_bend_on_pelvis",
                                     "line_twist_on_pelvis", "line_lengthening"])
    rel_split_err = float(np.abs(rel_sum - rel_lab).max())
    parts["residual"] = dr - sum(parts.values())

    jfp = g["shoulder_energy_transfer_jfp"].to_numpy(float)[win]
    stp = g["shoulder_energy_transfer_stp"].to_numpy(float)[win]
    gate = (jfp + stp) > 0
    tw = t[win]
    B = basis(pt("thorax_prox")[win], P[win], pt("thorax_ap")[win])
    Q = Rotation.from_rotvec(q).as_matrix() @ M0
    F_lab = np.einsum("nij,jk,nk->ni", B, Q, pt("shoulder_thorax_force")[win])

    femur = pt("lead_knee_jc") - pt("lead_hip")
    femur_sag = np.degrees(np.arctan2(femur[:, 0], -femur[:, 2]))
    out = {"hip_axis_cos": hip_cos, "pelvis_angles_mirrored": mirrored, "rel_split_error_m_s": rel_split_err,
           "lead_femur_sag_change_fp_br": float(np.interp(br, t, femur_sag) - np.interp(fp, t, femur_sag)),
           "lead_hip_angle_x_change_fp_br": float(np.interp(br, t, g["lead_hip_angle_x"]) - np.interp(fp, t, g["lead_hip_angle_x"])),
           "pelvis_angle_x_change_fp_br": float(np.interp(br, t, g["pelvis_angle_x"]) - np.interp(fp, t, g["pelvis_angle_x"]))}
    for k, v in {**parts, "total": dr}.items():
        vw = v[win]
        out[f"jfp_{k}"] = float(np.trapezoid(np.where(gate, np.einsum("ij,ij->i", F_lab, vw), 0.0), tw))
        out[f"vx_{k}"] = float(vw[gate, 0].mean())
    return out


def residualize(df: pd.DataFrame, col: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df["session_mass_kg"].to_numpy(float)])
    y = df[col].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def pr(df: pd.DataFrame, x: str, y: str) -> float:
    a, b = residualize(df, x), residualize(df, y)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fs = ROOT / "data" / "full_sig"
    energy = pd.read_csv(fs / "energy_flow.csv", usecols=[
        "session_pitch", "time", "fp_poi_time", "BR_time",
        "shoulder_energy_transfer_jfp", "shoulder_energy_transfer_stp"])
    forces = pd.read_csv(fs / "forces_moments.csv", usecols=[
        "session_pitch", "time", *[f"shoulder_thorax_force_{a}" for a in "xyz"]])
    pts = ["shoulder_jc", "thorax_prox", "thorax_dist", "thorax_ap", "centerofmass", "rear_hip", "lead_hip", "lead_knee_jc"]
    landmarks = pd.read_csv(fs / "landmarks.csv", usecols=[
        "session_pitch", "time", *[f"{p}_{a}" for p in pts for a in "xyz"]])
    angles = pd.read_csv(fs / "joint_angles.csv", usecols=[
        "session_pitch", "time", "pelvis_angle_x", "pelvis_angle_y", "pelvis_angle_z", "lead_hip_angle_x"])
    data = (energy.merge(forces, on=["session_pitch", "time"], validate="one_to_one")
            .merge(landmarks, on=["session_pitch", "time"], validate="one_to_one")
            .merge(angles, on=["session_pitch", "time"], validate="one_to_one"))
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "pitch_speed_mph", "p_throws"])
    throws_of = poi.set_index("session_pitch")["p_throws"].to_dict()
    poi = poi.drop(columns="p_throws")
    session_of = meta.set_index("session_pitch")["session"].to_dict()

    groups = dict(tuple(data.groupby("session_pitch", sort=False)))
    sessions: dict = {}
    for sp in groups:
        sessions.setdefault(session_of[sp], []).append(sp)
    rows = []
    for session, sps in sessions.items():
        q = fit_offset([prepare(groups[sp]) for sp in sps])
        for sp in sps:
            row = pitch_metrics(groups[sp], q, throws_of[sp])
            row.update({"session_pitch": sp, "session": session})
            rows.append(row)
    pitch = pd.DataFrame(rows).merge(meta[["session_pitch", "session_mass_kg"]], on="session_pitch").merge(
        poi, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411:
        raise ValueError("Expected 411 pitches")
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    ath = pitch.groupby("session").mean(numeric_only=True)
    ath.to_csv(OUT / "per_athlete.csv")

    tot, tot_res = ath["jfp_total"], residualize(ath, "jfp_total")
    table = []
    for k in PARTS + ["total"]:
        c = f"jfp_{k}"
        table.append({
            "part": k, "jfp_mean_j": ath[c].mean(),
            "share_of_between_athlete_variance": np.cov(ath[c], tot)[0, 1] / tot.var(),
            "share_mass_adjusted": np.cov(residualize(ath, c), tot_res)[0, 1] / tot_res.var(ddof=1),
            "partial_r_jfp_part_pitch_speed": pr(ath, c, "pitch_speed_mph"),
            "forward_velocity_mean_m_s": ath[f"vx_{k}"].mean(),
            "partial_r_forward_velocity_pitch_speed": pr(ath, f"vx_{k}", "pitch_speed_mph"),
        })
    table = pd.DataFrame(table)
    table.to_csv(OUT / "lean_sources.csv", index=False)

    checks = pd.Series({
        "pitches_pelvis_angles_mirrored": int(pitch["pelvis_angles_mirrored"].sum()),
        "pitches_hip_axis_cos_positive": int((pitch["hip_axis_cos"] > 0).sum()),
        "pitches_hip_axis_cos_negative": int((pitch["hip_axis_cos"] < 0).sum()),
        "min_abs_hip_axis_cos": float(pitch["hip_axis_cos"].abs().min()),
        "max_rel_split_error_m_s": float(pitch["rel_split_error_m_s"].max()),
        "athlete_mean_lead_femur_sag_change_deg": float(ath["lead_femur_sag_change_fp_br"].mean()),
        "athlete_mean_lead_hip_angle_x_change_deg": float(ath["lead_hip_angle_x_change_fp_br"].mean()),
        "athlete_mean_pelvis_angle_x_change_deg": float(ath["pelvis_angle_x_change_fp_br"].mean()),
        "athlete_r_pelvis_tilt_change_vs_lead_hip_x_change": float(ath["pelvis_angle_x_change_fp_br"].corr(ath["lead_hip_angle_x_change_fp_br"])),
        "athlete_r_jfp_pelvis_tilt_vs_lead_hip_x_change": float(ath["jfp_pelvis_tilt"].corr(ath["lead_hip_angle_x_change_fp_br"])),
    })
    checks.to_csv(OUT / "checks.csv", header=["value"])

    pd.set_option("display.width", 250)
    print(checks.round(4).to_string())
    print()
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
