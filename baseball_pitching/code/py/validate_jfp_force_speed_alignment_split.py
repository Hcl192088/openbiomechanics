"""Step 0: can official shoulder JFP be split into |F|, |v|, and alignment?

Window: start = fp_poi_time, end = BR_time; frames gated by (STP + JFP) > 0,
matching the positive shoulder transfer decomposition.

Frame-invariant route (no coordinate-frame guess):
    |F|        = norm(shoulder_thorax_force_xyz)            (N)
    |v|        = norm(d shoulder_jc_xyz / dt), np.gradient on the full trial (m/s)
    cos_derived = official JFP / (|F| |v|)
This identity is only physically valid if |v| matches the velocity Visual3D used;
the check is the fraction of gated frames with |cos_derived| > 1.

Independent cross-check: cos_rotated from the landmark-rotated thorax force
(mapping local x->AP, y->-ML, z->axial; see align_shoulder_power_formula.py).

Pre-set acceptance criteria (written before running):
  A. >= 99% of gated frames have |cos_derived| <= 1.05
  B. frame-level r(cos_derived, cos_rotated) >= 0.90
  C. athlete-level r(integrated rotated JFP, integrated official JFP) >= 0.95
If A fails, the |F||v|cos split is not used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("jfp_force_speed_alignment_outputs")


def thorax_rotation(prox: np.ndarray, dist: np.ndarray, ap_point: np.ndarray) -> np.ndarray:
    axial = prox - dist
    axial /= np.linalg.norm(axial, axis=1, keepdims=True)
    ap_raw = ap_point - (prox + dist) / 2
    ap = ap_raw - np.einsum("ij,ij->i", ap_raw, axial)[:, None] * axial
    ap /= np.linalg.norm(ap, axis=1, keepdims=True)
    ml = np.cross(ap, axial)
    ml /= np.linalg.norm(ml, axis=1, keepdims=True)
    base = np.stack([ml, ap, axial], axis=2)
    best_map = np.zeros((3, 3))
    best_map[1, 0] = 1
    best_map[0, 1] = -1
    best_map[2, 2] = 1
    return np.einsum("nij,jk->nik", base, best_map)


def per_pitch(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    g = g.sort_values("time")
    t_all = g["time"].to_numpy(float)
    pos = g[[f"shoulder_jc_{a}" for a in "xyz"]].to_numpy(float)
    vel_all = np.column_stack([np.gradient(pos[:, i], t_all) for i in range(3)])
    fp, br = g["fp_poi_time"].iloc[0], g["BR_time"].iloc[0]
    win = (t_all >= fp) & (t_all <= br)
    w = g[win]
    t = t_all[win]
    vel = vel_all[win]
    stp = w["shoulder_energy_transfer_stp"].to_numpy(float)
    jfp = w["shoulder_energy_transfer_jfp"].to_numpy(float)
    gate = (stp + jfp) > 0
    force_local = w[[f"shoulder_thorax_force_{a}" for a in "xyz"]].to_numpy(float)
    f_mag = np.linalg.norm(force_local, axis=1)
    v_mag = np.linalg.norm(vel, axis=1)
    denom = f_mag * v_mag
    cos_derived = np.divide(jfp, denom, out=np.full_like(jfp, np.nan), where=denom > 0)
    rot = thorax_rotation(
        w[[f"thorax_prox_{a}" for a in "xyz"]].to_numpy(float),
        w[[f"thorax_dist_{a}" for a in "xyz"]].to_numpy(float),
        w[[f"thorax_ap_{a}" for a in "xyz"]].to_numpy(float),
    )
    force_lab = np.einsum("nij,nj->ni", rot, force_local)
    jfp_rot = np.einsum("ij,ij->i", force_lab, vel)
    cos_rot = np.divide(jfp_rot, denom, out=np.full_like(jfp, np.nan), where=denom > 0)

    frames = pd.DataFrame({"gate": gate, "cos_derived": cos_derived, "cos_rotated": cos_rot,
                           "jfp": jfp, "jfp_rotated": jfp_rot})
    summary = pd.Series({
        "jfp_official_gated_j": float(np.trapezoid(np.where(gate, jfp, 0.0), t)),
        "jfp_rotated_gated_j": float(np.trapezoid(np.where(gate, jfp_rot, 0.0), t)),
    })
    return frames[frames["gate"]], summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=[
        "session_pitch", "time", "fp_poi_time", "BR_time",
        "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp"])
    forces = pd.read_csv(ROOT / "data/full_sig/forces_moments.csv", usecols=[
        "session_pitch", "time", *[f"shoulder_thorax_force_{a}" for a in "xyz"]])
    landmarks = pd.read_csv(ROOT / "data/full_sig/landmarks.csv", usecols=[
        "session_pitch", "time",
        *[f"{p}_{a}" for p in ["shoulder_jc", "thorax_prox", "thorax_dist", "thorax_ap"] for a in "xyz"]])
    data = energy.merge(forces, on=["session_pitch", "time"], validate="one_to_one").merge(
        landmarks, on=["session_pitch", "time"], validate="one_to_one")
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")

    frame_parts, rows = [], []
    for sp, g in data.groupby("session_pitch", sort=False):
        frames, summary = per_pitch(g)
        frame_parts.append(frames)
        summary["session_pitch"] = sp
        rows.append(summary)
    frames = pd.concat(frame_parts, ignore_index=True)
    pitch = pd.DataFrame(rows)
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session"])
    pitch = pitch.merge(meta, on="session_pitch", validate="one_to_one")
    athlete = pitch.groupby("session").mean(numeric_only=True)

    valid = frames["cos_derived"].notna() & frames["cos_rotated"].notna()
    cd = frames.loc[valid, "cos_derived"]
    result = {
        "gated_frames": int(len(frames)),
        "frames_with_valid_cos": int(valid.sum()),
        "A_share_abs_cos_derived_le_1_05": float((cd.abs() <= 1.05).mean()),
        "share_abs_cos_derived_le_1_00": float((cd.abs() <= 1.0).mean()),
        "cos_derived_p01": float(cd.quantile(0.01)),
        "cos_derived_median": float(cd.median()),
        "cos_derived_p99": float(cd.quantile(0.99)),
        "B_frame_r_cos_derived_vs_rotated": float(np.corrcoef(cd, frames.loc[valid, "cos_rotated"])[0, 1]),
        "frame_r_jfp_rotated_vs_official": float(np.corrcoef(frames["jfp"], frames["jfp_rotated"])[0, 1]),
        "pitch_r_integrated_jfp": float(pitch["jfp_official_gated_j"].corr(pitch["jfp_rotated_gated_j"])),
        "C_athlete_r_integrated_jfp": float(athlete["jfp_official_gated_j"].corr(athlete["jfp_rotated_gated_j"])),
        "athlete_mae_integrated_jfp_j": float((athlete["jfp_official_gated_j"] - athlete["jfp_rotated_gated_j"]).abs().mean()),
        "athlete_sd_official_jfp_j": float(athlete["jfp_official_gated_j"].std()),
    }
    result["A_pass"] = result["A_share_abs_cos_derived_le_1_05"] >= 0.99
    result["B_pass"] = result["B_frame_r_cos_derived_vs_rotated"] >= 0.90
    result["C_pass"] = result["C_athlete_r_integrated_jfp"] >= 0.95
    pd.Series(result).to_csv(OUT / "validation_summary.csv", header=["value"])
    pitch.to_csv(OUT / "per_pitch_integrated_jfp.csv", index=False)
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
