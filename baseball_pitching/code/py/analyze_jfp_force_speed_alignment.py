"""Split shoulder JFP (FP->BR positive-transfer frames) into force, shoulder speed,
alignment and duration, after validating the force direction.

Window: start = fp_poi_time, end = BR_time. Gate: (STP + JFP) > 0, matching the
positive shoulder transfer decomposition (analyze_shoulder_transfer_component_variance.py).

Direction reconstruction
------------------------
`shoulder_thorax_force_xyz` is resolved in the Visual3D thorax (RTA) frame.
A landmark thorax basis B (thorax_prox / thorax_dist / thorax_ap) with the
structural mapping M0 (local x->AP, y->-ML, z->axial; align_shoulder_power_formula.py)
gives r ~ 0.93 frame-wise. Remaining error is modelled as a constant rotation
offset Q per session (static-calibration offset between the landmark basis and
RTA):  F_lab = B @ Q @ M0 @ F_local,  JFP_hat = dot(F_lab, v_shoulder_jc).
Q (3 params) is fit by least squares over all FP->BR frames.
Validation: leave-one-pitch-out within session (Q fit on the other pitches).

Decomposition (per pitch, gated frames; official JFP used for energy)
---------------------------------------------------------------------
    |F| = norm(F_local), |v| = norm(d shoulder_jc / dt), cos = JFP / (|F||v|)
    E = integral JFP = T * mean|F| * mean|v| * mean(cos) * k
    log E = log T + log mF + log mV + log mC + log k     (exact identity)
k captures within-pitch co-variation (e.g. force and speed peaking together).
Between-athlete variance share of term X = cov(X, log E) / var(log E).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("jfp_force_speed_alignment_outputs")
M0 = np.zeros((3, 3))
M0[1, 0] = 1
M0[0, 1] = -1
M0[2, 2] = 1


def prepare(g: pd.DataFrame) -> dict:
    g = g.sort_values("time")
    t_all = g["time"].to_numpy(float)
    pos = g[[f"shoulder_jc_{a}" for a in "xyz"]].to_numpy(float)
    vel = np.column_stack([np.gradient(pos[:, i], t_all) for i in range(3)])
    win = (t_all >= g["fp_poi_time"].iloc[0]) & (t_all <= g["BR_time"].iloc[0])
    w = g[win]
    prox = w[[f"thorax_prox_{a}" for a in "xyz"]].to_numpy(float)
    dist = w[[f"thorax_dist_{a}" for a in "xyz"]].to_numpy(float)
    ap_point = w[[f"thorax_ap_{a}" for a in "xyz"]].to_numpy(float)
    axial = prox - dist
    axial /= np.linalg.norm(axial, axis=1, keepdims=True)
    ap = ap_point - (prox + dist) / 2
    ap -= np.einsum("ij,ij->i", ap, axial)[:, None] * axial
    ap /= np.linalg.norm(ap, axis=1, keepdims=True)
    ml = np.cross(ap, axial)
    jfp = w["shoulder_energy_transfer_jfp"].to_numpy(float)
    stp = w["shoulder_energy_transfer_stp"].to_numpy(float)
    return {
        "B": np.stack([ml, ap, axial], axis=2),
        "F": w[[f"shoulder_thorax_force_{a}" for a in "xyz"]].to_numpy(float),
        "V": vel[win],
        "J": jfp,
        "gate": (jfp + stp) > 0,
        "t": t_all[win],
    }


def predict(q: np.ndarray, p: dict) -> np.ndarray:
    Q = Rotation.from_rotvec(q).as_matrix() @ M0
    f_lab = np.einsum("nij,jk,nk->ni", p["B"], Q, p["F"])
    return np.einsum("ij,ij->i", f_lab, p["V"])


def fit_offset(pitches: list[dict]) -> np.ndarray:
    def resid(q: np.ndarray) -> np.ndarray:
        return np.concatenate([predict(q, p) - p["J"] for p in pitches])
    return least_squares(resid, np.zeros(3)).x


def integ(values: np.ndarray, p: dict) -> float:
    return float(np.trapezoid(np.where(p["gate"], values, 0.0), p["t"]))


def residualize(df: pd.DataFrame, col: str, controls: list[str]) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df[controls].to_numpy(float)])
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
    landmarks = pd.read_csv(fs / "landmarks.csv", usecols=[
        "session_pitch", "time",
        *[f"{p}_{a}" for p in ["shoulder_jc", "thorax_prox", "thorax_dist", "thorax_ap"] for a in "xyz"]])
    data = energy.merge(forces, on=["session_pitch", "time"], validate="one_to_one").merge(
        landmarks, on=["session_pitch", "time"], validate="one_to_one")
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "pitch_speed_mph"])

    prepared = {sp: prepare(g) for sp, g in data.groupby("session_pitch", sort=False)}
    session_of = meta.set_index("session_pitch")["session"].to_dict()
    sessions: dict = {}
    for sp in prepared:
        sessions.setdefault(session_of[sp], []).append(sp)
    if len(prepared) != 411 or len(sessions) != 100 or min(len(v) for v in sessions.values()) < 2:
        raise ValueError("Expected 411 pitches, 100 sessions, >= 2 pitches per session")

    # ---- validation: leave-one-pitch-out offset within session
    val_rows, frame_parts, pitch_rows = [], [], []
    for session, sps in sessions.items():
        q_session = fit_offset([prepared[s] for s in sps])
        for sp in sps:
            p = prepared[sp]
            q_lopo = fit_offset([prepared[o] for o in sps if o != sp])
            j_lopo = predict(q_lopo, p)
            j_base = predict(np.zeros(3), p)
            val_rows.append({
                "session_pitch": sp, "session": session,
                "frame_r_landmark_only": np.corrcoef(p["J"], j_base)[0, 1],
                "frame_r_lopo": np.corrcoef(p["J"], j_lopo)[0, 1],
                "frame_mae_lopo_w": np.mean(np.abs(p["J"] - j_lopo)),
                "jfp_official_j": integ(p["J"], p),
                "jfp_lopo_j": integ(j_lopo, p),
                "session_offset_deg": np.degrees(np.linalg.norm(q_session)),
            })
            # ---- decomposition inputs (direction from session offset)
            g = p["gate"]
            f_mag = np.linalg.norm(p["F"], axis=1)
            v_mag = np.linalg.norm(p["V"], axis=1)
            cos_derived = p["J"] / (f_mag * v_mag)
            cos_rot = predict(q_session, p) / (f_mag * v_mag)
            frame_parts.append(pd.DataFrame({"cos_derived": cos_derived[g], "cos_rotated": cos_rot[g]}))
            dt = float(np.median(np.diff(p["t"])))
            E = integ(p["J"], p)
            T = g.sum() * dt
            mF, mV, mC = f_mag[g].mean(), v_mag[g].mean(), cos_derived[g].mean()
            pitch_rows.append({
                "session_pitch": sp, "session": session, "jfp_j": E, "gated_duration_s": T,
                "mean_force_n": mF, "mean_shoulder_speed_m_s": mV, "mean_alignment_cos": mC,
                "covariation_k": E / (T * mF * mV * mC),
            })

    val = pd.DataFrame(val_rows)
    frames = pd.concat(frame_parts, ignore_index=True)
    ath_val = val.groupby("session")[["jfp_official_j", "jfp_lopo_j"]].mean()
    validation = pd.Series({
        "frame_r_landmark_only_median": val["frame_r_landmark_only"].median(),
        "frame_r_lopo_median": val["frame_r_lopo"].median(),
        "frame_r_lopo_p05": val["frame_r_lopo"].quantile(0.05),
        "pitch_r_integrated_lopo": val["jfp_official_j"].corr(val["jfp_lopo_j"]),
        "athlete_r_integrated_lopo": ath_val["jfp_official_j"].corr(ath_val["jfp_lopo_j"]),
        "athlete_mae_integrated_lopo_j": (ath_val["jfp_official_j"] - ath_val["jfp_lopo_j"]).abs().mean(),
        "athlete_sd_official_jfp_j": ath_val["jfp_official_j"].std(),
        "frame_r_cos_derived_vs_rotated": np.corrcoef(frames["cos_derived"], frames["cos_rotated"])[0, 1],
        "share_abs_cos_derived_le_1_05": (frames["cos_derived"].abs() <= 1.05).mean(),
        "session_offset_deg_median": val.groupby("session")["session_offset_deg"].first().median(),
    })
    validation.to_csv(OUT / "offset_validation_summary.csv", header=["value"])
    val.to_csv(OUT / "offset_validation_per_pitch.csv", index=False)

    # ---- decomposition
    pitch = pd.DataFrame(pitch_rows).merge(meta[["session_pitch", "session_mass_kg"]], on="session_pitch").merge(
        poi, on="session_pitch", validate="one_to_one")
    if (pitch[["jfp_j", "mean_alignment_cos", "covariation_k"]] <= 0).any().any():
        raise ValueError("Non-positive term; log decomposition invalid")
    terms = {"log_duration": "gated_duration_s", "log_force": "mean_force_n",
             "log_speed": "mean_shoulder_speed_m_s", "log_alignment": "mean_alignment_cos",
             "log_covariation": "covariation_k"}
    pitch["log_jfp"] = np.log(pitch["jfp_j"])
    for k, c in terms.items():
        pitch[k] = np.log(pitch[c])
    pitch.to_csv(OUT / "decomposition_per_pitch.csv", index=False)
    ath = pitch.groupby("session").mean(numeric_only=True)
    ath.to_csv(OUT / "decomposition_per_athlete.csv")

    logE = ath["log_jfp"]
    logE_res = residualize(ath, "log_jfp", ["session_mass_kg"])
    speed_res = residualize(ath, "pitch_speed_mph", ["session_mass_kg"])
    rows = []
    for k, c in terms.items():
        x_res = residualize(ath, k, ["session_mass_kg"])
        raw_res = residualize(ath, c, ["session_mass_kg"])
        rows.append({
            "term": k, "athlete_mean_raw": ath[c].mean(), "athlete_cv_raw": ath[c].std() / ath[c].mean(),
            "share_of_log_jfp_variance": np.cov(ath[k], logE)[0, 1] / logE.var(),
            "share_of_log_jfp_variance_mass_adjusted": np.cov(x_res, logE_res)[0, 1] / logE_res.var(ddof=1),
            "r_mass": ath[c].corr(ath["session_mass_kg"]),
            "r_pitch_speed": ath[c].corr(ath["pitch_speed_mph"]),
            "partial_r_pitch_speed_mass": float(np.corrcoef(raw_res, speed_res)[0, 1]),
        })
    decomp = pd.DataFrame(rows)
    decomp.to_csv(OUT / "decomposition_variance_shares.csv", index=False)

    pd.set_option("display.width", 250)
    print(validation.round(4).to_string())
    print()
    print(decomp.round(3).to_string(index=False))
    print(f"\nshare sum (raw) = {decomp['share_of_log_jfp_variance'].sum():.4f}")


if __name__ == "__main__":
    main()
