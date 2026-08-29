"""Validate exported vectors against Visual3D shoulder JFP/STP definitions."""

from pathlib import Path
from itertools import permutations, product

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("shoulder_power_formula_alignment_outputs")


def metrics(name: str, official: np.ndarray, candidate: np.ndarray) -> dict:
    valid = np.isfinite(official) & np.isfinite(candidate)
    x, y = official[valid], candidate[valid]
    return {
        "candidate": name,
        "n": len(x),
        "r": np.corrcoef(x, y)[0, 1],
        "mae_w": np.mean(np.abs(x - y)),
        "mean_error_w": np.mean(x - y),
        "slope_candidate_on_official": np.polyfit(x, y, 1)[0],
    }


def main() -> None:
    energy = pd.read_csv(
        ROOT / "data/full_sig/energy_flow.csv",
        usecols=[
            "session_pitch", "time", "fp_poi_time", "MER_time",
            "shoulder_energy_transfer_jfp", "shoulder_energy_transfer_stp",
            "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr",
        ],
    )
    force_moment_cols = [
        f"shoulder_{side}_{kind}_{axis}"
        for side in ["thorax", "upper_arm"]
        for kind in ["force", "moment"]
        for axis in "xyz"
    ]
    kinetics = pd.read_csv(
        ROOT / "data/full_sig/forces_moments.csv",
        usecols=["session_pitch", "time", *force_moment_cols],
    )
    velocities = pd.read_csv(
        ROOT / "data/full_sig/joint_velos.csv",
        usecols=[
            "session_pitch", "time",
            *[f"torso_velo_{a}" for a in "xyz"],
            *[f"shoulder_velo_{a}" for a in "xyz"],
        ],
    )
    landmarks = pd.read_csv(
        ROOT / "data/full_sig/landmarks.csv",
        usecols=[
            "session_pitch", "time",
            *[f"shoulder_jc_{a}" for a in "xyz"],
            *[f"elbow_jc_{a}" for a in "xyz"],
            *[f"thorax_prox_{a}" for a in "xyz"],
            *[f"thorax_dist_{a}" for a in "xyz"],
            *[f"thorax_ap_{a}" for a in "xyz"],
        ],
    )
    data = energy.merge(kinetics, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(velocities, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(landmarks, on=["session_pitch", "time"], validate="one_to_one")
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    data = data.sort_values(["session_pitch", "time"])
    for axis in "xyz":
        data[f"upper_arm_mid_{axis}"] = (
            data[f"shoulder_jc_{axis}"] + data[f"elbow_jc_{axis}"]
        ) / 2
        data[f"shoulder_jc_vel_{axis}"] = data.groupby("session_pitch", sort=False)[
            f"shoulder_jc_{axis}"
        ].transform(
            lambda s: np.gradient(
                s.to_numpy(float), data.loc[s.index, "time"].to_numpy(float)
            )
        )
        data[f"upper_arm_mid_vel_{axis}"] = data.groupby("session_pitch", sort=False)[
            f"upper_arm_mid_{axis}"
        ].transform(
            lambda s: np.gradient(
                s.to_numpy(float), data.loc[s.index, "time"].to_numpy(float)
            )
        )
        for smooth_window in [5, 7, 9, 11, 15, 21]:
            data[f"shoulder_jc_vel_sg{smooth_window}_{axis}"] = data.groupby(
                "session_pitch", sort=False
            )[f"shoulder_jc_{axis}"].transform(
                lambda s: savgol_filter(
                    s.to_numpy(float), smooth_window, 3,
                    deriv=1, delta=1 / 360, mode="interp"
                )
            )

    window = data[(data.time >= data.fp_poi_time) & (data.time <= data.MER_time)].copy()
    jfp = window.shoulder_energy_transfer_jfp.to_numpy(float)
    velocity = window[[f"shoulder_jc_vel_{a}" for a in "xyz"]].to_numpy(float)
    midpoint_velocity = window[
        [f"upper_arm_mid_vel_{a}" for a in "xyz"]
    ].to_numpy(float)
    rows = []
    for side in ["thorax", "upper_arm"]:
        force = window[[f"shoulder_{side}_force_{a}" for a in "xyz"]].to_numpy(float)
        dot = np.einsum("ij,ij->i", force, velocity)
        rows.append(metrics(f"{side}_force_dot_d_shoulder_jc", jfp, dot))
        rows.append(metrics(f"negative_{side}_force_dot_d_shoulder_jc", jfp, -dot))
        midpoint_dot = np.einsum("ij,ij->i", force, midpoint_velocity)
        rows.append(metrics(f"{side}_force_dot_d_upper_arm_mid", jfp, midpoint_dot))
        rows.append(metrics(f"negative_{side}_force_dot_d_upper_arm_mid", jfp, -midpoint_dot))

    # Reconstruct an orthonormal thorax frame from the three model landmarks.
    prox = window[[f"thorax_prox_{a}" for a in "xyz"]].to_numpy(float)
    dist = window[[f"thorax_dist_{a}" for a in "xyz"]].to_numpy(float)
    ap_point = window[[f"thorax_ap_{a}" for a in "xyz"]].to_numpy(float)
    axial = prox - dist
    axial /= np.linalg.norm(axial, axis=1, keepdims=True)
    ap_raw = ap_point - (prox + dist) / 2
    ap = ap_raw - np.einsum("ij,ij->i", ap_raw, axial)[:, None] * axial
    ap /= np.linalg.norm(ap, axis=1, keepdims=True)
    ml = np.cross(ap, axial)
    ml /= np.linalg.norm(ml, axis=1, keepdims=True)
    base = np.stack([ml, ap, axial], axis=2)
    for side in ["thorax", "upper_arm"]:
        local_force = window[
            [f"shoulder_{side}_force_{a}" for a in "xyz"]
        ].to_numpy(float)
        for perm in permutations(range(3)):
            for signs in product([-1.0, 1.0], repeat=3):
                signed_permutation = np.zeros((3, 3))
                for local_axis, base_axis in enumerate(perm):
                    signed_permutation[base_axis, local_axis] = signs[local_axis]
                if np.linalg.det(signed_permutation) < 0:
                    continue
                rotation = np.einsum("nij,jk->nik", base, signed_permutation)
                lab_force = np.einsum("nij,nj->ni", rotation, local_force)
                candidate = np.einsum("ij,ij->i", lab_force, velocity)
                label = "".join("map"[i] for i in perm) + "_" + "".join(
                    "+" if x > 0 else "-" for x in signs
                )
                rows.append(metrics(
                    f"rotated_{side}_force__{label}", jfp, candidate
                ))
    # Best structural thorax mapping above: local x=+AP, y=-ML, z=+axial.
    best_map = np.zeros((3, 3))
    best_map[1, 0] = 1
    best_map[0, 1] = -1
    best_map[2, 2] = 1
    best_rotation = np.einsum("nij,jk->nik", base, best_map)
    thorax_local_force = window[
        [f"shoulder_thorax_force_{a}" for a in "xyz"]
    ].to_numpy(float)
    best_lab_force = np.einsum("nij,nj->ni", best_rotation, thorax_local_force)
    for smooth_window in [5, 7, 9, 11, 15, 21]:
        smooth_velocity = window[
            [f"shoulder_jc_vel_sg{smooth_window}_{a}" for a in "xyz"]
        ].to_numpy(float)
        candidate = np.einsum("ij,ij->i", best_lab_force, smooth_velocity)
        rows.append(metrics(f"rotated_thorax_force_savgol_{smooth_window}", jfp, candidate))

    official_stp = window.shoulder_energy_transfer_stp.to_numpy(float)
    official_thorax = (
        window.thorax_dist_seg_pwr + window.shoulder_energy_transfer_jfp
    ).to_numpy(float)
    official_arm = (
        window.upper_arm_prox_seg_pwr - window.shoulder_energy_transfer_jfp
    ).to_numpy(float)
    torso = np.radians(window[[f"torso_velo_{a}" for a in "xyz"]].to_numpy(float))
    shoulder = np.radians(window[[f"shoulder_velo_{a}" for a in "xyz"]].to_numpy(float))
    angular_candidates = {"torso": torso, "shoulder": shoulder, "torso_plus_shoulder": torso + shoulder}
    thorax_moment = window[[f"shoulder_thorax_moment_{a}" for a in "xyz"]].to_numpy(float)
    reconstructed_thorax_stp = -np.einsum("ij,ij->i", thorax_moment, torso)
    reconstructed_jfp_balance = (
        reconstructed_thorax_stp - window.thorax_dist_seg_pwr.to_numpy(float)
    )
    rows.append(metrics(
        "jfp_from_thorax_power_balance", jfp, reconstructed_jfp_balance
    ))
    reconstructed_arm_stp = (
        window.upper_arm_prox_seg_pwr.to_numpy(float) - reconstructed_jfp_balance
    )
    reconstructed_magnitude = np.minimum(
        np.abs(reconstructed_thorax_stp), np.abs(reconstructed_arm_stp)
    )
    reconstructed_transfer_stp = np.select(
        [
            (reconstructed_thorax_stp < 0) & (reconstructed_arm_stp > 0),
            (reconstructed_thorax_stp > 0) & (reconstructed_arm_stp < 0),
        ],
        [reconstructed_magnitude, -reconstructed_magnitude],
        default=0.0,
    )
    rows.append(metrics(
        "stp_from_csv_power_balance",
        official_stp,
        reconstructed_transfer_stp,
    ))
    rows.append(metrics(
        "stp_plus_jfp_from_csv_power_balance",
        official_stp + jfp,
        reconstructed_transfer_stp + reconstructed_jfp_balance,
    ))
    for side in ["thorax", "upper_arm"]:
        moment = window[[f"shoulder_{side}_moment_{a}" for a in "xyz"]].to_numpy(float)
        for angular_name, angular in angular_candidates.items():
            dot = np.einsum("ij,ij->i", moment, angular)
            for sign_name, signed in [("positive", dot), ("negative", -dot)]:
                rows.append(metrics(f"thorax_stp__{sign_name}_{side}_moment_dot_{angular_name}", official_thorax, signed))
                rows.append(metrics(f"upper_arm_stp__{sign_name}_{side}_moment_dot_{angular_name}", official_arm, signed))

    thorax = official_thorax
    arm = official_arm
    magnitude = np.minimum(np.abs(thorax), np.abs(arm))
    reconstructed_stp = np.select(
        [(thorax < 0) & (arm > 0), (thorax > 0) & (arm < 0)],
        [magnitude, -magnitude], default=0.0,
    )
    rows.append(metrics("bottleneck_from_recovered_two_sides", official_stp, reconstructed_stp))
    result = pd.DataFrame(rows).sort_values(["candidate"])
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "candidate_validation.csv", index=False)
    print(f"coverage: rows={len(data)}, FP-MER rows={len(window)}, pitches={window.session_pitch.nunique()}")
    print("\nJFP candidates")
    print(result[result.candidate.str.contains("shoulder_jc|upper_arm_mid|jfp_from", regex=True)].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBest thorax-side STP candidates")
    print(result[result.candidate.str.startswith("thorax_stp")].sort_values("mae_w").head(8).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBest upper-arm-side STP candidates")
    print(result[result.candidate.str.startswith("upper_arm_stp")].sort_values("mae_w").head(8).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nExact transfer reconstruction")
    print(result[result.candidate.str.contains("bottleneck_from|from_csv_power_balance", regex=True)].to_string(index=False, float_format=lambda x: f"{x:.12f}"))
    print("\nBest landmark-rotated force candidates")
    print(result[result.candidate.str.startswith("rotated_")].sort_values("mae_w").head(12).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
