"""Reconstruct and decompose upper-arm-side shoulder STP by arm-frame axes."""

from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("upper_arm_stp_axis_outputs")


def score(name: str, official: np.ndarray, candidate: np.ndarray) -> dict:
    return {
        "candidate": name,
        "r": np.corrcoef(official, candidate)[0, 1],
        "mae": np.mean(np.abs(official - candidate)),
    }


def main() -> None:
    energy = pd.read_csv(
        ROOT / "data/full_sig/energy_flow.csv",
        usecols=[
            "session_pitch", "time", "fp_poi_time", "MER_time", "BR_time",
            "shoulder_energy_transfer_jfp", "shoulder_energy_transfer_stp",
            "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr",
        ],
    ).dropna()
    moments = pd.read_csv(
        ROOT / "data/full_sig/forces_moments.csv",
        usecols=[
            "session_pitch", "time",
            *[f"shoulder_thorax_moment_{a}" for a in "xyz"],
            *[f"shoulder_upper_arm_moment_{a}" for a in "xyz"],
        ],
    ).dropna()
    velos = pd.read_csv(
        ROOT / "data/full_sig/joint_velos.csv",
        usecols=[
            "session_pitch", "time",
            *[f"torso_velo_{a}" for a in "xyz"],
            *[f"shoulder_velo_{a}" for a in "xyz"],
        ],
    ).dropna()
    angles = pd.read_csv(
        ROOT / "data/full_sig/joint_angles.csv",
        usecols=["session_pitch", "time", *[f"shoulder_angle_{a}" for a in "xyz"]],
    ).dropna()
    data = energy.merge(moments, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(velos, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(angles, on=["session_pitch", "time"], validate="one_to_one")
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    w = data[(data.time >= data.fp_poi_time) & (data.time <= data.BR_time)].copy()
    angle_values = w[[f"shoulder_angle_{a}" for a in "xyz"]].to_numpy(float)
    m_t = w[[f"shoulder_thorax_moment_{a}" for a in "xyz"]].to_numpy(float)
    m_a = w[[f"shoulder_upper_arm_moment_{a}" for a in "xyz"]].to_numpy(float)
    official_arm_stp = (
        w.upper_arm_prox_seg_pwr - w.shoulder_energy_transfer_jfp
    ).to_numpy(float)
    torso_omega = np.radians(
        w[[f"torso_velo_{a}" for a in "xyz"]].to_numpy(float)
    )
    shoulder_relative = np.radians(
        w[[f"shoulder_velo_{a}" for a in "xyz"]].to_numpy(float)
    )

    candidates = []
    best_error = np.inf
    best_label = None
    best_parameters = None
    best_direction = None
    sample = np.arange(0, len(w), 10)
    sample_angles = angle_values[sample]
    sample_m_t = m_t[sample]
    sample_m_a = m_a[sample]
    for perm in permutations(range(3)):
        for signs in product([-1.0, 1.0], repeat=3):
            for offsets in product([-90.0, 0.0, 90.0], repeat=3):
                candidate_angles = (
                    sample_angles[:, perm] * np.asarray(signs) + np.asarray(offsets)
                )
                for sequence in ["ZYZ", "zyz"]:
                    matrices = Rotation.from_euler(
                        sequence, candidate_angles, degrees=True
                    ).as_matrix()
                    for direction in ["R", "RT"]:
                        transformed = (
                            np.einsum("nij,nj->ni", matrices, sample_m_t)
                            if direction == "R"
                            else np.einsum("nji,nj->ni", matrices, sample_m_t)
                        )
                        label = (
                            f"{sequence}_perm{perm}_sign{signs}_offset{offsets}_{direction}"
                        )
                        moment_error = np.mean(
                            np.linalg.norm(transformed - sample_m_a, axis=1)
                        )
                        moment_r = np.corrcoef(
                            transformed.ravel(), sample_m_a.ravel()
                        )[0, 1]
                        candidates.append({
                            "candidate": label,
                            "moment_vector_r": moment_r,
                            "moment_vector_mae_nm": moment_error,
                        })
                        if moment_error < best_error:
                            best_error = moment_error
                            best_label = label
                            best_parameters = (perm, signs, offsets, sequence)
                            best_direction = direction
    validation = pd.DataFrame(candidates).sort_values("moment_vector_mae_nm")
    perm, signs, offsets, sequence = best_parameters
    best_angles = angle_values[:, perm] * np.asarray(signs) + np.asarray(offsets)
    best_matrix = Rotation.from_euler(sequence, best_angles, degrees=True).as_matrix()
    best_transformed_moment = (
        np.einsum("nij,nj->ni", best_matrix, m_t)
        if best_direction == "R"
        else np.einsum("nji,nj->ni", best_matrix, m_t)
    )
    full_moment_r = np.corrcoef(best_transformed_moment.ravel(), m_a.ravel())[0, 1]
    full_moment_mae = np.mean(np.linalg.norm(best_transformed_moment - m_a, axis=1))
    torso_in_arm = (
        np.einsum("nij,nj->ni", best_matrix, torso_omega)
        if best_direction == "R"
        else np.einsum("nji,nj->ni", best_matrix, torso_omega)
    )
    arm_absolute = shoulder_relative + torso_in_arm
    axis_power = m_a * arm_absolute
    reconstructed = axis_power.sum(axis=1)
    power_validation = pd.DataFrame([
        score("relative_shoulder_velocity_only", official_arm_stp, (m_a * shoulder_relative).sum(axis=1)),
        score("absolute_arm_velocity_reconstructed", official_arm_stp, reconstructed),
    ])

    for index, axis in enumerate("xyz"):
        w[f"arm_stp_{axis}_w"] = axis_power[:, index]
    w["arm_stp_axis_sum_w"] = reconstructed
    w["official_arm_stp_w"] = official_arm_stp
    jfp = w.shoulder_energy_transfer_jfp
    thorax_stp = w.thorax_dist_seg_pwr + jfp
    arm_stp = w.upper_arm_prox_seg_pwr - jfp
    w["arm_limited_forward"] = (
        (thorax_stp < 0) & (arm_stp > 0) & (arm_stp.abs() < thorax_stp.abs())
        & (w.shoulder_energy_transfer_stp > 0)
    )
    active = w[w.arm_limited_forward].copy()
    per_pitch = active.groupby("session_pitch", as_index=False).agg(
        frame_n=("time", "size"),
        x_mean_w=("arm_stp_x_w", "mean"),
        y_mean_w=("arm_stp_y_w", "mean"),
        z_mean_w=("arm_stp_z_w", "mean"),
        total_mean_w=("official_arm_stp_w", "mean"),
    )
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session"])
    per_pitch = per_pitch.merge(meta, on="session_pitch", validate="one_to_one")
    athlete = per_pitch.groupby("session", as_index=False).agg(
        x_mean_w=("x_mean_w", "mean"), y_mean_w=("y_mean_w", "mean"),
        z_mean_w=("z_mean_w", "mean"), total_mean_w=("total_mean_w", "mean"),
    )
    summary = athlete[["x_mean_w", "y_mean_w", "z_mean_w", "total_mean_w"]].agg(
        ["mean", "median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]
    ).T
    summary.columns = ["mean", "median", "q1", "q3"]

    OUT.mkdir(exist_ok=True)
    validation.to_csv(OUT / "rotation_validation.csv", index=False)
    power_validation.to_csv(OUT / "power_validation.csv", index=False)
    per_pitch.to_csv(OUT / "per_pitch.csv", index=False)
    athlete.to_csv(OUT / "per_pitcher.csv", index=False)
    summary.to_csv(OUT / "axis_summary.csv")
    print(f"coverage: rows={len(w)}, pitches={w.session_pitch.nunique()}, arm-limited pitches={len(per_pitch)}, pitchers={len(athlete)}")
    print("best rotation", best_label)
    print(f"full best moment validation: r={full_moment_r:.6f}, MAE={full_moment_mae:.6f} Nm")
    print(validation.head(5).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\npower validation")
    print(power_validation.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\naxis summary during arm-limited forward STP")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
