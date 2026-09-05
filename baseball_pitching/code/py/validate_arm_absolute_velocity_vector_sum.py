"""Test whether published relative arm and absolute torso velocities recover arm STP.

The thorax-to-upper-arm coordinate rotation is estimated independently at each
frame from the paired shoulder force and moment vectors published in both
coordinate systems.  This is a diagnostic, not an official Visual3D export.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/full_sig"
OUT = Path(__file__).with_name("arm_absolute_velocity_vector_sum_outputs")


def basis(force: np.ndarray, moment: np.ndarray) -> np.ndarray:
    e1 = force / np.linalg.norm(force)
    remainder = moment - e1 * np.dot(moment, e1)
    e2 = remainder / np.linalg.norm(remainder)
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def metrics(group: pd.DataFrame, label: str) -> dict:
    official = group["official_arm_stp_w"].to_numpy(float)
    predicted = group["reconstructed_arm_stp_w"].to_numpy(float)
    error = official - predicted
    return {
        "group": label,
        "frames": len(group),
        "pitches": group.session_pitch.nunique(),
        "r": np.corrcoef(official, predicted)[0, 1],
        "mae_w": np.mean(np.abs(error)),
        "rmse_w": np.sqrt(np.mean(error**2)),
        "bias_official_minus_reconstructed_w": np.mean(error),
    }


def main() -> None:
    energy = pd.read_csv(
        DATA / "energy_flow.csv",
        usecols=[
            "session_pitch", "time", "fp_poi_time", "BR_time",
            "shoulder_energy_transfer_jfp", "upper_arm_prox_seg_pwr",
        ],
    )
    kinetics = pd.read_csv(
        DATA / "forces_moments.csv",
        usecols=[
            "session_pitch", "time",
            *[
                f"shoulder_{reference}_{kind}_{axis}"
                for reference in ("thorax", "upper_arm")
                for kind in ("force", "moment")
                for axis in "xyz"
            ],
        ],
    )
    velocities = pd.read_csv(
        DATA / "joint_velos.csv",
        usecols=[
            "session_pitch", "time",
            *[f"torso_velo_{axis}" for axis in "xyz"],
            *[f"shoulder_velo_{axis}" for axis in "xyz"],
        ],
    )
    handedness = pd.read_csv(
        ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", "p_throws"]
    )
    data = energy.merge(kinetics, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(velocities, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(handedness, on="session_pitch", validate="many_to_one")
    data = data[(data.time >= data.fp_poi_time) & (data.time <= data.BR_time)].dropna().copy()

    rotations = []
    for row in data.itertuples(index=False):
        thorax_force = np.array([getattr(row, f"shoulder_thorax_force_{a}") for a in "xyz"])
        arm_force = np.array([getattr(row, f"shoulder_upper_arm_force_{a}") for a in "xyz"])
        thorax_moment = np.array([getattr(row, f"shoulder_thorax_moment_{a}") for a in "xyz"])
        arm_moment = np.array([getattr(row, f"shoulder_upper_arm_moment_{a}") for a in "xyz"])
        rotations.append(basis(arm_force, arm_moment) @ basis(thorax_force, thorax_moment).T)
    rotations = np.asarray(rotations)

    torso = np.radians(data[[f"torso_velo_{a}" for a in "xyz"]].to_numpy(float))
    relative = np.radians(data[[f"shoulder_velo_{a}" for a in "xyz"]].to_numpy(float))
    arm_absolute = relative + np.einsum("nij,nj->ni", rotations, torso)
    arm_moment = data[[f"shoulder_upper_arm_moment_{a}" for a in "xyz"]].to_numpy(float)
    data["reconstructed_arm_stp_w"] = np.sum(arm_moment * arm_absolute, axis=1)
    data["official_arm_stp_w"] = (
        data.upper_arm_prox_seg_pwr - data.shoulder_energy_transfer_jfp
    )

    summary = [metrics(data, "all")]
    summary.extend(metrics(group, hand) for hand, group in data.groupby("p_throws"))
    summary = pd.DataFrame(summary)
    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "validation_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
