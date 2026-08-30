"""Reconstruct throwing-upper-arm global angular velocity from OBP C3D markers.

This first-stage validation intentionally analyzes one right-handed pitch.  It
uses the official shoulder joint centre from landmarks.csv and the Visual3D
upper-arm anatomical-frame definition (proximal joint plus lateral/medial
distal elbow targets).  The reconstruction is not accepted unless moment dot
angular-velocity reproduces the official upper-arm-side shoulder STP.
"""

from pathlib import Path
from itertools import permutations, product

import ezc3d
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
C3D_ROOT = ROOT / "data/c3d"
FULL_SIG = ROOT / "data/full_sig"
PITCH = "3034_2"
RATE_HZ = 360.0


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def lowpass(values: np.ndarray) -> np.ndarray:
    sos = butter(4, 20.0, btype="lowpass", fs=RATE_HZ, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def find_c3d() -> Path:
    session, pitch = PITCH.split("_")
    matches = list(C3D_ROOT.glob(f"**/*_{int(session):06d}_*_{int(pitch):03d}_*.c3d"))
    matches = [path for path in matches if "_model.c3d" not in path.name]
    if len(matches) != 1:
        raise ValueError(f"Expected one C3D for {PITCH}, found {matches}")
    return matches[0]


def angular_velocity_body(rotation: np.ndarray) -> np.ndarray:
    """Return global angular velocity resolved in the current body frame."""
    relative = np.einsum("nij,nkj->nik", rotation[1:], rotation[:-1])
    omega_lab_mid = Rotation.from_matrix(relative).as_rotvec() * RATE_HZ
    omega_lab = np.empty((len(rotation), 3))
    omega_lab[1:-1] = (omega_lab_mid[:-1] + omega_lab_mid[1:]) / 2.0
    omega_lab[0] = omega_lab_mid[0]
    omega_lab[-1] = omega_lab_mid[-1]
    return np.einsum("nji,nj->ni", rotation, omega_lab)


def anatomical_frame(shoulder: np.ndarray, lateral: np.ndarray, medial: np.ndarray) -> np.ndarray:
    elbow = (lateral + medial) / 2.0
    z_axis = unit(shoulder - elbow)
    raw_x = lateral - medial
    x_axis = unit(raw_x - np.sum(raw_x * z_axis, axis=1, keepdims=True) * z_axis)
    y_axis = unit(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=2)


def rigid_rotation(template: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Least-squares proper rotation mapping a static marker template to frames."""
    template_centered = template - template.mean(axis=0)
    rotations = []
    for frame in observed:
        frame_centered = frame - frame.mean(axis=0)
        u, _, vt = np.linalg.svd(template_centered.T @ frame_centered)
        candidate = vt.T @ u.T
        if np.linalg.det(candidate) < 0:
            vt[-1] *= -1
            candidate = vt.T @ u.T
        rotations.append(candidate)
    return np.asarray(rotations)


def score(official: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(official) & np.isfinite(candidate)
    return (
        float(np.corrcoef(official[valid], candidate[valid])[0, 1]),
        float(np.mean(np.abs(official[valid] - candidate[valid]))),
    )


def main() -> None:
    path = find_c3d()
    c3d = ezc3d.c3d(str(path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    points = c3d["data"]["points"][:3].transpose(2, 1, 0)
    if float(c3d["header"]["points"]["frame_rate"]) != RATE_HZ:
        raise ValueError("Unexpected C3D point rate; no fallback permitted")
    markers = {
        name: lowpass(points[:, labels.index(name), :])
        for name in ("RELB", "RMELB", "RUPA")
    }
    static_path = next(path.parent.glob("*_model.c3d"))
    static_c3d = ezc3d.c3d(str(static_path))
    static_labels = static_c3d["parameters"]["POINT"]["LABELS"]["value"]
    static_points = static_c3d["data"]["points"][:3].transpose(2, 1, 0)
    static = {
        name: np.mean(static_points[:, static_labels.index(name), :], axis=0)
        for name in ("CLAV", "C7", "STRN", "T10", "LSHO", "RSHO", "RELB", "RMELB", "RUPA")
    }
    thorax_prox = (static["CLAV"] + static["C7"]) / 2.0
    thorax_dist = (static["STRN"] + static["T10"]) / 2.0
    thorax_z = (thorax_prox - thorax_dist) / np.linalg.norm(thorax_prox - thorax_dist)
    shoulder_width = np.linalg.norm(static["RSHO"] - static["LSHO"])
    static_shoulder = static["RSHO"] - 0.17 * shoulder_width * thorax_z
    static_frame = anatomical_frame(
        static_shoulder[None], static["RELB"][None], static["RMELB"][None]
    )[0]
    static_cluster = np.stack([
        static_shoulder, static["RELB"], static["RMELB"], static["RUPA"]
    ])

    energy = pd.read_csv(
        FULL_SIG / "energy_flow.csv",
        usecols=[
            "session_pitch", "time", "fp_poi_time", "BR_time",
            "shoulder_energy_transfer_jfp", "upper_arm_prox_seg_pwr",
        ],
    )
    moments = pd.read_csv(
        FULL_SIG / "forces_moments.csv",
        usecols=["session_pitch", "time", *[f"shoulder_upper_arm_moment_{a}" for a in "xyz"]],
    )
    landmarks = pd.read_csv(
        FULL_SIG / "landmarks.csv",
        usecols=["session_pitch", "time", *[f"shoulder_jc_{a}" for a in "xyz"]],
    )
    data = energy.merge(moments, on=["session_pitch", "time"], validate="one_to_one")
    data = data.merge(landmarks, on=["session_pitch", "time"], validate="one_to_one")
    data = data[data.session_pitch.astype(str) == PITCH].copy()
    if data.empty:
        raise ValueError(f"Pitch {PITCH} missing from full-signal tables")

    official = (
        data.upper_arm_prox_seg_pwr - data.shoulder_energy_transfer_jfp
    ).to_numpy(float)
    moment = data[[f"shoulder_upper_arm_moment_{a}" for a in "xyz"]].to_numpy(float)
    shoulder = data[[f"shoulder_jc_{a}" for a in "xyz"]].to_numpy(float)
    results = []
    for offset in range(points.shape[0] - len(data) + 1):
        lateral = markers["RELB"][offset:offset + len(data)]
        medial = markers["RMELB"][offset:offset + len(data)]
        direct_rotation = anatomical_frame(shoulder, lateral, medial)
        observed_cluster = np.stack([
            shoulder, lateral, medial, markers["RUPA"][offset:offset + len(data)]
        ], axis=1)
        tracked_rotation = np.einsum(
            "nij,jk->nik", rigid_rotation(static_cluster, observed_cluster), static_frame
        )
        for method, rotation in (("direct", direct_rotation), ("static_cluster", tracked_rotation)):
            omega = angular_velocity_body(rotation)
            for permutation in permutations(range(3)):
                for signs in product((-1.0, 1.0), repeat=3):
                    label = f"{method}_perm{permutation}_sign{signs}"
                    candidate_omega = omega[:, permutation] * np.asarray(signs)
                    reconstructed = np.sum(moment * candidate_omega, axis=1)
                    whole_r, whole_mae = score(official, reconstructed)
                    window = (data.time >= data.fp_poi_time) & (data.time <= data.BR_time)
                    window_r, window_mae = score(official[window], reconstructed[window])
                    results.append({
                        "offset_frames": offset,
                        "axis_orientation": label,
                        "whole_r": whole_r,
                        "whole_mae_w": whole_mae,
                        "fp_br_r": window_r,
                        "fp_br_mae_w": window_mae,
                    })

    validation = pd.DataFrame(results).sort_values(
        ["fp_br_mae_w", "fp_br_r"], ascending=[True, False]
    )
    print(f"C3D={path}")
    print(f"coverage: full-signal frames={len(data)}, C3D frames={len(points)}")
    print(validation.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
