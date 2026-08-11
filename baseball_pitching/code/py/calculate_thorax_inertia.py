"""Reconstruct Visual3D RTA (Thorax/Ab) inertia from static model C3Ds.

The equations mirror the project's v6_model_hybrid_lm.mdh dimensions and
Visual3D's default elliptical-cylinder inertial properties.  The analysis unit
is one session/pitcher because each session has one static model C3D.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd


REQUIRED_MARKERS = ("C7", "CLAV", "RSHO", "LSHO", "STRN", "T10")
RTA_MASS_FRACTION = 0.355
KNOWN_LABEL_PREFIX = "Skeleton_001_"


def static_marker_centres(
    model_path: Path,
) -> tuple[dict[str, np.ndarray], float, dict[str, np.ndarray]]:
    c3d = ezc3d.c3d(str(model_path))
    units = c3d["parameters"]["POINT"]["UNITS"]["value"]
    if units != ["m"]:
        raise ValueError(f"{model_path}: expected POINT unit ['m'], got {units!r}")

    raw_labels = [label.strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    labels = [
        label.removeprefix(KNOWN_LABEL_PREFIX) if label.startswith(KNOWN_LABEL_PREFIX) else label
        for label in raw_labels
    ]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{model_path}: duplicate labels after namespace normalization")
    missing = [marker for marker in REQUIRED_MARKERS if marker not in labels]
    if missing:
        raise ValueError(f"{model_path}: missing markers {missing}")

    points = np.asarray(c3d["data"]["points"], dtype=float)
    centres: dict[str, np.ndarray] = {}
    frame_spreads: list[float] = []
    marker_xyz: dict[str, np.ndarray] = {}
    common_valid = np.ones(points.shape[2], dtype=bool)
    for marker in REQUIRED_MARKERS:
        marker_data = points[:, labels.index(marker), :]
        valid = np.isfinite(marker_data[:3]).all(axis=0) & (marker_data[3] >= 0)
        if not valid.any():
            raise ValueError(f"{model_path}: marker {marker} has no valid frames")
        xyz = marker_data[:3, valid].T
        centres[marker] = xyz.mean(axis=0)
        frame_spreads.append(float(np.max(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1))))
        marker_xyz[marker] = marker_data[:3].T
        common_valid &= valid
    if not common_valid.any():
        raise ValueError(f"{model_path}: required markers have no common valid frame")

    frame_points = {marker: xyz[common_valid] for marker, xyz in marker_xyz.items()}
    frame_prox = 0.5 * (frame_points["CLAV"] + frame_points["C7"])
    frame_dist = 0.5 * (frame_points["STRN"] + frame_points["T10"])
    frame_dimensions = {
        "length": np.linalg.norm(frame_prox - frame_dist, axis=1),
        "depth_radius": 0.5 * np.linalg.norm(frame_points["C7"] - frame_points["CLAV"], axis=1),
        "shoulder_radius": 0.5
        * np.linalg.norm(frame_points["RSHO"] - frame_points["LSHO"], axis=1),
    }
    return centres, max(frame_spreads), frame_dimensions


def reconstruct_session(row: pd.Series, model_path: Path) -> dict[str, float | str]:
    markers, max_marker_spread_m, frame_dimensions = static_marker_centres(model_path)
    prox = 0.5 * (markers["CLAV"] + markers["C7"])
    dist = 0.5 * (markers["STRN"] + markers["T10"])

    depth_radius_m = 0.5 * np.linalg.norm(markers["C7"] - markers["CLAV"])
    shoulder_radius_m = 0.5 * np.linalg.norm(markers["RSHO"] - markers["LSHO"])
    length_m = np.linalg.norm(prox - dist)
    body_mass_kg = float(row["session_mass_kg"])
    height_m = float(row["session_height_m"])
    segment_mass_kg = RTA_MASS_FRACTION * body_mass_kg

    ixx = segment_mass_kg * (3.0 * depth_radius_m**2 + length_m**2) / 12.0
    iyy = segment_mass_kg * (3.0 * shoulder_radius_m**2 + length_m**2) / 12.0
    izz = segment_mass_kg * (depth_radius_m**2 + shoulder_radius_m**2) / 4.0
    frame_izz = (
        segment_mass_kg
        * (frame_dimensions["depth_radius"] ** 2 + frame_dimensions["shoulder_radius"] ** 2)
        / 4.0
    )

    return {
        "session": str(row["session"]),
        "modelname_new": str(row["modelname_new"]),
        "body_mass_kg": body_mass_kg,
        "height_m": height_m,
        "rta_mass_kg": segment_mass_kg,
        "rta_length_m": length_m,
        "rta_depth_radius_m": depth_radius_m,
        "rta_shoulder_radius_m": shoulder_radius_m,
        "rta_ixx_kg_m2": ixx,
        "rta_iyy_kg_m2": iyy,
        "rta_izz_kg_m2": izz,
        "rta_izz_frame_mean_kg_m2": float(frame_izz.mean()),
        "rta_izz_frame_sd_kg_m2": float(frame_izz.std(ddof=1)) if len(frame_izz) > 1 else 0.0,
        "rta_izz_frame_range_kg_m2": float(np.ptp(frame_izz)),
        "rta_izz_max_frame_deviation_pct": float(np.max(np.abs(frame_izz - izz)) / izz * 100.0),
        "rta_izz_mean_coordinate_difference_pct": float((izz - frame_izz.mean()) / izz * 100.0),
        "max_static_marker_spread_m": max_marker_spread_m,
    }


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    return {
        "mae": float(np.mean(np.abs(residual))),
        "median_ape_pct": float(np.median(np.abs(residual / actual)) * 100.0),
        "p95_ape_pct": float(np.percentile(np.abs(residual / actual), 95) * 100.0),
        "prediction_r2": float(1.0 - np.sum(residual**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def leave_one_out_zero_intercept(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(y)
    total_xy = float(np.dot(x, y))
    total_xx = float(np.dot(x, x))
    for index in range(len(y)):
        coefficient = (total_xy - x[index] * y[index]) / (total_xx - x[index] ** 2)
        predictions[index] = coefficient * x[index]
    return predictions


def fit_allometric(
    mass: np.ndarray,
    height: np.ndarray,
    inertia: np.ndarray,
    free_mass_exponent: bool,
    free_height_exponent: bool,
    fixed_mass_exponent: float = 0.0,
    fixed_height_exponent: float = 0.0,
) -> tuple[float, float, float]:
    log_mass = np.log(mass)
    log_height = np.log(height)
    target = (
        np.log(inertia)
        - fixed_mass_exponent * log_mass
        - fixed_height_exponent * log_height
    )
    columns = [np.ones(len(inertia))]
    if free_mass_exponent:
        columns.append(log_mass)
    if free_height_exponent:
        columns.append(log_height)
    coefficients = np.linalg.lstsq(np.column_stack(columns), target, rcond=None)[0]
    position = 1
    mass_exponent = fixed_mass_exponent
    height_exponent = fixed_height_exponent
    if free_mass_exponent:
        mass_exponent = float(coefficients[position])
        position += 1
    if free_height_exponent:
        height_exponent = float(coefficients[position])
    return float(np.exp(coefficients[0])), mass_exponent, height_exponent


def allometric_prediction(
    mass: np.ndarray,
    height: np.ndarray,
    coefficient: float,
    mass_exponent: float,
    height_exponent: float,
) -> np.ndarray:
    return coefficient * mass**mass_exponent * height**height_exponent


def leave_one_out_allometric(
    mass: np.ndarray,
    height: np.ndarray,
    inertia: np.ndarray,
    free_mass_exponent: bool,
    free_height_exponent: bool,
    fixed_mass_exponent: float = 0.0,
    fixed_height_exponent: float = 0.0,
) -> np.ndarray:
    predictions = np.empty_like(inertia)
    for index in range(len(inertia)):
        keep = np.arange(len(inertia)) != index
        model = fit_allometric(
            mass[keep],
            height[keep],
            inertia[keep],
            free_mass_exponent,
            free_height_exponent,
            fixed_mass_exponent,
            fixed_height_exponent,
        )
        predictions[index] = allometric_prediction(
            mass[index : index + 1], height[index : index + 1], *model
        )[0]
    return predictions


def bootstrap_allometric_ci(
    mass: np.ndarray,
    height: np.ndarray,
    inertia: np.ndarray,
    free_mass_exponent: bool,
    free_height_exponent: bool,
    fixed_mass_exponent: float = 0.0,
    fixed_height_exponent: float = 0.0,
    repetitions: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260811)
    estimates = np.empty((repetitions, 3))
    for repetition in range(repetitions):
        indices = rng.integers(0, len(inertia), len(inertia))
        estimates[repetition] = fit_allometric(
            mass[indices],
            height[indices],
            inertia[indices],
            free_mass_exponent,
            free_height_exponent,
            fixed_mass_exponent,
            fixed_height_exponent,
        )
    return np.percentile(estimates, 2.5, axis=0), np.percentile(estimates, 97.5, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--metadata", type=Path, default=project_root / "data" / "metadata.csv")
    parser.add_argument("--c3d-root", type=Path, default=project_root / "data" / "c3d")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, dtype={"session": str})
    required_columns = {"session", "session_mass_kg", "session_height_m", "modelname_new"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"metadata missing columns: {missing_columns}")

    consistency = metadata.groupby("session")[list(required_columns - {"session"})].nunique(dropna=False)
    inconsistent = consistency[(consistency > 1).any(axis=1)]
    if not inconsistent.empty:
        raise ValueError(f"session-level metadata are inconsistent: {inconsistent.index.tolist()}")
    sessions = metadata[list(required_columns)].drop_duplicates("session").sort_values("session")

    model_files: dict[str, Path] = {}
    for model_path in args.c3d_root.glob("*/*_model.c3d"):
        if model_path.name in model_files:
            raise ValueError(f"duplicate model filename: {model_path.name}")
        model_files[model_path.name] = model_path

    missing_models = sorted(set(sessions["modelname_new"]) - set(model_files))
    if missing_models:
        raise FileNotFoundError(f"missing model C3Ds ({len(missing_models)}): {missing_models[:10]}")

    records = [
        reconstruct_session(row, model_files[str(row["modelname_new"])])
        for _, row in sessions.iterrows()
    ]
    result = pd.DataFrame.from_records(records)

    # Physically dimensioned height-only approximation: Izz = c * body_mass * height^2.
    x = (result["body_mass_kg"] * result["height_m"] ** 2).to_numpy()
    y = result["rta_izz_kg_m2"].to_numpy()
    coefficient = float(np.dot(x, y) / np.dot(x, x))
    height_prediction = leave_one_out_zero_intercept(x, y)

    mass_x = result["body_mass_kg"].to_numpy()
    height_x = result["height_m"].to_numpy()
    mass_coefficient = float(np.dot(mass_x, y) / np.dot(mass_x, mass_x))
    mass_prediction = leave_one_out_zero_intercept(mass_x, y)
    implied_kz_m = float(np.sqrt(mass_coefficient / RTA_MASS_FRACTION))
    exact_kz_m = np.sqrt(y / result["rta_mass_kg"].to_numpy())

    depth_ratio = result["rta_depth_radius_m"] / result["height_m"]
    shoulder_ratio = result["rta_shoulder_radius_m"] / result["height_m"]
    median_geometry_coefficient = float(
        RTA_MASS_FRACTION
        * (np.median(depth_ratio) ** 2 + np.median(shoulder_ratio) ** 2)
        / 4.0
    )
    median_geometry_prediction = median_geometry_coefficient * x

    print(f"sessions={len(result)}")
    print(f"Izz kg*m^2: mean={y.mean():.6f}, SD={y.std(ddof=1):.6f}, range={y.min():.6f}..{y.max():.6f}")
    print(
        "dimensions m (mean): "
        f"L={result['rta_length_m'].mean():.4f}, "
        f"d={result['rta_depth_radius_m'].mean():.4f}, "
        f"r={result['rta_shoulder_radius_m'].mean():.4f}"
    )
    print(f"median d/height={np.median(depth_ratio):.6f}")
    print(f"median r/height={np.median(shoulder_ratio):.6f}")
    print(f"mean r^2 share of Izz geometry={np.mean(result['rta_shoulder_radius_m']**2 / (result['rta_depth_radius_m']**2 + result['rta_shoulder_radius_m']**2)):.4f}")
    print(f"kz m: mean={exact_kz_m.mean():.6f}, SD={exact_kz_m.std(ddof=1):.6f}, range={exact_kz_m.min():.6f}..{exact_kz_m.max():.6f}")
    print(f"max static marker spread m={result['max_static_marker_spread_m'].max():.6f}")
    print(
        "static-frame reconstruction: "
        f"median max deviation={result['rta_izz_max_frame_deviation_pct'].median():.4f}%, "
        f"maximum max deviation={result['rta_izz_max_frame_deviation_pct'].max():.4f}%, "
        f"maximum mean-coordinate difference={result['rta_izz_mean_coordinate_difference_pct'].abs().max():.6f}%"
    )
    print(f"corr(Izz, body mass)={np.corrcoef(y, result['body_mass_kg'])[0, 1]:.6f}")
    print(f"corr(Izz, mass*height^2)={np.corrcoef(y, x)[0, 1]:.6f}")
    print(f"corr(L, height)={np.corrcoef(result['rta_length_m'], result['height_m'])[0, 1]:.6f}")
    print(f"corr(d, height)={np.corrcoef(result['rta_depth_radius_m'], result['height_m'])[0, 1]:.6f}")
    print(f"corr(r, height)={np.corrcoef(result['rta_shoulder_radius_m'], result['height_m'])[0, 1]:.6f}")
    print(f"best zero-intercept coefficient for body mass={mass_coefficient:.9f} m^2")
    print(f"implied constant kz={implied_kz_m:.6f} m")
    print(f"mass-only LOOCV metrics={prediction_metrics(y, mass_prediction)}")
    print(f"best zero-intercept coefficient for mass*height^2={coefficient:.9f}")
    print(f"height approximation LOOCV metrics={prediction_metrics(y, height_prediction)}")
    print(f"median-geometry coefficient for mass*height^2={median_geometry_coefficient:.9f}")
    print(f"median-geometry metrics={prediction_metrics(y, median_geometry_prediction)}")

    result["izz_pred_constant_kz_loocv"] = mass_prediction
    result["izz_ape_constant_kz_loocv_pct"] = np.abs((y - mass_prediction) / y) * 100.0
    result["izz_pred_mass_height_sq_loocv"] = height_prediction
    result["izz_ape_mass_height_sq_loocv_pct"] = np.abs((y - height_prediction) / y) * 100.0

    allometric_models = {
        "c*mass^a": (True, False, 0.0, 0.0),
        "c*mass*height^b": (False, True, 1.0, 0.0),
        "c*mass^a*height^b": (True, True, 0.0, 0.0),
    }
    output_names = {
        "c*mass^a": "mass_power",
        "c*mass*height^b": "mass_height_power",
        "c*mass^a*height^b": "mass_height_both_power",
    }
    for name, settings in allometric_models.items():
        fitted = fit_allometric(mass_x, height_x, y, *settings)
        loo_prediction = leave_one_out_allometric(mass_x, height_x, y, *settings)
        lower, upper = bootstrap_allometric_ci(mass_x, height_x, y, *settings)
        print(
            f"{name}: coefficient={fitted[0]:.9f}, mass_exp={fitted[1]:.6f}, "
            f"height_exp={fitted[2]:.6f}"
        )
        print(
            f"{name} bootstrap95: coefficient={lower[0]:.9f}..{upper[0]:.9f}, "
            f"mass_exp={lower[1]:.6f}..{upper[1]:.6f}, "
            f"height_exp={lower[2]:.6f}..{upper[2]:.6f}"
        )
        print(f"{name} LOOCV metrics={prediction_metrics(y, loo_prediction)}")
        output_name = output_names[name]
        result[f"izz_pred_{output_name}_loocv"] = loo_prediction
        result[f"izz_ape_{output_name}_loocv_pct"] = np.abs((y - loo_prediction) / y) * 100.0

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
