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


def static_marker_centres(model_path: Path) -> tuple[dict[str, np.ndarray], float]:
    c3d = ezc3d.c3d(str(model_path))
    units = c3d["parameters"]["POINT"]["UNITS"]["value"]
    if units != ["m"]:
        raise ValueError(f"{model_path}: expected POINT unit ['m'], got {units!r}")

    raw_labels = [label.strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    labels = [
        label.removeprefix(KNOWN_LABEL_PREFIX) if label.startswith(KNOWN_LABEL_PREFIX) else label
        for label in raw_labels
    ]
    missing = [marker for marker in REQUIRED_MARKERS if marker not in labels]
    if missing:
        raise ValueError(f"{model_path}: missing markers {missing}")

    points = np.asarray(c3d["data"]["points"], dtype=float)
    centres: dict[str, np.ndarray] = {}
    frame_spreads: list[float] = []
    for marker in REQUIRED_MARKERS:
        marker_data = points[:, labels.index(marker), :]
        valid = np.isfinite(marker_data[:3]).all(axis=0) & (marker_data[3] >= 0)
        if not valid.any():
            raise ValueError(f"{model_path}: marker {marker} has no valid frames")
        xyz = marker_data[:3, valid].T
        centres[marker] = xyz.mean(axis=0)
        frame_spreads.append(float(np.max(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1))))
    return centres, max(frame_spreads)


def reconstruct_session(row: pd.Series, model_path: Path) -> dict[str, float | str]:
    markers, max_marker_spread_m = static_marker_centres(model_path)
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

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
