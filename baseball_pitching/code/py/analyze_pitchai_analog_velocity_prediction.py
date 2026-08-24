"""Benchmark PitchAI-like metrics against OpenBiomechanics pitch speed.

Only metrics with documented OpenBiomechanics analogues are used. PitchAI
peak arm speed, ball path length, and PK-to-BR time are not substituted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
POI_PATH = ROOT / "baseball_pitching" / "data" / "poi" / "poi_metrics.csv"
METADATA_PATH = ROOT / "baseball_pitching" / "data" / "metadata.csv"
OUTPUT_DIR = ROOT / "baseball_pitching" / "data" / "pitchai_analog_prediction"

TARGET = "pitch_speed_mph"
GROUP = "user"

STRICT_FEATURES = [
    "max_shoulder_external_rotation",
    "stride_length",
    "elbow_varus_moment",
    "lead_knee_extension_angular_velo_max",
]

EXPANDED_FEATURES = [
    "max_rotation_hip_shoulder_separation",
    "max_shoulder_horizontal_abduction",
    *STRICT_FEATURES,
]

MODEL_FEATURES = {
    "strict_four": STRICT_FEATURES,
    "strict_without_elbow_moment": [
        "max_shoulder_external_rotation",
        "stride_length",
        "lead_knee_extension_angular_velo_max",
    ],
    "expanded_six": EXPANDED_FEATURES,
    "expanded_without_elbow_moment": [
        feature for feature in EXPANDED_FEATURES if feature != "elbow_varus_moment"
    ],
    **{f"single__{feature}": [feature] for feature in EXPANDED_FEATURES},
}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.abs(actual - predicted)
    return {
        "mae_mph": float(mean_absolute_error(actual, predicted)),
        "rmse_mph": float(mean_squared_error(actual, predicted) ** 0.5),
        "median_ae_mph": float(np.median(error)),
        "p90_ae_mph": float(np.quantile(error, 0.90)),
        "r2": float(r2_score(actual, predicted)),
        "bias_mph": float(np.mean(predicted - actual)),
    }


def model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    poi = pd.read_csv(POI_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    if metadata["session_pitch"].duplicated().any():
        raise ValueError("metadata session_pitch is not unique")

    data = poi.merge(
        metadata[["session_pitch", GROUP]],
        on="session_pitch",
        how="left",
        validate="one_to_one",
    )
    if data[GROUP].isna().any():
        raise ValueError("POI rows are missing user after metadata join")

    analysis_columns = [TARGET, GROUP, "session_pitch", *EXPANDED_FEATURES]
    analysis = data[analysis_columns].dropna().copy()
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(analysis, analysis[TARGET], groups=analysis[GROUP]))
    y = analysis[TARGET].to_numpy(float)

    baseline = cross_val_predict(
        DummyRegressor(strategy="median"),
        np.zeros((len(analysis), 1)),
        y,
        groups=analysis[GROUP],
        cv=splits,
    )
    results = {"median_baseline": {"features": [], **metrics(y, baseline)}}
    predictions = {"median_baseline": baseline}

    for name, features in MODEL_FEATURES.items():
        predicted = cross_val_predict(
            model(),
            analysis[features],
            y,
            groups=analysis[GROUP],
            cv=splits,
            n_jobs=1,
        )
        predictions[name] = predicted
        results[name] = {"features": features, **metrics(y, predicted)}

    report = {
        "poi_path": str(POI_PATH),
        "metadata_path": str(METADATA_PATH),
        "validation": "5-fold GroupKFold by metadata.user",
        "rows": int(len(analysis)),
        "athletes": int(analysis[GROUP].nunique()),
        "pitch_speed_summary_mph": analysis[TARGET].describe().to_dict(),
        "feature_summary": analysis[EXPANDED_FEATURES].describe().to_dict(),
        "strict_mapping": STRICT_FEATURES,
        "approximate_mapping_sensitivity": [
            "max_rotation_hip_shoulder_separation",
            "max_shoulder_horizontal_abduction",
        ],
        "not_mapped": ["peak_arm_speed", "ball_path_length", "pk_to_br_time"],
        "models": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output = analysis[["session_pitch", GROUP, TARGET]].copy()
    for name, predicted in predictions.items():
        output[f"predicted__{name}"] = predicted
    output.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
