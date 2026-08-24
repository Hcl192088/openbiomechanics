"""Estimate pitch speed from high-performance assessment data.

The validation split is grouped by athlete so repeated assessments from the
same athlete never appear in both training and validation folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "high_performance" / "data" / "hp_obp.csv"
OUTPUT_DIR = ROOT / "baseball_pitching" / "data" / "hp_pitch_speed_prediction"

TARGET = "pitch_speed_mph"
GROUP = "athlete_uid"
LEVEL = "playing_level"

# HP assessment fields only. Dates, hitting outcomes, pitching HSS, target
# group labels, and identifiers are deliberately excluded as predictors.
EXCLUDED = {
    "test_date",
    "playing_level",
    "bat_speed_mph_group",
    "pitch_speed_mph_group",
    "pitching_session_date",
    "pitch_speed_mph",
    "pitching_max_hss",
    "hitting_session_date",
    "bat_speed_mph",
    "hitting_max_hss",
    "athlete_uid",
}


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(y_true - y_pred)
    return {
        "mae_mph": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mph": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "median_ae_mph": float(np.median(absolute_error)),
        "p90_ae_mph": float(np.quantile(absolute_error, 0.90)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_mph": float(np.mean(y_pred - y_true)),
    }


def build_models(numeric_columns: list[str], include_level: bool) -> dict[str, Pipeline]:
    transformers: list[tuple] = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_columns,
        )
    ]
    if include_level:
        transformers.append(
            (
                "level",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [LEVEL],
            )
        )

    linear_preprocessor = ColumnTransformer(transformers)

    tree_transformers: list[tuple] = [
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            numeric_columns,
        )
    ]
    if include_level:
        tree_transformers.append(
            (
                "level",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                [LEVEL],
            )
        )
    tree_preprocessor = ColumnTransformer(tree_transformers)

    return {
        "median_baseline": Pipeline(
            [
                ("preprocess", linear_preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "ridge": Pipeline(
            [
                ("preprocess", linear_preprocessor),
                ("model", RidgeCV(alphas=np.logspace(-3, 4, 50))),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_features=0.7,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def feature_panels(all_numeric_columns: list[str]) -> dict[str, list[str]]:
    cmj = [column for column in all_numeric_columns if column.endswith("_cmj")]
    imtp = [column for column in all_numeric_columns if column.endswith("_imtp")]
    return {
        "vald_pdf_report_compatible": [
            "jump_height_(imp-mom)_[cm]_mean_cmj",
            "peak_power_[w]_mean_cmj",
            "peak_power_/_bm_[w/kg]_mean_cmj",
            "rsi-modified_[m/s]_mean_cmj",
            "concentric_peak_force_[n]_mean_cmj",
            "jump_height_(imp-mom)_[cm]_mean_sj",
            "peak_power_[w]_mean_sj",
            "peak_power_/_bm_[w/kg]_mean_sj",
            "peak_vertical_force_[n]_max_imtp",
            "net_peak_vertical_force_[n]_max_imtp",
            "best_rsi_(flight/contact_time)_mean_ht",
            "peak_takeoff_force_[n]_mean_pp",
            "peak_eccentric_force_[n]_mean_pp",
            "relative_strength",
            "body_weight_[lbs]",
        ],
        "cmj_plus_body_weight": cmj + ["body_weight_[lbs]"],
        "cmj_imtp_plus_body_weight": cmj
        + imtp
        + ["relative_strength", "body_weight_[lbs]"],
        "all_hp": all_numeric_columns,
    }


def evaluate(data: pd.DataFrame, minimum_speed: float, label: str) -> dict:
    analysis = data.loc[data[TARGET].gt(minimum_speed)].copy()
    numeric_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED and pd.api.types.is_numeric_dtype(data[column])
    ]
    group_cv = GroupKFold(n_splits=5)
    y = analysis[TARGET].to_numpy(dtype=float)
    groups = analysis[GROUP].to_numpy()

    panels = feature_panels(numeric_columns)
    result = {
        "label": label,
        "target_filter": f"{TARGET} > {minimum_speed}",
        "rows": int(len(analysis)),
        "athletes": int(analysis[GROUP].nunique()),
        "predictor_count": len(numeric_columns),
        "feature_panels": panels,
        "models": {},
    }
    prediction_frames = []

    for panel_name, panel_columns in panels.items():
      for include_level in (False, True):
        level_suffix = "hp_only" if not include_level else "plus_playing_level"
        feature_set = f"{panel_name}__{level_suffix}"
        x_columns = panel_columns + ([LEVEL] if include_level else [])
        x = analysis[x_columns]
        for model_name, model in build_models(panel_columns, include_level).items():
            predictions = cross_val_predict(
                model,
                x,
                y,
                groups=groups,
                cv=group_cv,
                n_jobs=1,
                method="predict",
            )
            key = f"{feature_set}__{model_name}"
            metrics = metric_summary(y, predictions)
            complete = x.notna().all(axis=1).to_numpy()
            complete_profile = {
                "n": int(complete.sum()),
                **metric_summary(y[complete], predictions[complete]),
            }
            by_level = {}
            for level, index in analysis.groupby(LEVEL, dropna=False).groups.items():
                positions = analysis.index.get_indexer(index)
                by_level[str(level)] = {
                    "n": int(len(positions)),
                    **metric_summary(y[positions], predictions[positions]),
                }
            result["models"][key] = {
                **metrics,
                "complete_profile": complete_profile,
                "by_playing_level": by_level,
            }
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "analysis": label,
                        GROUP: groups,
                        LEVEL: analysis[LEVEL].to_numpy(),
                        "actual_mph": y,
                        "predicted_mph": predictions,
                        "error_mph": predictions - y,
                        "absolute_error_mph": np.abs(predictions - y),
                        "model": key,
                    }
                )
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / f"oof_predictions_{label}.csv", index=False
    )
    return result


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    report = {
        "data_path": str(DATA_PATH),
        "validation": "5-fold GroupKFold by athlete_uid",
        "invalid_zero_speed_rows": int(data[TARGET].eq(0).sum()),
        "analyses": [
            evaluate(data, minimum_speed=0, label="positive_speed"),
            evaluate(data, minimum_speed=40, label="speed_over_40_sensitivity"),
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
