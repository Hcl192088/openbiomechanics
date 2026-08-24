"""Test incremental pitch-speed prediction from shoulder IR strength.

All comparisons use the same rows with both ShoulderIRL and ShoulderIRR
observed. Cross-validation is grouped by athlete, and paired MAE differences
receive an athlete-cluster bootstrap confidence interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from predict_pitch_speed_from_hp import build_models, metric_summary


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "high_performance" / "data" / "hp_obp.csv"
OUTPUT_DIR = ROOT / "baseball_pitching" / "data" / "shoulder_ir_increment"

TARGET = "pitch_speed_mph"
GROUP = "athlete_uid"
IR_LEFT = "ShoulderIRL"
IR_RIGHT = "ShoulderIRR"

BASE_FEATURES = [
    "peak_power_[w]_mean_cmj",
    "peak_power_[w]_mean_sj",
    "peak_vertical_force_[n]_max_imtp",
    "best_rsi_(flight/contact_time)_mean_ht",
    "peak_eccentric_force_[n]_mean_pp",
    "body_weight_[lbs]",
]


def add_ir_derivatives(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    left = result[IR_LEFT].astype(float)
    right = result[IR_RIGHT].astype(float)
    result["ShoulderIR_mean"] = (left + right) / 2
    result["ShoulderIR_min"] = np.minimum(left, right)
    result["ShoulderIR_max"] = np.maximum(left, right)
    result["ShoulderIR_left_minus_right"] = left - right
    result["ShoulderIR_abs_asymmetry_pct"] = (
        (left - right).abs() / np.maximum(left, right) * 100
    )
    return result


def grouped_predictions(
    data: pd.DataFrame, features: list[str], splits: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    model = build_models(features, include_level=False)["extra_trees"]
    return cross_val_predict(
        model,
        data[features],
        data[TARGET].to_numpy(float),
        groups=data[GROUP].to_numpy(),
        cv=splits,
        n_jobs=1,
    )


def cluster_bootstrap_delta_mae(
    data: pd.DataFrame,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    repetitions: int = 10_000,
) -> dict[str, float]:
    actual = data[TARGET].to_numpy(float)
    delta = np.abs(actual - candidate_predictions) - np.abs(actual - baseline_predictions)
    grouped = pd.DataFrame({GROUP: data[GROUP].to_numpy(), "delta": delta}).groupby(GROUP)[
        "delta"
    ].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    boot = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {
        "delta_mae_mph": float(delta.mean()),
        "ci95_low_mph": float(np.quantile(boot, 0.025)),
        "ci95_high_mph": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_improvement": float(np.mean(boot < 0)),
        "bootstrap_repetitions": repetitions,
    }


def main() -> None:
    source = pd.read_csv(DATA_PATH)
    speed = pd.to_numeric(source[TARGET], errors="coerce")
    cohort = source.loc[
        speed.gt(40) & source[[IR_LEFT, IR_RIGHT]].notna().all(axis=1)
    ].copy()
    cohort = add_ir_derivatives(cohort)

    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(cohort, cohort[TARGET], groups=cohort[GROUP]))
    y = cohort[TARGET].to_numpy(float)

    feature_sets = {
        "baseline_six_hp": BASE_FEATURES,
        "baseline_plus_irl": BASE_FEATURES + [IR_LEFT],
        "baseline_plus_irr": BASE_FEATURES + [IR_RIGHT],
        "baseline_plus_both_raw": BASE_FEATURES + [IR_LEFT, IR_RIGHT],
        "baseline_plus_ir_mean": BASE_FEATURES + ["ShoulderIR_mean"],
        "baseline_plus_ir_min": BASE_FEATURES + ["ShoulderIR_min"],
        "baseline_plus_ir_max": BASE_FEATURES + ["ShoulderIR_max"],
        "baseline_plus_ir_mean_and_asymmetry": BASE_FEATURES
        + ["ShoulderIR_mean", "ShoulderIR_abs_asymmetry_pct"],
        "baseline_plus_ir_mean_and_signed_difference": BASE_FEATURES
        + ["ShoulderIR_mean", "ShoulderIR_left_minus_right"],
        "shoulder_ir_both_only": [IR_LEFT, IR_RIGHT],
        "shoulder_ir_mean_only": ["ShoulderIR_mean"],
    }

    predictions = {}
    results = {}
    for name, features in feature_sets.items():
        predicted = grouped_predictions(cohort, features, splits)
        predictions[name] = predicted
        results[name] = {
            "features": features,
            **metric_summary(y, predicted),
        }

    baseline_predictions = predictions["baseline_six_hp"]
    for name in results:
        if name == "baseline_six_hp" or name.startswith("shoulder_ir_"):
            continue
        results[name]["paired_vs_baseline"] = cluster_bootstrap_delta_mae(
            cohort, baseline_predictions, predictions[name]
        )

    complete_baseline = cohort[BASE_FEATURES].notna().all(axis=1).to_numpy()
    complete_case_results = {}
    for name, predicted in predictions.items():
        complete_case_results[name] = {
            "n": int(complete_baseline.sum()),
            **metric_summary(y[complete_baseline], predicted[complete_baseline]),
        }

    output = {
        "data_path": str(DATA_PATH),
        "target_filter": f"{TARGET} > 40",
        "cohort_filter": "ShoulderIRL and ShoulderIRR both observed",
        "rows": int(len(cohort)),
        "athletes": int(cohort[GROUP].nunique()),
        "playing_level_counts": {
            str(key): int(value)
            for key, value in cohort["playing_level"].value_counts(dropna=False).items()
        },
        "ir_summary": cohort[[IR_LEFT, IR_RIGHT]].describe().to_dict(),
        "baseline_complete_rows": int(complete_baseline.sum()),
        "models": results,
        "complete_baseline_profile_models": complete_case_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_frame = cohort[[GROUP, "playing_level", TARGET, IR_LEFT, IR_RIGHT]].copy()
    for name, predicted in predictions.items():
        prediction_frame[f"predicted__{name}"] = predicted
    prediction_frame.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
