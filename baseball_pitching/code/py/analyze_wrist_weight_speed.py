#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare wrist velocity, body mass, and pitch speed.

The analysis uses the corrected wrist velocity definition from
``upper_limb_linear_velocity_analysis.py`` and keeps all model comparisons on
the same pitch-level rows and the same pitcher-grouped folds.
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from upper_limb_linear_velocity_analysis import calculate_linear_velocity


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LANDMARKS_ZIP = DATA_DIR / "full_sig" / "landmarks.zip"
METADATA_CSV = DATA_DIR / "metadata.csv"


def load_pitch_metrics():
    landmark_cols = [
        "session_pitch",
        "time",
        "BR_time",
        "wrist_jc_x",
        "wrist_jc_y",
        "wrist_jc_z",
    ]
    with zipfile.ZipFile(LANDMARKS_ZIP) as archive:
        with archive.open("landmarks.csv") as handle:
            landmarks = pd.read_csv(handle, usecols=landmark_cols)

    metadata_cols = [
        "user",
        "session",
        "session_pitch",
        "session_mass_kg",
        "pitch_speed_mph",
    ]
    metadata = pd.read_csv(METADATA_CSV, usecols=metadata_cols)

    rows = []
    for session_pitch, group in landmarks.groupby("session_pitch", sort=False):
        meta = metadata.loc[metadata["session_pitch"] == session_pitch]
        if meta.empty or pd.isna(meta["pitch_speed_mph"].iloc[0]):
            continue

        group = group.sort_values("time").reset_index(drop=True)
        times = group["time"].to_numpy(dtype=float)
        positions = group[["wrist_jc_x", "wrist_jc_y", "wrist_jc_z"]].to_numpy(dtype=float)
        velocity = calculate_linear_velocity(positions, times)
        br_index = np.argmin(np.abs(times - group["BR_time"].iloc[0]))

        rows.append(
            {
                "session_pitch": session_pitch,
                "max_wrist_velocity": float(np.max(velocity)),
                "br_wrist_velocity": float(velocity[br_index]),
            }
        )

    wrist = pd.DataFrame(rows)
    data = metadata.merge(wrist, on="session_pitch", how="inner", validate="one_to_one")
    data = data.dropna(
        subset=["user", "session_mass_kg", "pitch_speed_mph", "max_wrist_velocity", "br_wrist_velocity"]
    ).reset_index(drop=True)
    if data.empty:
        raise ValueError("沒有可用的手腕速度、體重與球速資料")
    return data


def fit_in_sample(data, features):
    x = data[features]
    y = data["pitch_speed_mph"].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    prediction = model.predict(x)
    r2 = r2_score(y, prediction)
    n = len(data)
    parameter_count = len(features) + 1
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - parameter_count)
    return {
        "features": "+".join(features),
        "n": n,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "rmse_mph": mean_squared_error(y, prediction) ** 0.5,
        "rss": float(np.sum((y - prediction) ** 2)),
        "coefficients": dict(zip(features, model.coef_)),
        "intercept": float(model.intercept_),
    }


def grouped_oof_metrics(data, feature_sets):
    y = data["pitch_speed_mph"].to_numpy(dtype=float)
    groups = data["user"].to_numpy()
    folds = GroupKFold(n_splits=min(5, data["user"].nunique()))
    predictions = {name: np.full(len(data), np.nan) for name in feature_sets}

    for train_index, test_index in folds.split(data, y, groups):
        for name, features in feature_sets.items():
            model = LinearRegression().fit(data.iloc[train_index][features], y[train_index])
            predictions[name][test_index] = model.predict(data.iloc[test_index][features])

    results = {}
    for name, prediction in predictions.items():
        results[name] = {
            "r2": r2_score(y, prediction),
            "rmse_mph": mean_squared_error(y, prediction) ** 0.5,
            "mae_mph": mean_absolute_error(y, prediction),
        }
    return results


def within_pitcher_correlations(data):
    centered = data.copy()
    for variable in ["max_wrist_velocity", "br_wrist_velocity", "pitch_speed_mph"]:
        centered[variable] = centered[variable] - centered.groupby("user")[variable].transform("mean")

    result = {}
    for variable in ["max_wrist_velocity", "br_wrist_velocity"]:
        correlation = pearsonr(centered[variable], centered["pitch_speed_mph"])
        result[variable] = {"r": correlation.statistic, "p": correlation.pvalue}
    return result


def closest_speed_pair(data, velocity_column, tolerance_mps=0.05, same_user=False):
    best = None
    groups = data.groupby("user") if same_user else [("all", data)]
    for user, group in groups:
        group = group.sort_values(velocity_column).reset_index(drop=True)
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                velocity_difference = abs(group.at[first, velocity_column] - group.at[second, velocity_column])
                if velocity_difference > tolerance_mps:
                    continue
                speed_difference = abs(group.at[first, "pitch_speed_mph"] - group.at[second, "pitch_speed_mph"])
                if best is None or speed_difference > best["speed_difference_mph"]:
                    best = {
                        "user": user,
                        "velocity_column": velocity_column,
                        "tolerance_mps": tolerance_mps,
                        "velocity_difference_mps": velocity_difference,
                        "speed_difference_mph": speed_difference,
                        "first": group.iloc[first][["session_pitch", velocity_column, "pitch_speed_mph"]].to_dict(),
                        "second": group.iloc[second][["session_pitch", velocity_column, "pitch_speed_mph"]].to_dict(),
                    }
    return best


def main():
    data = load_pitch_metrics()
    print(f"Rows: {len(data)}; users: {data['user'].nunique()}")

    feature_sets = {
        "wrist": ["max_wrist_velocity"],
        "mass": ["session_mass_kg"],
        "wrist_plus_mass": ["max_wrist_velocity", "session_mass_kg"],
        "br_wrist": ["br_wrist_velocity"],
        "br_wrist_plus_mass": ["br_wrist_velocity", "session_mass_kg"],
    }
    in_sample = {name: fit_in_sample(data, features) for name, features in feature_sets.items()}
    oof = grouped_oof_metrics(data, feature_sets)

    nested = in_sample["wrist"]["rss"] - in_sample["wrist_plus_mass"]["rss"]
    denominator_df = in_sample["wrist_plus_mass"]["n"] - 3
    f_value = nested / (in_sample["wrist_plus_mass"]["rss"] / denominator_df)
    partial_f = {"delta_r2": in_sample["wrist_plus_mass"]["r2"] - in_sample["wrist"]["r2"], "F": f_value, "p": stats.f.sf(f_value, 1, denominator_df)}

    print("In-sample:")
    for name, result in in_sample.items():
        print(name, {key: round(value, 6) for key, value in result.items() if isinstance(value, (int, float, np.floating))})
    print("Partial F for mass after max wrist:", partial_f)
    print("Pitcher-grouped 5-fold OOF:", oof)
    print("Within-pitcher correlations:", within_pitcher_correlations(data))
    print("Closest max-wrist pair across users:", closest_speed_pair(data, "max_wrist_velocity"))
    print("Closest max-wrist pair within same user:", closest_speed_pair(data, "max_wrist_velocity", same_user=True))
    print("Closest BR-wrist pair across users:", closest_speed_pair(data, "br_wrist_velocity"))


if __name__ == "__main__":
    main()
