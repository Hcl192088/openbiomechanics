"""Test whether hip-shoulder separation and torso speed predict joint-high FP-MER power/duration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().with_name("separation_torso_joint_power_duration_outputs")
RANDOM_STATE = 20260829
SEPARATIONS = [
    "rotation_hip_shoulder_separation_fp",
    "max_rotation_hip_shoulder_separation",
]
OUTCOMES = ["duration_z", "power_z", "joint_high_score"]


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def cv_r2(data: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[float, float]:
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), data[predictors], data[outcome], cv=cv, scoring="r2"
    )
    return float(scores.mean()), float(scores.std(ddof=1))


def fit_models(athlete: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    coefficient_rows = []
    for separation in SEPARATIONS:
        sep_z = f"{separation}_z"
        for outcome in OUTCOMES:
            specifications = {
                "mass": ["mass_z"],
                "additive": ["mass_z", sep_z, "torso_velo_z"],
                "interaction": [
                    "mass_z",
                    sep_z,
                    "torso_velo_z",
                    "sep_torso_interaction",
                ],
            }
            frame = athlete.copy()
            frame["sep_torso_interaction"] = frame[sep_z] * frame["torso_velo_z"]
            for model_name, predictors in specifications.items():
                clean = frame[[outcome, *predictors]].dropna()
                fit = smf.ols(f"{outcome} ~ {' + '.join(predictors)}", data=clean).fit()
                cv_mean, cv_sd = cv_r2(clean, outcome, predictors)
                model_rows.append(
                    {
                        "separation_definition": separation,
                        "outcome": outcome,
                        "model": model_name,
                        "n": len(clean),
                        "r2": fit.rsquared,
                        "adjusted_r2": fit.rsquared_adj,
                        "cv_r2_mean": cv_mean,
                        "cv_r2_sd": cv_sd,
                    }
                )
                if model_name == "interaction":
                    for term in predictors:
                        coefficient_rows.append(
                            {
                                "separation_definition": separation,
                                "outcome": outcome,
                                "term": term,
                                "beta": fit.params[term],
                                "p": fit.pvalues[term],
                            }
                        )
    return pd.DataFrame(model_rows), pd.DataFrame(coefficient_rows)


def group_comparison(athlete: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for separation in SEPARATIONS:
        for joint_group, group in athlete.groupby("above_both_means"):
            rows.append(
                {
                    "separation_definition": separation,
                    "above_both_means": bool(joint_group),
                    "n": len(group),
                    "separation_mean": group[separation].mean(),
                    "torso_velo_mean": group["max_torso_rotational_velo"].mean(),
                    "mass_mean": group["session_mass_kg"].mean(),
                    "duration_mean_s": group["fp_mer_duration_s"].mean(),
                    "power_mean_w": group["fp_mer_positive_mean_power_w"].mean(),
                }
            )
    return pd.DataFrame(rows)


def within_between_models(pitch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = pitch.copy()
    data["duration_z"] = zscore(data["fp_mer_duration_s"])
    data["power_z"] = zscore(data["fp_mer_positive_mean_power_w"])
    data["mass_z"] = zscore(data["session_mass_kg"])
    for separation in SEPARATIONS:
        for variable in [separation, "max_torso_rotational_velo"]:
            between = data.groupby("session")[variable].transform("mean")
            data[f"{variable}_between_z"] = zscore(between)
            data[f"{variable}_within_z"] = zscore(data[variable] - between)
        terms = [
            "mass_z",
            f"{separation}_between_z",
            f"{separation}_within_z",
            "max_torso_rotational_velo_between_z",
            "max_torso_rotational_velo_within_z",
        ]
        for outcome in ["duration_z", "power_z"]:
            fit = smf.mixedlm(
                f"{outcome} ~ {' + '.join(terms)}",
                data=data,
                groups=data["session"],
            ).fit(reml=False, method="lbfgs")
            for term in terms:
                rows.append(
                    {
                        "separation_definition": separation,
                        "outcome": outcome,
                        "term": term,
                        "beta": fit.params[term],
                        "p": fit.pvalues[term],
                        "n_pitches": len(data),
                        "n_sessions": data["session"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    transfer = pd.read_csv(
        Path(__file__).resolve().with_name("total_shoulder_transfer_outputs")
        / "per_pitch.csv"
    )
    poi = pd.read_csv(
        ROOT / "data" / "poi" / "poi_metrics.csv",
        usecols=["session_pitch", *SEPARATIONS, "max_torso_rotational_velo"],
    )
    pitch = transfer.merge(poi, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411 or pitch["session"].nunique() != 100:
        raise ValueError("Expected 411 pitches and 100 sessions")

    athlete = pitch.groupby("session", as_index=False).agg(
        session_mass_kg=("session_mass_kg", "first"),
        pitch_n=("session_pitch", "size"),
        fp_mer_duration_s=("fp_mer_duration_s", "mean"),
        fp_mer_positive_mean_power_w=("fp_mer_positive_mean_power_w", "mean"),
        rotation_hip_shoulder_separation_fp=(
            "rotation_hip_shoulder_separation_fp",
            "mean",
        ),
        max_rotation_hip_shoulder_separation=(
            "max_rotation_hip_shoulder_separation",
            "mean",
        ),
        max_torso_rotational_velo=("max_torso_rotational_velo", "mean"),
    )
    athlete["duration_z"] = zscore(athlete["fp_mer_duration_s"])
    athlete["power_z"] = zscore(athlete["fp_mer_positive_mean_power_w"])
    athlete["joint_high_score"] = athlete[["duration_z", "power_z"]].min(axis=1)
    athlete["mass_z"] = zscore(athlete["session_mass_kg"])
    athlete["torso_velo_z"] = zscore(athlete["max_torso_rotational_velo"])
    for separation in SEPARATIONS:
        athlete[f"{separation}_z"] = zscore(athlete[separation])
    athlete["above_both_means"] = (athlete["duration_z"] > 0) & (
        athlete["power_z"] > 0
    )

    models, coefficients = fit_models(athlete)
    groups = group_comparison(athlete)
    mixed = within_between_models(pitch)
    correlations = athlete[
        [
            *SEPARATIONS,
            "max_torso_rotational_velo",
            "fp_mer_duration_s",
            "fp_mer_positive_mean_power_w",
            "joint_high_score",
            "session_mass_kg",
        ]
    ].corr()

    OUTPUT_DIR.mkdir(exist_ok=True)
    athlete.to_csv(OUTPUT_DIR / "per_pitcher.csv", index=False)
    models.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "interaction_model_coefficients.csv", index=False)
    groups.to_csv(OUTPUT_DIR / "joint_high_group_comparison.csv", index=False)
    mixed.to_csv(OUTPUT_DIR / "within_between_mixed_models.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "correlations.csv")

    print(f"coverage: pitches={len(pitch)}, pitchers={len(athlete)}")
    print(f"above both means={int(athlete['above_both_means'].sum())}")
    print("\nCorrelations")
    print(correlations.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nModel comparison")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInteraction-model coefficients")
    print(coefficients.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nJoint-high group comparison")
    print(groups.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPitch-level within/between mixed models")
    print(mixed.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
