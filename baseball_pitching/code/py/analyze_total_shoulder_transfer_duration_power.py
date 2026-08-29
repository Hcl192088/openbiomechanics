"""Analyze FP-to-BR shoulder STP, JFP, and total transfer duration/power."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().with_name("total_shoulder_transfer_outputs")
RANDOM_STATE = 20260829


def summarize_pitch(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time")
    time = group["time"].to_numpy(float)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError(f"Invalid FP-BR time series for {group.name}")
    stp_j = float(np.trapezoid(group["shoulder_energy_transfer_stp"], time))
    jfp_j = float(np.trapezoid(group["shoulder_energy_transfer_jfp"], time))
    duration_s = float(time[-1] - time[0])
    return pd.Series(
        {
            "fp_br_duration_s": duration_s,
            "stp_j": stp_j,
            "jfp_j": jfp_j,
            "reconstructed_total_j": stp_j + jfp_j,
            "stp_mean_power_w": stp_j / duration_s,
            "jfp_mean_power_w": jfp_j / duration_s,
            "total_mean_power_w": (stp_j + jfp_j) / duration_s,
        }
    )


def repeated_cv_r2(data: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[float, float]:
    clean = data[[outcome, *predictors]].dropna()
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), clean[predictors], clean[outcome], cv=cv, scoring="r2"
    )
    return float(scores.mean()), float(scores.std(ddof=1))


def compare_models(
    athlete: pd.DataFrame, outcome: str, power: str, outcome_type: str
) -> pd.DataFrame:
    specifications = {
        "mass": ["session_mass_kg"],
        "mass_duration": ["session_mass_kg", "fp_br_duration_s"],
        "mass_power": ["session_mass_kg", power],
        "mass_duration_power": ["session_mass_kg", "fp_br_duration_s", power],
    }
    rows = []
    for name, predictors in specifications.items():
        clean = athlete[[outcome, *predictors]].dropna()
        fit = smf.ols(f"{outcome} ~ {' + '.join(predictors)}", data=clean).fit()
        cv_mean, cv_sd = repeated_cv_r2(clean, outcome, predictors)
        rows.append(
            {
                "outcome_type": outcome_type,
                "outcome": outcome,
                "power_predictor": power,
                "model": name,
                "n": len(clean),
                "r2": fit.rsquared,
                "adjusted_r2": fit.rsquared_adj,
                "cv_r2_mean": cv_mean,
                "cv_r2_sd": cv_sd,
            }
        )
    return pd.DataFrame(rows)


def standardized_full_model(
    athlete: pd.DataFrame, outcome: str, power: str, outcome_type: str
) -> pd.DataFrame:
    columns = [outcome, "session_mass_kg", "fp_br_duration_s", power]
    clean = athlete[columns].dropna()
    z = (clean - clean.mean()) / clean.std(ddof=0)
    fit = smf.ols(
        f"{outcome} ~ session_mass_kg + fp_br_duration_s + {power}", data=z
    ).fit()
    terms = ["session_mass_kg", "fp_br_duration_s", power]
    return pd.DataFrame(
        {
            "outcome_type": outcome_type,
            "term": terms,
            "standardized_beta": fit.params[terms].to_numpy(),
            "p": fit.pvalues[terms].to_numpy(),
        }
    )


def mixed_official_total(per_pitch: pd.DataFrame) -> pd.DataFrame:
    data = per_pitch.dropna(
        subset=[
            "official_total_j",
            "session_mass_kg",
            "fp_br_duration_s",
            "total_mean_power_w",
        ]
    ).copy()
    for variable in ["fp_br_duration_s", "total_mean_power_w"]:
        data[f"{variable}_between"] = data.groupby("session")[variable].transform("mean")
        data[f"{variable}_within"] = data[variable] - data[f"{variable}_between"]
    formula = (
        "official_total_j ~ session_mass_kg + fp_br_duration_s_between + "
        "fp_br_duration_s_within + total_mean_power_w_between + "
        "total_mean_power_w_within"
    )
    fit = smf.mixedlm(formula, data=data, groups=data["session"]).fit(
        reml=False, method="lbfgs"
    )
    terms = [
        "session_mass_kg",
        "fp_br_duration_s_between",
        "fp_br_duration_s_within",
        "total_mean_power_w_between",
        "total_mean_power_w_within",
    ]
    return pd.DataFrame(
        {
            "term": terms,
            "coefficient": fit.params[terms].to_numpy(),
            "standard_error": fit.bse[terms].to_numpy(),
            "p": fit.pvalues[terms].to_numpy(),
            "n_pitches": len(data),
            "n_sessions": data["session"].nunique(),
        }
    )


def main() -> None:
    energy = pd.read_csv(
        ROOT / "data" / "full_sig" / "energy_flow.csv",
        usecols=[
            "session_pitch",
            "time",
            "fp_poi_time",
            "BR_time",
            "shoulder_energy_transfer_stp",
            "shoulder_energy_transfer_jfp",
        ],
    ).dropna()
    window = energy[
        (energy["time"] >= energy["fp_poi_time"])
        & (energy["time"] <= energy["BR_time"])
    ].copy()
    per_pitch = (
        window.groupby("session_pitch", sort=False)
        .apply(summarize_pitch, include_groups=False)
        .reset_index()
    )
    poi = pd.read_csv(
        ROOT / "data" / "poi" / "poi_metrics.csv",
        usecols=["session_pitch", "shoulder_transfer_fp_br"],
    ).rename(columns={"shoulder_transfer_fp_br": "official_total_j"})
    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    per_pitch = (
        per_pitch.merge(poi, on="session_pitch", validate="one_to_one")
        .merge(metadata, on="session_pitch", validate="one_to_one")
    )
    if len(per_pitch) != 411 or per_pitch["session"].nunique() != 100:
        raise ValueError("Expected complete coverage of 411 pitches and 100 sessions")

    athlete = per_pitch.groupby("session", as_index=False).agg(
        session_mass_kg=("session_mass_kg", "first"),
        pitch_n=("session_pitch", "size"),
        fp_br_duration_s=("fp_br_duration_s", "mean"),
        stp_j=("stp_j", "mean"),
        jfp_j=("jfp_j", "mean"),
        reconstructed_total_j=("reconstructed_total_j", "mean"),
        official_total_j=("official_total_j", "mean"),
        duration_sum_s=("fp_br_duration_s", "sum"),
        stp_sum_j=("stp_j", "sum"),
        jfp_sum_j=("jfp_j", "sum"),
        reconstructed_total_sum_j=("reconstructed_total_j", "sum"),
    )
    athlete["stp_mean_power_w"] = athlete["stp_sum_j"] / athlete["duration_sum_s"]
    athlete["jfp_mean_power_w"] = athlete["jfp_sum_j"] / athlete["duration_sum_s"]
    athlete["total_mean_power_w"] = (
        athlete["reconstructed_total_sum_j"] / athlete["duration_sum_s"]
    )

    identities = {
        "stp": athlete["stp_j"] - athlete["fp_br_duration_s"] * athlete["stp_mean_power_w"],
        "jfp": athlete["jfp_j"] - athlete["fp_br_duration_s"] * athlete["jfp_mean_power_w"],
        "total": athlete["reconstructed_total_j"]
        - athlete["fp_br_duration_s"] * athlete["total_mean_power_w"],
    }
    if max(error.abs().max() for error in identities.values()) > 1e-10:
        raise ValueError("Energy-duration-power identity failed")

    definitions = [
        ("stp_j", "stp_mean_power_w", "full_fp_br_stp"),
        ("jfp_j", "jfp_mean_power_w", "full_fp_br_jfp"),
        ("reconstructed_total_j", "total_mean_power_w", "reconstructed_total"),
        ("official_total_j", "total_mean_power_w", "official_total"),
    ]
    models = pd.concat(
        [compare_models(athlete, outcome, power, label) for outcome, power, label in definitions],
        ignore_index=True,
    )
    coefficients = pd.concat(
        [
            standardized_full_model(athlete, outcome, power, label)
            for outcome, power, label in definitions
        ],
        ignore_index=True,
    )
    mixed = mixed_official_total(per_pitch)

    validation = pd.DataFrame(
        {
            "level": ["pitch", "pitcher"],
            "n": [len(per_pitch), len(athlete)],
            "official_reconstructed_r": [
                per_pitch["official_total_j"].corr(per_pitch["reconstructed_total_j"]),
                athlete["official_total_j"].corr(athlete["reconstructed_total_j"]),
            ],
            "official_mean_j": [per_pitch["official_total_j"].mean(), athlete["official_total_j"].mean()],
            "reconstructed_mean_j": [
                per_pitch["reconstructed_total_j"].mean(),
                athlete["reconstructed_total_j"].mean(),
            ],
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    per_pitch.to_csv(OUTPUT_DIR / "per_pitch.csv", index=False)
    athlete.to_csv(OUTPUT_DIR / "per_pitcher.csv", index=False)
    models.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "standardized_coefficients.csv", index=False)
    mixed.to_csv(OUTPUT_DIR / "official_total_within_between_mixed_model.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "official_reconstruction_validation.csv", index=False)

    print(f"coverage: pitches={len(per_pitch)}, pitchers={len(athlete)}")
    print("\nOfficial versus reconstructed total")
    print(validation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPitcher means")
    print(
        athlete[["fp_br_duration_s", "stp_j", "jfp_j", "reconstructed_total_j", "official_total_j"]]
        .mean()
        .to_string(float_format=lambda x: f"{x:.4f}")
    )
    print("\nModel comparison")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nStandardized full models")
    print(coefficients.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOfficial total pitch-level within/between model")
    print(mixed.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
