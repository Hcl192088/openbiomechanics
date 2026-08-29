"""Analyze positive shoulder transfer power over FP-MER and FP-BR.

Official FP-BR shoulder transfer is reconstructed as the integral of
max(STP + JFP, 0). FP-MER is the primary window for studying how pitchers
combine high positive transfer power with a long energy-transfer interval.
"""

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


def integrate_window(group: pd.DataFrame, start: float, end: float) -> dict[str, float]:
    window = group[(group["time"] >= start) & (group["time"] <= end)].sort_values("time")
    time = window["time"].to_numpy(float)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError(f"Invalid time series for {group.name}: {start} to {end}")
    total_power = (
        window["shoulder_energy_transfer_stp"]
        + window["shoulder_energy_transfer_jfp"]
    ).to_numpy(float)
    positive_power = np.maximum(total_power, 0.0)
    duration = float(time[-1] - time[0])
    positive_energy = float(np.trapezoid(positive_power, time))
    net_energy = float(np.trapezoid(total_power, time))
    return {
        "duration_s": duration,
        "positive_energy_j": positive_energy,
        "net_energy_j": net_energy,
        "positive_mean_power_w": positive_energy / duration,
        "negative_energy_j": positive_energy - net_energy,
    }


def summarize_pitch(group: pd.DataFrame) -> pd.Series:
    fp = group["fp_poi_time"].dropna().unique()
    mer = group["MER_time"].dropna().unique()
    br = group["BR_time"].dropna().unique()
    if len(fp) != 1 or len(mer) != 1 or len(br) != 1:
        raise ValueError(f"Ambiguous event time for {group.name}")
    fp_mer = integrate_window(group, float(fp[0]), float(mer[0]))
    fp_br = integrate_window(group, float(fp[0]), float(br[0]))
    result = {f"fp_mer_{key}": value for key, value in fp_mer.items()}
    result.update({f"fp_br_{key}": value for key, value in fp_br.items()})
    result["mer_share_of_fp_br"] = (
        fp_mer["positive_energy_j"] / fp_br["positive_energy_j"]
    )
    return pd.Series(result)


def cv_r2(data: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[float, float]:
    clean = data[[outcome, *predictors]].dropna()
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), clean[predictors], clean[outcome], cv=cv, scoring="r2"
    )
    return float(scores.mean()), float(scores.std(ddof=1))


def model_table(
    athlete: pd.DataFrame,
    label: str,
    outcome: str,
    duration: str,
    power: str,
) -> pd.DataFrame:
    specifications = {
        "mass": ["session_mass_kg"],
        "mass_duration": ["session_mass_kg", duration],
        "mass_power": ["session_mass_kg", power],
        "mass_duration_power": ["session_mass_kg", duration, power],
    }
    rows = []
    for name, predictors in specifications.items():
        clean = athlete[[outcome, *predictors]].dropna()
        fit = smf.ols(f"{outcome} ~ {' + '.join(predictors)}", data=clean).fit()
        cv_mean, cv_sd = cv_r2(clean, outcome, predictors)
        rows.append(
            {
                "analysis": label,
                "model": name,
                "n": len(clean),
                "r2": fit.rsquared,
                "adjusted_r2": fit.rsquared_adj,
                "cv_r2_mean": cv_mean,
                "cv_r2_sd": cv_sd,
            }
        )
    return pd.DataFrame(rows)


def standardized_model(
    athlete: pd.DataFrame,
    label: str,
    outcome: str,
    duration: str,
    power: str,
) -> pd.DataFrame:
    columns = [outcome, "session_mass_kg", duration, power]
    clean = athlete[columns].dropna()
    z = (clean - clean.mean()) / clean.std(ddof=0)
    fit = smf.ols(
        f"{outcome} ~ session_mass_kg + {duration} + {power}", data=z
    ).fit()
    terms = ["session_mass_kg", duration, power]
    return pd.DataFrame(
        {
            "analysis": label,
            "term": terms,
            "standardized_beta": fit.params[terms].to_numpy(),
            "p": fit.pvalues[terms].to_numpy(),
        }
    )


def mixed_model(
    per_pitch: pd.DataFrame,
    label: str,
    outcome: str,
    duration: str,
    power: str,
) -> pd.DataFrame:
    data = per_pitch.dropna(subset=[outcome, "session_mass_kg", duration, power]).copy()
    for variable in [duration, power]:
        data[f"{variable}_between"] = data.groupby("session")[variable].transform("mean")
        data[f"{variable}_within"] = data[variable] - data[f"{variable}_between"]
    terms = [
        "session_mass_kg",
        f"{duration}_between",
        f"{duration}_within",
        f"{power}_between",
        f"{power}_within",
    ]
    fit = smf.mixedlm(
        f"{outcome} ~ {' + '.join(terms)}", data=data, groups=data["session"]
    ).fit(reml=False, method="lbfgs")
    return pd.DataFrame(
        {
            "analysis": label,
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
            "MER_time",
            "BR_time",
            "shoulder_energy_transfer_stp",
            "shoulder_energy_transfer_jfp",
        ],
    ).dropna()
    per_pitch = (
        energy.groupby("session_pitch", sort=False)
        .apply(summarize_pitch, include_groups=False)
        .reset_index()
    )
    poi = pd.read_csv(
        ROOT / "data" / "poi" / "poi_metrics.csv",
        usecols=["session_pitch", "shoulder_transfer_fp_br"],
    ).rename(columns={"shoulder_transfer_fp_br": "official_fp_br_j"})
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
        official_fp_br_j=("official_fp_br_j", "mean"),
        fp_mer_duration_s=("fp_mer_duration_s", "mean"),
        fp_mer_positive_energy_j=("fp_mer_positive_energy_j", "mean"),
        fp_mer_duration_sum_s=("fp_mer_duration_s", "sum"),
        fp_mer_energy_sum_j=("fp_mer_positive_energy_j", "sum"),
        fp_br_duration_s=("fp_br_duration_s", "mean"),
        fp_br_positive_energy_j=("fp_br_positive_energy_j", "mean"),
        fp_br_net_energy_j=("fp_br_net_energy_j", "mean"),
        fp_br_negative_energy_j=("fp_br_negative_energy_j", "mean"),
        fp_br_duration_sum_s=("fp_br_duration_s", "sum"),
        fp_br_energy_sum_j=("fp_br_positive_energy_j", "sum"),
        mer_share_of_fp_br=("mer_share_of_fp_br", "mean"),
    )
    for phase in ["fp_mer", "fp_br"]:
        athlete[f"{phase}_positive_mean_power_w"] = (
            athlete[f"{phase}_energy_sum_j"] / athlete[f"{phase}_duration_sum_s"]
        )
    athlete["fp_mer_above_both_means"] = (
        (athlete["fp_mer_duration_s"] > athlete["fp_mer_duration_s"].mean())
        & (
            athlete["fp_mer_positive_mean_power_w"]
            > athlete["fp_mer_positive_mean_power_w"].mean()
        )
    )
    athlete["fp_mer_top_quartile_both"] = (
        (
            athlete["fp_mer_duration_s"]
            >= athlete["fp_mer_duration_s"].quantile(0.75)
        )
        & (
            athlete["fp_mer_positive_mean_power_w"]
            >= athlete["fp_mer_positive_mean_power_w"].quantile(0.75)
        )
    )

    definitions = [
        (
            "fp_mer_positive_transfer",
            "fp_mer_positive_energy_j",
            "fp_mer_duration_s",
            "fp_mer_positive_mean_power_w",
        ),
        (
            "fp_br_reconstructed_positive",
            "fp_br_positive_energy_j",
            "fp_br_duration_s",
            "fp_br_positive_mean_power_w",
        ),
        (
            "fp_br_official",
            "official_fp_br_j",
            "fp_br_duration_s",
            "fp_br_positive_mean_power_w",
        ),
    ]
    models = pd.concat(
        [model_table(athlete, *definition) for definition in definitions],
        ignore_index=True,
    )
    coefficients = pd.concat(
        [standardized_model(athlete, *definition) for definition in definitions],
        ignore_index=True,
    )
    mixed = pd.concat(
        [mixed_model(per_pitch, *definition) for definition in definitions],
        ignore_index=True,
    )

    validation = pd.DataFrame(
        {
            "level": ["pitch", "pitcher"],
            "n": [len(per_pitch), len(athlete)],
            "official_positive_r": [
                per_pitch["official_fp_br_j"].corr(per_pitch["fp_br_positive_energy_j"]),
                athlete["official_fp_br_j"].corr(athlete["fp_br_positive_energy_j"]),
            ],
            "official_minus_positive_mean_j": [
                (per_pitch["official_fp_br_j"] - per_pitch["fp_br_positive_energy_j"]).mean(),
                (athlete["official_fp_br_j"] - athlete["fp_br_positive_energy_j"]).mean(),
            ],
            "mae_j": [
                (per_pitch["official_fp_br_j"] - per_pitch["fp_br_positive_energy_j"]).abs().mean(),
                (athlete["official_fp_br_j"] - athlete["fp_br_positive_energy_j"]).abs().mean(),
            ],
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    per_pitch.to_csv(OUTPUT_DIR / "per_pitch.csv", index=False)
    athlete.to_csv(OUTPUT_DIR / "per_pitcher.csv", index=False)
    models.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "standardized_coefficients.csv", index=False)
    mixed.to_csv(OUTPUT_DIR / "within_between_mixed_models.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "official_reconstruction_validation.csv", index=False)

    print(f"coverage: pitches={len(per_pitch)}, pitchers={len(athlete)}")
    print("\nOfficial FP-BR validation")
    print(validation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPhase summary (pitcher means)")
    summary_columns = [
        "fp_mer_duration_s",
        "fp_mer_positive_energy_j",
        "fp_mer_positive_mean_power_w",
        "fp_br_duration_s",
        "fp_br_positive_energy_j",
        "fp_br_negative_energy_j",
        "mer_share_of_fp_br",
    ]
    print(athlete[summary_columns].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nCorrelations")
    correlation_columns = [
        "session_mass_kg",
        "fp_mer_duration_s",
        "fp_mer_positive_mean_power_w",
        "fp_mer_positive_energy_j",
    ]
    print(athlete[correlation_columns].corr().to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        "\nJoint-high pitchers: "
        f"above both means={int(athlete['fp_mer_above_both_means'].sum())}; "
        f"top quartile on both={int(athlete['fp_mer_top_quartile_both'].sum())}"
    )
    print("\nModel comparison")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nStandardized models")
    print(coefficients.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
