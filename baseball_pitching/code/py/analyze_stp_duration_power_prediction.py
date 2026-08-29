"""Decompose thorax-limited shoulder STP energy into duration and mean power.

The primary unit is the pitcher/session (n=100). Each session's pooled mean
power is total energy divided by total active duration, so mean pitch energy
equals mean pitch duration times pooled mean power exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().with_name("stp_duration_power_outputs")
RANDOM_STATE = 20260829


def summarize_pitch(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time")
    time = group["time"].to_numpy(float)
    thorax = group["thorax_stp"].to_numpy(float)
    upper_arm = group["upper_arm_stp"].to_numpy(float)
    official_stp = group["shoulder_energy_transfer_stp"].to_numpy(float)
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"Non-increasing time for {group.name}")

    thorax_limited = (
        (thorax < 0)
        & (upper_arm > 0)
        & (np.abs(thorax) < np.abs(upper_arm))
    )
    interval_mask = thorax_limited[:-1] & thorax_limited[1:]
    dt = np.diff(time)
    interval_energy = 0.5 * (official_stp[:-1] + official_stp[1:]) * dt
    duration = float(dt[interval_mask].sum())
    energy = float(interval_energy[interval_mask].sum())
    return pd.Series(
        {
            "duration_s": duration,
            "stp_energy_j": energy,
            "mean_power_w": energy / duration if duration > 0 else np.nan,
        }
    )


def cv_r2(frame: pd.DataFrame, predictors: list[str], outcome: str) -> tuple[float, float]:
    clean = frame[[*predictors, outcome]].dropna()
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), clean[predictors], clean[outcome], cv=cv, scoring="r2"
    )
    return float(scores.mean()), float(scores.std(ddof=1))


def model_table(athlete: pd.DataFrame, outcome: str, prefix: str) -> pd.DataFrame:
    specifications = {
        "mass": ["session_mass_kg"],
        "mass_duration": ["session_mass_kg", "duration_s"],
        "mass_power": ["session_mass_kg", "pooled_mean_power_w"],
        "mass_duration_power": [
            "session_mass_kg",
            "duration_s",
            "pooled_mean_power_w",
        ],
    }
    rows = []
    for name, predictors in specifications.items():
        clean = athlete[[outcome, *predictors]].dropna()
        fit = smf.ols(f"{outcome} ~ {' + '.join(predictors)}", data=clean).fit()
        cv_mean, cv_sd = cv_r2(clean, predictors, outcome)
        rows.append(
            {
                "window": prefix,
                "outcome": outcome,
                "model": name,
                "n": len(clean),
                "r2": fit.rsquared,
                "adjusted_r2": fit.rsquared_adj,
                "cv_r2_mean": cv_mean,
                "cv_r2_sd": cv_sd,
            }
        )
    return pd.DataFrame(rows)


def standardized_coefficients(athlete: pd.DataFrame, outcome: str) -> pd.DataFrame:
    columns = ["session_mass_kg", "duration_s", "pooled_mean_power_w", outcome]
    clean = athlete[columns].dropna().copy()
    z = (clean - clean.mean()) / clean.std(ddof=0)
    fit = smf.ols(
        f"{outcome} ~ session_mass_kg + duration_s + pooled_mean_power_w", data=z
    ).fit()
    return pd.DataFrame(
        {
            "term": ["session_mass_kg", "duration_s", "pooled_mean_power_w"],
            "standardized_beta": fit.params[
                ["session_mass_kg", "duration_s", "pooled_mean_power_w"]
            ].to_numpy(),
            "p": fit.pvalues[
                ["session_mass_kg", "duration_s", "pooled_mean_power_w"]
            ].to_numpy(),
        }
    )


def within_between_models(per_pitch: pd.DataFrame) -> pd.DataFrame:
    data = per_pitch.dropna(
        subset=["stp_energy_j", "duration_s", "mean_power_w", "session_mass_kg"]
    ).copy()
    for variable in ["duration_s", "mean_power_w"]:
        data[f"{variable}_between"] = data.groupby("session")[variable].transform("mean")
        data[f"{variable}_within"] = data[variable] - data[f"{variable}_between"]
    fit = smf.mixedlm(
        "stp_energy_j ~ session_mass_kg + duration_s_between + duration_s_within + "
        "mean_power_w_between + mean_power_w_within",
        data=data,
        groups=data["session"],
    ).fit(reml=False, method="lbfgs")
    terms = [
        "session_mass_kg",
        "duration_s_between",
        "duration_s_within",
        "mean_power_w_between",
        "mean_power_w_within",
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
            "thorax_dist_seg_pwr",
            "upper_arm_prox_seg_pwr",
            "shoulder_energy_transfer_jfp",
            "shoulder_energy_transfer_stp",
        ],
    ).dropna()
    energy["thorax_stp"] = (
        energy["thorax_dist_seg_pwr"] + energy["shoulder_energy_transfer_jfp"]
    )
    energy["upper_arm_stp"] = (
        energy["upper_arm_prox_seg_pwr"] - energy["shoulder_energy_transfer_jfp"]
    )
    window = energy[
        (energy["time"] >= energy["fp_poi_time"])
        & (energy["time"] <= energy["BR_time"])
    ].copy()
    per_pitch = (
        window.groupby("session_pitch", sort=False)
        .apply(summarize_pitch, include_groups=False)
        .reset_index()
    )
    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    per_pitch = per_pitch.merge(metadata, on="session_pitch", validate="one_to_one")
    if len(per_pitch) != 411 or per_pitch["session"].nunique() != 100:
        raise ValueError(
            f"Expected 411 pitches/100 sessions, got {len(per_pitch)}/"
            f"{per_pitch['session'].nunique()}"
        )

    athlete = per_pitch.groupby("session", as_index=False).agg(
        session_mass_kg=("session_mass_kg", "first"),
        pitch_n=("session_pitch", "size"),
        duration_s=("duration_s", "mean"),
        stp_energy_j=("stp_energy_j", "mean"),
        duration_sum_s=("duration_s", "sum"),
        energy_sum_j=("stp_energy_j", "sum"),
    )
    athlete["pooled_mean_power_w"] = (
        athlete["energy_sum_j"] / athlete["duration_sum_s"]
    )
    identity_error = athlete["stp_energy_j"] - (
        athlete["duration_s"] * athlete["pooled_mean_power_w"]
    )
    if identity_error.abs().max() > 1e-10:
        raise ValueError(f"Energy identity failed: max error={identity_error.abs().max()}")

    positive = athlete[
        (athlete["duration_s"] > 0)
        & (athlete["pooled_mean_power_w"] > 0)
        & (athlete["stp_energy_j"] > 0)
    ].copy()
    positive["log_energy_j"] = np.log(positive["stp_energy_j"])
    positive["log_duration_s"] = np.log(positive["duration_s"])
    positive["log_power_w"] = np.log(positive["pooled_mean_power_w"])

    tables = [model_table(athlete, "stp_energy_j", "fp_br_thorax_limited")]
    log_fit = smf.ols(
        "log_energy_j ~ log_duration_s + log_power_w", data=positive
    ).fit()
    coefficients = standardized_coefficients(athlete, "stp_energy_j")
    mixed = within_between_models(per_pitch)

    OUTPUT_DIR.mkdir(exist_ok=True)
    per_pitch.to_csv(OUTPUT_DIR / "per_pitch.csv", index=False)
    athlete.to_csv(OUTPUT_DIR / "per_pitcher.csv", index=False)
    models = pd.concat(tables, ignore_index=True)
    models.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "standardized_coefficients.csv", index=False)
    mixed.to_csv(OUTPUT_DIR / "within_between_mixed_model.csv", index=False)

    print(f"pitches={len(per_pitch)}; sessions={len(athlete)}")
    print(
        f"positive-duration sessions={(athlete['duration_s'] > 0).sum()}; "
        f"max identity error={identity_error.abs().max():.3e} J"
    )
    print(
        f"means: energy={athlete['stp_energy_j'].mean():.3f} J, "
        f"duration={athlete['duration_s'].mean():.5f} s, "
        f"pooled power={athlete['pooled_mean_power_w'].mean():.1f} W"
    )
    print("\nModel comparison")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nStandardized full-model coefficients")
    print(coefficients.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nExact log identity check")
    print(
        f"duration coefficient={log_fit.params['log_duration_s']:.9f}; "
        f"power coefficient={log_fit.params['log_power_w']:.9f}; "
        f"R2={log_fit.rsquared:.12f}"
    )
    print("\nPitch-level within/between mixed model")
    print(mixed.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
