"""Conserving STP/JFP decomposition of positive shoulder transfer over FP-MER."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("stp_jfp_positive_transfer_outputs")
RANDOM_STATE = 20260829
KINEMATICS = [
    "rotation_hip_shoulder_separation_fp",
    "max_rotation_hip_shoulder_separation",
    "max_torso_rotational_velo",
]


def trapz_gated(values: np.ndarray, gate: np.ndarray, time: np.ndarray) -> float:
    return float(np.trapezoid(np.where(gate, values, 0.0), time))


def summarize(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time").copy()
    fp = group["fp_poi_time"].dropna().unique()
    mer = group["MER_time"].dropna().unique()
    if len(fp) != 1 or len(mer) != 1:
        raise ValueError(f"Ambiguous FP/MER for {group.name}")

    # Differentiate the full available trajectory before cutting the window.
    t_all = group["time"].to_numpy(float)
    if np.any(np.diff(t_all) <= 0):
        raise ValueError(f"Non-increasing time for {group.name}")
    for axis in "xyz":
        group[f"shoulder_jc_vel_{axis}"] = np.gradient(
            group[f"shoulder_jc_{axis}"].to_numpy(float), t_all
        )

    w = group[(group["time"] >= fp[0]) & (group["time"] <= mer[0])]
    t = w["time"].to_numpy(float)
    if len(t) < 2:
        raise ValueError(f"Insufficient FP-MER samples for {group.name}")
    stp = w["shoulder_energy_transfer_stp"].to_numpy(float)
    jfp = w["shoulder_energy_transfer_jfp"].to_numpy(float)
    total = stp + jfp
    gate = total > 0

    total_j = float(np.trapezoid(np.maximum(total, 0.0), t))
    stp_j = trapz_gated(stp, gate, t)
    jfp_j = trapz_gated(jfp, gate, t)

    force = w[[f"shoulder_thorax_force_{a}" for a in "xyz"]].to_numpy(float)
    velocity = w[[f"shoulder_jc_vel_{a}" for a in "xyz"]].to_numpy(float)
    reconstructed_jfp = np.einsum("ij,ij->i", force, velocity)
    force_mag = np.linalg.norm(force, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    denominator = force_mag * speed
    alignment = np.divide(
        reconstructed_jfp,
        denominator,
        out=np.full_like(reconstructed_jfp, np.nan),
        where=denominator > 0,
    )

    thorax_stp = (
        w["thorax_dist_seg_pwr"].to_numpy(float) + jfp
    )
    arm_stp = w["upper_arm_prox_seg_pwr"].to_numpy(float) - jfp
    valid_transfer = (thorax_stp * arm_stp < 0) & (stp > 0)
    thorax_limited = valid_transfer & (np.abs(thorax_stp) <= np.abs(arm_stp))
    arm_limited = valid_transfer & ~thorax_limited

    return pd.Series(
        {
            "fp_mer_duration_s": float(t[-1] - t[0]),
            "positive_total_j": total_j,
            "gated_stp_j": stp_j,
            "gated_jfp_j": jfp_j,
            "conservation_error_j": total_j - stp_j - jfp_j,
            "stp_positive_j": trapz_gated(np.maximum(stp, 0), gate, t),
            "stp_negative_j": trapz_gated(np.minimum(stp, 0), gate, t),
            "jfp_positive_j": trapz_gated(np.maximum(jfp, 0), gate, t),
            "jfp_negative_j": trapz_gated(np.minimum(jfp, 0), gate, t),
            "mean_force_n": float(np.nanmean(force_mag[gate])),
            "mean_shoulder_jc_speed_m_s": float(np.nanmean(speed[gate])),
            "mean_force_velocity_alignment": float(np.nanmean(alignment[gate])),
            "official_jfp_vs_force_velocity_r": float(
                np.corrcoef(jfp[gate], reconstructed_jfp[gate])[0, 1]
            ),
            "official_jfp_force_velocity_mae_w": float(
                np.mean(np.abs(jfp[gate] - reconstructed_jfp[gate]))
            ),
            "thorax_limited_fraction": float(thorax_limited.sum() / valid_transfer.sum()),
            "arm_limited_fraction": float(arm_limited.sum() / valid_transfer.sum()),
        }
    )


def cv_r2(data: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[float, float]:
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=RANDOM_STATE)
    scores = cross_val_score(
        LinearRegression(), data[predictors], data[outcome], cv=cv, scoring="r2"
    )
    return float(scores.mean()), float(scores.std(ddof=1))


def model_panels(athlete: pd.DataFrame) -> pd.DataFrame:
    panels = {
        "mass": ["session_mass_kg"],
        "mass_plus_fp_separation_torso": [
            "session_mass_kg", "rotation_hip_shoulder_separation_fp",
            "max_torso_rotational_velo",
        ],
        "mass_plus_max_separation_torso": [
            "session_mass_kg", "max_rotation_hip_shoulder_separation",
            "max_torso_rotational_velo",
        ],
    }
    rows = []
    for outcome in ["gated_stp_j", "gated_jfp_j"]:
        for name, predictors in panels.items():
            clean = athlete[[outcome, *predictors]].dropna()
            fit = smf.ols(f"{outcome} ~ {' + '.join(predictors)}", data=clean).fit()
            cv_mean, cv_sd = cv_r2(clean, outcome, predictors)
            rows.append(
                {
                    "outcome": outcome,
                    "panel": name,
                    "n": len(clean),
                    "r2": fit.rsquared,
                    "adjusted_r2": fit.rsquared_adj,
                    "cv_r2_mean": cv_mean,
                    "cv_r2_sd": cv_sd,
                }
            )
    return pd.DataFrame(rows)


def standardized_coefficients(athlete: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["gated_stp_j", "gated_jfp_j"]:
        for separation in KINEMATICS[:2]:
            columns = ["session_mass_kg", separation, "max_torso_rotational_velo"]
            clean = athlete[[outcome, *columns]].dropna()
            z = (clean - clean.mean()) / clean.std(ddof=0)
            fit = smf.ols(f"{outcome} ~ {' + '.join(columns)}", data=z).fit()
            for term in columns:
                rows.append({
                    "outcome": outcome,
                    "separation_definition": separation,
                    "term": term,
                    "standardized_beta": fit.params[term],
                    "p": fit.pvalues[term],
                })
    return pd.DataFrame(rows)


def main() -> None:
    energy_cols = [
        "session_pitch", "time", "fp_poi_time", "MER_time",
        "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
        "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr",
    ]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=energy_cols)
    landmarks = pd.read_csv(
        ROOT / "data/full_sig/landmarks.csv",
        usecols=["session_pitch", "time", "shoulder_jc_x", "shoulder_jc_y", "shoulder_jc_z"],
    )
    forces = pd.read_csv(
        ROOT / "data/full_sig/forces_moments.csv",
        usecols=["session_pitch", "time", *[f"shoulder_thorax_force_{a}" for a in "xyz"]],
    )
    frame = energy.merge(
        landmarks, on=["session_pitch", "time"], validate="one_to_one"
    ).merge(forces, on=["session_pitch", "time"], validate="one_to_one")
    if len(frame) != len(energy):
        raise ValueError("Exact time-key merge lost energy-flow rows; no fallback permitted")

    pitch = (
        frame.groupby("session_pitch", sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    poi = pd.read_csv(
        ROOT / "data/poi/poi_metrics.csv", usecols=["session_pitch", *KINEMATICS]
    )
    meta = pd.read_csv(
        ROOT / "data/metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    pitch = pitch.merge(poi, on="session_pitch", validate="one_to_one").merge(
        meta, on="session_pitch", validate="one_to_one"
    )
    if len(pitch) != 411 or pitch["session"].nunique() != 100:
        raise ValueError("Expected 411 pitches and 100 sessions")

    athlete = pitch.groupby("session", as_index=False).agg(
        session_mass_kg=("session_mass_kg", "first"),
        pitch_n=("session_pitch", "size"),
        **{c: (c, "mean") for c in pitch.columns if c not in {
            "session_pitch", "session", "session_mass_kg"
        }},
    )
    correlations = athlete[[
        "gated_stp_j", "gated_jfp_j", "positive_total_j", "session_mass_kg",
        *KINEMATICS, "mean_force_n", "mean_shoulder_jc_speed_m_s",
        "mean_force_velocity_alignment", "thorax_limited_fraction",
    ]].corr()
    models = model_panels(athlete)
    coefficients = standardized_coefficients(athlete)

    OUT.mkdir(exist_ok=True)
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    athlete.to_csv(OUT / "per_pitcher.csv", index=False)
    correlations.to_csv(OUT / "pitcher_correlations.csv")
    models.to_csv(OUT / "model_panels.csv", index=False)
    coefficients.to_csv(OUT / "standardized_coefficients.csv", index=False)

    print(f"coverage: merged_rows={len(frame)}, pitches={len(pitch)}, pitchers={len(athlete)}")
    print(f"max_abs_conservation_error_j={pitch['conservation_error_j'].abs().max():.12g}")
    print("\nPitcher means")
    print(athlete[["positive_total_j", "gated_stp_j", "gated_jfp_j", "stp_negative_j", "jfp_negative_j"]].mean())
    print("\nJFP reconstruction diagnostic (pitch-level distributions)")
    print(pitch[["official_jfp_vs_force_velocity_r", "official_jfp_force_velocity_mae_w"]].describe())
    print("\nPitcher correlations")
    print(correlations.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nModel panels")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nStandardized coefficients")
    print(coefficients.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
