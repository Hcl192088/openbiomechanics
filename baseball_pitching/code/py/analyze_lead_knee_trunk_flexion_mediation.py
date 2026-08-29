"""Test whether lead-knee extension velocity relates to pitch speed through trunk flexion velocity.

This is an observational mediation/path analysis, not a causal identification design.
The primary pitch-level estimates use an athlete-cluster bootstrap. Athlete-mean
and within-athlete-centered models are reported as sensitivity analyses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_FILE = BASE_DIR / "data" / "metadata.csv"
POI_FILE = BASE_DIR / "data" / "poi" / "poi_metrics.csv"
JOINT_VELOS_FILE = BASE_DIR / "data" / "full_sig" / "joint_velos.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "lead_knee_trunk_flexion_mediation_outputs"

EXPOSURE = "lead_knee_extension_angular_velo_max"
MEDIATORS = {
    "peak_forward_flexion_velocity_fp_br": "trunk_flexion_velo_peak_fp_br",
    "peak_forward_flexion_velocity_after_knee_peak": "trunk_flexion_velo_peak_after_knee_peak",
}
OUTCOME = "pitch_speed_mph"
COVARIATES = ["session_height_m", "session_mass_kg", "age_yrs"]
BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 20260829


def derive_fp_br_kinematics() -> pd.DataFrame:
    usecols = [
        "session_pitch",
        "time",
        "fp_poi_time",
        "BR_time",
        "lead_knee_velo_x",
        "torso_velo_x",
    ]
    joint_velos = pd.read_csv(JOINT_VELOS_FILE, usecols=usecols)
    records: list[dict[str, float | str]] = []
    for session_pitch, group in joint_velos.groupby("session_pitch", sort=False):
        fp = group["fp_poi_time"].iloc[0]
        br = group["BR_time"].iloc[0]
        record: dict[str, float | str] = {"session_pitch": session_pitch}
        if pd.isna(fp) or pd.isna(br) or br <= fp:
            record.update(
                lead_knee_velo_peak_reconstructed=np.nan,
                lead_knee_velo_peak_time=np.nan,
                trunk_flexion_velo_peak_fp_br=np.nan,
                trunk_flexion_velo_peak_time=np.nan,
                trunk_flexion_velo_peak_after_knee_peak=np.nan,
                trunk_flexion_peak_after_knee_peak_time=np.nan,
                knee_peak_precedes_trunk_peak=np.nan,
                knee_to_trunk_peak_ms=np.nan,
            )
            records.append(record)
            continue

        window = group.loc[
            group["time"].between(fp, br, inclusive="both"),
            ["time", "lead_knee_velo_x", "torso_velo_x"],
        ].dropna()
        if window.empty:
            records.append(record)
            continue

        # Project validation shows raw torso_velo_x has the opposite sign to
        # d(torso_angle_x)/dt. Flexion-positive velocity is therefore -raw.
        window = window.assign(trunk_flexion_velo=-window["torso_velo_x"])
        knee_idx = window["lead_knee_velo_x"].idxmax()
        trunk_idx = window["trunk_flexion_velo"].idxmax()
        knee_time = float(window.loc[knee_idx, "time"])
        trunk_time = float(window.loc[trunk_idx, "time"])
        after = window.loc[window["time"] >= knee_time]
        after_idx = after["trunk_flexion_velo"].idxmax()

        record.update(
            lead_knee_velo_peak_reconstructed=float(window.loc[knee_idx, "lead_knee_velo_x"]),
            lead_knee_velo_peak_time=knee_time,
            trunk_flexion_velo_peak_fp_br=float(window.loc[trunk_idx, "trunk_flexion_velo"]),
            trunk_flexion_velo_peak_time=trunk_time,
            trunk_flexion_velo_peak_after_knee_peak=float(after.loc[after_idx, "trunk_flexion_velo"]),
            trunk_flexion_peak_after_knee_peak_time=float(after.loc[after_idx, "time"]),
            knee_peak_precedes_trunk_peak=float(knee_time <= trunk_time),
            knee_to_trunk_peak_ms=(trunk_time - knee_time) * 1000.0,
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def prepare_data() -> pd.DataFrame:
    metadata = pd.read_csv(
        METADATA_FILE,
        usecols=["session_pitch", "user", OUTCOME, *COVARIATES],
    )
    poi = pd.read_csv(POI_FILE, usecols=["session_pitch", EXPOSURE])
    derived = derive_fp_br_kinematics()
    data = (
        metadata.merge(poi, on="session_pitch", validate="one_to_one")
        .merge(derived, on="session_pitch", validate="one_to_one")
    )
    valid = data[[EXPOSURE, "lead_knee_velo_peak_reconstructed"]].dropna()
    if len(valid) != len(data):
        raise RuntimeError(f"Expected complete reconstructed exposure data; got {len(valid)}/{len(data)}")
    # The canonical POI exposure is retained for the primary model. Its current
    # source values do not exactly reproduce from the current full-signal file
    # for every pitch, so reconstructed timing is explicitly sensitivity-only.
    return data


def zscore(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = data.copy()
    for column in columns:
        sd = result[column].std(ddof=1)
        if pd.isna(sd) or sd == 0:
            raise ValueError(f"Cannot standardize {column}")
        result[column] = (result[column] - result[column].mean()) / sd
    return result


def fit_paths(data: pd.DataFrame, exposure: str, mediator: str, covariates: list[str]) -> dict[str, float]:
    columns = [exposure, mediator, OUTCOME, *covariates]
    clean = zscore(data[columns].dropna(), columns)
    x = clean[exposure].to_numpy()
    m = clean[mediator].to_numpy()
    y = clean[OUTCOME].to_numpy()
    cov = clean[covariates].to_numpy() if covariates else np.empty((len(clean), 0))
    med_design = np.column_stack([np.ones(len(clean)), x, cov])
    out_design = np.column_stack([np.ones(len(clean)), x, m, cov])
    total_design = np.column_stack([np.ones(len(clean)), x, cov])
    med_beta = np.linalg.lstsq(med_design, m, rcond=None)[0]
    out_beta = np.linalg.lstsq(out_design, y, rcond=None)[0]
    total_beta = np.linalg.lstsq(total_design, y, rcond=None)[0]
    a = float(med_beta[1])
    b = float(out_beta[2])
    indirect = a * b
    total = float(total_beta[1])
    return {
        "n_pitches": float(len(clean)),
        "path_a": a,
        "path_b": b,
        "indirect_effect": indirect,
        "direct_effect": float(out_beta[1]),
        "total_effect": total,
        "proportion_mediated": indirect / total if total != 0 else np.nan,
    }


def cluster_bootstrap(
    data: pd.DataFrame,
    exposure: str,
    mediator: str,
    covariates: list[str],
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, int]:
    columns = [exposure, mediator, OUTCOME, *covariates]
    clusters = [
        group[columns].dropna().to_numpy(dtype=float)
        for _, group in data.groupby("user", sort=False)
    ]
    clusters = [cluster for cluster in clusters if len(cluster)]
    rng = np.random.default_rng(RANDOM_SEED)
    estimates: list[dict[str, float]] = []
    for _ in range(reps):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sample = np.vstack([clusters[index] for index in selected])
        sd = sample.std(axis=0, ddof=1)
        if np.any(sd == 0) or np.any(np.isnan(sd)):
            continue
        z = (sample - sample.mean(axis=0)) / sd
        x, m, y = z[:, 0], z[:, 1], z[:, 2]
        cov = z[:, 3:]
        med_design = np.column_stack([np.ones(len(z)), x, cov])
        out_design = np.column_stack([np.ones(len(z)), x, m, cov])
        total_design = np.column_stack([np.ones(len(z)), x, cov])
        try:
            med_beta = np.linalg.lstsq(med_design, m, rcond=None)[0]
            out_beta = np.linalg.lstsq(out_design, y, rcond=None)[0]
            total_beta = np.linalg.lstsq(total_design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        a = float(med_beta[1])
        b = float(out_beta[2])
        indirect = a * b
        total = float(total_beta[1])
        estimates.append(
            {
                "path_a": a,
                "path_b": b,
                "indirect_effect": indirect,
                "direct_effect": float(out_beta[1]),
                "total_effect": total,
                "proportion_mediated": indirect / total if total != 0 else np.nan,
            }
        )
    if len(estimates) < reps * 0.95:
        raise RuntimeError(f"Only {len(estimates)}/{reps} cluster bootstrap samples succeeded")
    return pd.DataFrame(estimates), len(clusters)


def athlete_mean_data(data: pd.DataFrame, mediator: str) -> pd.DataFrame:
    columns = [EXPOSURE, mediator, OUTCOME, *COVARIATES]
    return data.groupby("user", as_index=False)[columns].mean()


def within_athlete_data(data: pd.DataFrame, mediator: str) -> pd.DataFrame:
    result = data[["user", EXPOSURE, mediator, OUTCOME]].dropna().copy()
    for column in [EXPOSURE, mediator, OUTCOME]:
        result[column] -= result.groupby("user")[column].transform("mean")
    return result


def summarize_model(
    data: pd.DataFrame,
    mediator_name: str,
    mediator: str,
    analysis: str,
    covariates: list[str],
) -> dict[str, float | str]:
    point = fit_paths(data, EXPOSURE, mediator, covariates)
    bootstrap, n_athletes = cluster_bootstrap(data, EXPOSURE, mediator, covariates)
    row: dict[str, float | str] = {
        "analysis": analysis,
        "exposure": EXPOSURE,
        "mediator": mediator_name,
        "mediator_column": mediator,
        "n_athletes": n_athletes,
        **point,
    }
    for metric in [
        "path_a",
        "path_b",
        "indirect_effect",
        "direct_effect",
        "total_effect",
        "proportion_mediated",
    ]:
        row[f"{metric}_ci_lower"] = float(bootstrap[metric].quantile(0.025))
        row[f"{metric}_ci_upper"] = float(bootstrap[metric].quantile(0.975))
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    data.to_csv(OUTPUT_DIR / "derived_pitch_level_data.csv", index=False)

    results: list[dict[str, float | str]] = []
    for mediator_name, mediator in MEDIATORS.items():
        results.append(summarize_model(data, mediator_name, mediator, "pitch_level_clustered", COVARIATES))

        athlete_data = athlete_mean_data(data, mediator)
        results.append(summarize_model(athlete_data, mediator_name, mediator, "athlete_means", COVARIATES))

        within_data = within_athlete_data(data, mediator)
        results.append(summarize_model(within_data, mediator_name, mediator, "within_athlete_centered", []))

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUTPUT_DIR / "mediation_results.csv", index=False)

    timing = data["knee_to_trunk_peak_ms"].dropna()
    timing_summary = pd.DataFrame(
        [
            {
                "n_pitches": len(timing),
                "n_athletes": data["user"].nunique(),
                "knee_peak_precedes_trunk_peak_percent": 100.0 * data["knee_peak_precedes_trunk_peak"].mean(),
                "knee_to_trunk_peak_ms_median": timing.median(),
                "knee_to_trunk_peak_ms_q1": timing.quantile(0.25),
                "knee_to_trunk_peak_ms_q3": timing.quantile(0.75),
                "exposure_reconstruction_exact_n": int(
                    np.isclose(
                        data[EXPOSURE],
                        data["lead_knee_velo_peak_reconstructed"],
                        atol=1e-9,
                    ).sum()
                ),
                "exposure_reconstruction_correlation": data[
                    [EXPOSURE, "lead_knee_velo_peak_reconstructed"]
                ].corr().iloc[0, 1],
                "exposure_reconstruction_median_abs_error": (
                    data[EXPOSURE] - data["lead_knee_velo_peak_reconstructed"]
                ).abs().median(),
                "exposure_reconstruction_max_abs_error": (
                    data[EXPOSURE] - data["lead_knee_velo_peak_reconstructed"]
                ).abs().max(),
            }
        ]
    )
    timing_summary.to_csv(OUTPUT_DIR / "timing_and_validation_summary.csv", index=False)

    display = [
        "analysis",
        "mediator",
        "n_pitches",
        "n_athletes",
        "path_a",
        "path_b",
        "indirect_effect",
        "indirect_effect_ci_lower",
        "indirect_effect_ci_upper",
        "total_effect",
    ]
    print(results_frame[display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTiming and validation:")
    print(timing_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
