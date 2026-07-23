"""Mediation analysis for shoulder-hip separation and pitch speed in pitchers < 6 ft.

The independent unit is the athlete. Multiple pitches are averaged within athlete
before fitting the models. Original source files are read only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_FILE = BASE_DIR / "data" / "metadata.csv"
POI_FILE = BASE_DIR / "data" / "poi" / "poi_metrics.csv"
JOINT_VELOS_FILE = BASE_DIR / "data" / "full_sig" / "joint_velos.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "short_pitcher_mediation_outputs"
HEIGHT_CUTOFF_M = 1.8288
BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 20260723

OUTCOME = "pitch_speed_mph"
COVARIATES = ["session_height_m", "session_mass_kg", "age_yrs"]
EXPOSURES = {
    "separation_at_fp_poi": "rotation_hip_shoulder_separation_fp",
    "maximum_separation": "max_rotation_hip_shoulder_separation",
}
MEDIATORS = {
    "peak_torso_rotation_velocity_fp_br": "max_torso_rotational_velo",
    "peak_separation_closing_speed_fp_br": "separation_closing_speed_fp_br",
}


def derive_closing_speed() -> pd.DataFrame:
    columns = [
        "session_pitch",
        "time",
        "fp_poi_time",
        "BR_time",
        "torso_pelvis_velo_z",
    ]
    joint_velos = pd.read_csv(JOINT_VELOS_FILE, usecols=columns)
    records = []

    for session_pitch, group in joint_velos.groupby("session_pitch", sort=False):
        fp_time = group["fp_poi_time"].iloc[0]
        br_time = group["BR_time"].iloc[0]
        if pd.isna(fp_time) or pd.isna(br_time) or br_time < fp_time:
            closing_speed = np.nan
        else:
            window = group.loc[
                group["time"].between(fp_time, br_time, inclusive="both"),
                "torso_pelvis_velo_z",
            ]
            # Positive angle means separation and negative velocity means closing.
            closing_speed = -window.min() if not window.empty else np.nan
        records.append(
            {
                "session_pitch": session_pitch,
                "separation_closing_speed_fp_br": closing_speed,
            }
        )

    return pd.DataFrame.from_records(records)


def prepare_athlete_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        "session_pitch",
        "user",
        OUTCOME,
        "session_height_m",
        "session_mass_kg",
        "age_yrs",
    ]
    poi_columns = ["session_pitch", *EXPOSURES.values(), "max_torso_rotational_velo"]

    metadata = pd.read_csv(METADATA_FILE, usecols=metadata_columns)
    poi = pd.read_csv(POI_FILE, usecols=poi_columns)
    closing_speed = derive_closing_speed()

    pitch_data = (
        metadata.loc[metadata["session_height_m"] < HEIGHT_CUTOFF_M]
        .merge(poi, on="session_pitch", how="inner", validate="one_to_one")
        .merge(closing_speed, on="session_pitch", how="inner", validate="one_to_one")
    )

    athlete_data = (
        pitch_data.groupby("user", as_index=False)
        .agg(
            n_pitches=("session_pitch", "size"),
            **{
                column: (column, "mean")
                for column in [
                    OUTCOME,
                    *COVARIATES,
                    *EXPOSURES.values(),
                    *MEDIATORS.values(),
                ]
            },
        )
        .sort_values("user")
        .reset_index(drop=True)
    )
    return pitch_data, athlete_data


def zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        standard_deviation = result[column].std(ddof=1)
        if standard_deviation == 0 or pd.isna(standard_deviation):
            raise ValueError(f"Cannot standardize {column}: zero or missing SD")
        result[column] = (
            result[column] - result[column].mean()
        ) / standard_deviation
    return result


def estimate_paths(
    athlete_data: pd.DataFrame, exposure: str, mediator: str
) -> dict[str, float]:
    columns = [exposure, mediator, OUTCOME, *COVARIATES]
    data = zscore(athlete_data[columns].dropna(), columns)

    mediator_model = sm.OLS(
        data[mediator], sm.add_constant(data[[exposure, *COVARIATES]])
    ).fit()
    outcome_model = sm.OLS(
        data[OUTCOME],
        sm.add_constant(data[[exposure, mediator, *COVARIATES]]),
    ).fit()
    total_model = sm.OLS(
        data[OUTCOME], sm.add_constant(data[[exposure, *COVARIATES]])
    ).fit()

    path_a = mediator_model.params[exposure]
    path_b = outcome_model.params[mediator]
    indirect = path_a * path_b
    direct = outcome_model.params[exposure]
    total = total_model.params[exposure]

    return {
        "n_athletes": len(data),
        "path_a": path_a,
        "path_b": path_b,
        "indirect_effect": indirect,
        "direct_effect": direct,
        "total_effect": total,
        "proportion_mediated": indirect / total if total != 0 else np.nan,
        "total_model_r_squared": total_model.rsquared,
        "outcome_model_r_squared": outcome_model.rsquared,
    }


def estimate_paths_fast(
    athlete_data: pd.DataFrame, exposure: str, mediator: str
) -> dict[str, float]:
    """Numerically equivalent OLS path estimates without model-object overhead."""
    columns = [exposure, mediator, OUTCOME, *COVARIATES]
    data = zscore(athlete_data[columns].dropna(), columns)

    def fit(y_column: str, x_columns: list[str]) -> tuple[np.ndarray, float]:
        design = np.column_stack(
            [np.ones(len(data)), data[x_columns].to_numpy(dtype=float)]
        )
        outcome = data[y_column].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
        residuals = outcome - design @ coefficients
        total_sum_squares = np.square(outcome - outcome.mean()).sum()
        r_squared = 1 - np.square(residuals).sum() / total_sum_squares
        return coefficients, r_squared

    mediator_coefficients, _ = fit(mediator, [exposure, *COVARIATES])
    outcome_coefficients, outcome_r_squared = fit(
        OUTCOME, [exposure, mediator, *COVARIATES]
    )
    total_coefficients, total_r_squared = fit(OUTCOME, [exposure, *COVARIATES])

    path_a = mediator_coefficients[1]
    path_b = outcome_coefficients[2]
    indirect = path_a * path_b
    direct = outcome_coefficients[1]
    total = total_coefficients[1]
    return {
        "n_athletes": len(data),
        "path_a": path_a,
        "path_b": path_b,
        "indirect_effect": indirect,
        "direct_effect": direct,
        "total_effect": total,
        "proportion_mediated": indirect / total if total != 0 else np.nan,
        "total_model_r_squared": total_r_squared,
        "outcome_model_r_squared": outcome_r_squared,
    }


def bootstrap_intervals(
    athlete_data: pd.DataFrame,
    exposure: str,
    mediator: str,
    repetitions: int = BOOTSTRAP_REPS,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    columns = [exposure, mediator, OUTCOME, *COVARIATES]
    matrix = athlete_data[columns].dropna().to_numpy(dtype=float)
    estimates = np.empty((repetitions, 8), dtype=float)
    successful = 0

    for _ in range(repetitions):
        indices = rng.integers(0, len(matrix), size=len(matrix))
        sample = matrix[indices]
        try:
            means = sample.mean(axis=0)
            standard_deviations = sample.std(axis=0, ddof=1)
            if np.any(standard_deviations == 0):
                continue
            z = (sample - means) / standard_deviations

            x, m, y = z[:, 0], z[:, 1], z[:, 2]
            covariates = z[:, 3:]
            mediator_design = np.column_stack([np.ones(len(z)), x, covariates])
            outcome_design = np.column_stack(
                [np.ones(len(z)), x, m, covariates]
            )
            total_design = np.column_stack([np.ones(len(z)), x, covariates])

            mediator_beta = np.linalg.lstsq(
                mediator_design, m, rcond=None
            )[0]
            outcome_beta = np.linalg.lstsq(outcome_design, y, rcond=None)[0]
            total_beta = np.linalg.lstsq(total_design, y, rcond=None)[0]

            path_a = mediator_beta[1]
            path_b = outcome_beta[2]
            indirect = path_a * path_b
            direct = outcome_beta[1]
            total = total_beta[1]
            total_residuals = y - total_design @ total_beta
            outcome_residuals = y - outcome_design @ outcome_beta
            total_sum_squares = np.square(y - y.mean()).sum()
            estimates[successful] = [
                path_a,
                path_b,
                indirect,
                direct,
                total,
                indirect / total if total != 0 else np.nan,
                1 - np.square(total_residuals).sum() / total_sum_squares,
                1 - np.square(outcome_residuals).sum() / total_sum_squares,
            ]
            successful += 1
        except np.linalg.LinAlgError:
            continue

    if successful < repetitions * 0.95:
        raise RuntimeError(
            f"Only {successful} of {repetitions} bootstrap samples succeeded"
        )

    interval_columns = [
        "path_a",
        "path_b",
        "indirect_effect",
        "direct_effect",
        "total_effect",
        "proportion_mediated",
        "total_model_r_squared",
        "outcome_model_r_squared",
    ]
    bootstrap = pd.DataFrame(
        estimates[:successful], columns=interval_columns
    )
    return {
        column: (
            bootstrap[column].quantile(0.025),
            bootstrap[column].quantile(0.975),
        )
        for column in interval_columns
    }


def bootstrap_pitch_cluster_intervals(
    pitch_data: pd.DataFrame,
    exposure: str,
    mediator: str,
    repetitions: int = BOOTSTRAP_REPS,
) -> dict[str, tuple[float, float]]:
    """Bootstrap pitch-level models by resampling athletes as whole clusters."""
    rng = np.random.default_rng(RANDOM_SEED)
    columns = [exposure, mediator, OUTCOME, *COVARIATES]
    clusters = [
        group[columns].dropna().to_numpy(dtype=float)
        for _, group in pitch_data.groupby("user", sort=False)
    ]
    estimates = []

    for _ in range(repetitions):
        sampled_clusters = rng.integers(0, len(clusters), size=len(clusters))
        sample = np.vstack([clusters[index] for index in sampled_clusters])
        means = sample.mean(axis=0)
        standard_deviations = sample.std(axis=0, ddof=1)
        if np.any(standard_deviations == 0):
            continue
        z = (sample - means) / standard_deviations

        x, m, y = z[:, 0], z[:, 1], z[:, 2]
        covariates = z[:, 3:]
        mediator_design = np.column_stack([np.ones(len(z)), x, covariates])
        outcome_design = np.column_stack([np.ones(len(z)), x, m, covariates])
        total_design = np.column_stack([np.ones(len(z)), x, covariates])

        try:
            mediator_beta = np.linalg.lstsq(
                mediator_design, m, rcond=None
            )[0]
            outcome_beta = np.linalg.lstsq(outcome_design, y, rcond=None)[0]
            total_beta = np.linalg.lstsq(total_design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        path_a = mediator_beta[1]
        path_b = outcome_beta[2]
        indirect = path_a * path_b
        direct = outcome_beta[1]
        total = total_beta[1]
        total_residuals = y - total_design @ total_beta
        outcome_residuals = y - outcome_design @ outcome_beta
        total_sum_squares = np.square(y - y.mean()).sum()
        estimates.append(
            {
                "path_a": path_a,
                "path_b": path_b,
                "indirect_effect": indirect,
                "direct_effect": direct,
                "total_effect": total,
                "proportion_mediated": indirect / total
                if total != 0
                else np.nan,
                "total_model_r_squared": 1
                - np.square(total_residuals).sum() / total_sum_squares,
                "outcome_model_r_squared": 1
                - np.square(outcome_residuals).sum() / total_sum_squares,
            }
        )

    if len(estimates) < repetitions * 0.95:
        raise RuntimeError(
            f"Only {len(estimates)} of {repetitions} cluster bootstrap samples succeeded"
        )

    bootstrap = pd.DataFrame(estimates)
    return {
        column: (
            bootstrap[column].quantile(0.025),
            bootstrap[column].quantile(0.975),
        )
        for column in bootstrap.columns
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pitch_data, athlete_data = prepare_athlete_data()
    athlete_data.to_csv(OUTPUT_DIR / "athlete_level_data.csv", index=False)

    results = []
    for exposure_name, exposure in EXPOSURES.items():
        for mediator_name, mediator in MEDIATORS.items():
            point = estimate_paths(athlete_data, exposure, mediator)
            fast_point = estimate_paths_fast(athlete_data, exposure, mediator)
            for metric in [
                "path_a",
                "path_b",
                "indirect_effect",
                "direct_effect",
                "total_effect",
                "proportion_mediated",
                "total_model_r_squared",
                "outcome_model_r_squared",
            ]:
                if not np.isclose(point[metric], fast_point[metric], atol=1e-10):
                    raise RuntimeError(
                        f"Fast OLS disagrees with statsmodels for {metric}"
                    )
            intervals = bootstrap_intervals(athlete_data, exposure, mediator)
            row = {
                "exposure": exposure_name,
                "exposure_column": exposure,
                "mediator": mediator_name,
                "mediator_column": mediator,
                "n_pitches": len(pitch_data),
                **point,
            }
            for metric, (lower, upper) in intervals.items():
                row[f"{metric}_ci_lower"] = lower
                row[f"{metric}_ci_upper"] = upper
            results.append(row)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUTPUT_DIR / "mediation_results.csv", index=False)

    pitch_results = []
    naive_pitch_results = []
    for exposure_name, exposure in EXPOSURES.items():
        for mediator_name, mediator in MEDIATORS.items():
            point = estimate_paths(pitch_data, exposure, mediator)
            intervals = bootstrap_pitch_cluster_intervals(
                pitch_data, exposure, mediator
            )
            row = {
                "exposure": exposure_name,
                "exposure_column": exposure,
                "mediator": mediator_name,
                "mediator_column": mediator,
                **point,
                "n_pitches": len(pitch_data),
                "n_athletes": pitch_data["user"].nunique(),
            }
            for metric, (lower, upper) in intervals.items():
                row[f"{metric}_ci_lower"] = lower
                row[f"{metric}_ci_upper"] = upper
            pitch_results.append(row)

            naive_intervals = bootstrap_intervals(
                pitch_data, exposure, mediator
            )
            naive_row = {
                "exposure": exposure_name,
                "exposure_column": exposure,
                "mediator": mediator_name,
                "mediator_column": mediator,
                **point,
                "n_pitches": len(pitch_data),
                "n_athletes": pitch_data["user"].nunique(),
            }
            for metric, (lower, upper) in naive_intervals.items():
                naive_row[f"{metric}_ci_lower"] = lower
                naive_row[f"{metric}_ci_upper"] = upper
            naive_pitch_results.append(naive_row)

    pitch_results_frame = pd.DataFrame(pitch_results)
    pitch_results_frame.to_csv(
        OUTPUT_DIR / "pitch_level_clustered_mediation_results.csv", index=False
    )
    naive_pitch_results_frame = pd.DataFrame(naive_pitch_results)
    naive_pitch_results_frame.to_csv(
        OUTPUT_DIR / "pitch_level_naive_bootstrap_sensitivity.csv", index=False
    )

    display_columns = [
        "exposure",
        "mediator",
        "n_athletes",
        "total_model_r_squared",
        "path_a",
        "path_b",
        "indirect_effect",
        "indirect_effect_ci_lower",
        "indirect_effect_ci_upper",
        "direct_effect",
        "total_effect",
        "proportion_mediated",
        "proportion_mediated_ci_lower",
        "proportion_mediated_ci_upper",
    ]
    print(results_frame[display_columns].to_string(index=False, float_format="%.4f"))
    print("\nPitch-level models with athlete-cluster bootstrap:")
    print(
        pitch_results_frame[display_columns].to_string(
            index=False, float_format="%.4f"
        )
    )
    print("\nNaive pitch-level bootstrap (sensitivity only; pitches treated as independent):")
    print(
        naive_pitch_results_frame[display_columns].to_string(
            index=False, float_format="%.4f"
        )
    )
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
