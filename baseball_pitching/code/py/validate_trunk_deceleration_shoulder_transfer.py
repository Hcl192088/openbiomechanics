from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]


def residualize(values, covariates):
    design = np.column_stack([np.ones(len(covariates)), covariates])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlation(frame, x, y, controls=()):
    clean = frame[[x, y, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    x_values = clean[x].to_numpy(float)
    y_values = clean[y].to_numpy(float)
    if controls:
        covariates = clean[list(controls)].to_numpy(float)
        x_values = residualize(x_values, covariates)
        y_values = residualize(y_values, covariates)
    r, p = stats.pearsonr(x_values, y_values)
    return len(clean), r, p


def multivariable_summary(frame, target, predictors):
    clean = frame[[target, *predictors]].replace([np.inf, -np.inf], np.nan)
    y = clean[target].to_numpy(float)
    x = clean[predictors].to_numpy(float)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(x, y)
    fitted = model.predict(x)
    r2 = r2_score(y, fitted)
    n = len(y)
    k = len(predictors)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    folds = KFold(n_splits=10, shuffle=True, random_state=20260729)
    cross_validated = cross_val_predict(model, x, y, cv=folds)
    return r2, adjusted_r2, r2_score(y, cross_validated)


def build_pitch_metrics():
    poi = pd.read_csv(ROOT / "data" / "poi" / "poi_metrics.csv")
    metadata = pd.read_csv(ROOT / "data" / "metadata.csv")
    velos = pd.read_csv(
        ROOT / "data" / "full_sig" / "joint_velos.csv",
        usecols=["session_pitch", "time", "torso_velo_z", "fp_poi_time", "BR_time"],
    )
    energy = pd.read_csv(
        ROOT / "data" / "full_sig" / "energy_flow.csv",
        usecols=[
            "session_pitch",
            "time",
            "shoulder_energy_transfer_stp",
            "shoulder_energy_transfer_jfp",
            "fp_poi_time",
            "BR_time",
        ],
    )

    rows = []
    for session_pitch, group in velos.groupby("session_pitch", sort=False):
        group = group.sort_values("time")
        fp = group["fp_poi_time"].dropna().unique()
        br = group["BR_time"].dropna().unique()
        if len(fp) != 1 or len(br) != 1:
            raise ValueError(f"Ambiguous FP/BR for {session_pitch}")
        window = group[
            (group["time"] >= float(fp[0])) & (group["time"] <= float(br[0]))
        ].dropna(subset=["torso_velo_z"])
        if len(window) < 3:
            continue

        omega = window["torso_velo_z"].abs().to_numpy(float)
        time = window["time"].to_numpy(float)
        peak_index = int(np.argmax(omega))
        post_omega = omega[peak_index:]
        post_time = time[peak_index:]
        if len(post_omega) < 2:
            continue
        total_drop = post_omega[0] - post_omega[-1]
        integrated_negative = np.sum(np.maximum(-np.diff(post_omega), 0))
        rows.append(
            {
                "session_pitch": session_pitch,
                "omega_peak": post_omega[0],
                "omega_br": post_omega[-1],
                "torso_peak_time": post_time[0],
                "delta_omega": total_drop,
                "integrated_negative_domega": integrated_negative,
                "delta_omega_sq": post_omega[0] ** 2 - post_omega[-1] ** 2,
                "fraction_peak_lost": total_drop / post_omega[0],
                "peak_to_br_duration": post_time[-1] - post_time[0],
            }
        )

    pitch = pd.DataFrame(rows)
    peak_time_by_pitch = pitch.set_index("session_pitch")["torso_peak_time"]

    energy_rows = []
    for session_pitch, group in energy.groupby("session_pitch", sort=False):
        group = group.sort_values("time")
        fp = group["fp_poi_time"].dropna().unique()
        br = group["BR_time"].dropna().unique()
        if len(fp) != 1 or len(br) != 1:
            raise ValueError(f"Ambiguous energy FP/BR for {session_pitch}")
        window = group[
            (group["time"] >= float(fp[0])) & (group["time"] <= float(br[0]))
        ].dropna(subset=["shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp"])
        if len(window) < 2:
            continue
        time = window["time"].to_numpy(float)
        stp = np.trapezoid(window["shoulder_energy_transfer_stp"], time)
        jfp = np.trapezoid(window["shoulder_energy_transfer_jfp"], time)
        peak_time = float(peak_time_by_pitch.loc[session_pitch])
        peak_window = window[window["time"] >= peak_time]
        if len(peak_window) < 2:
            continue
        peak_window_time = peak_window["time"].to_numpy(float)
        stp_peak_br = np.trapezoid(
            peak_window["shoulder_energy_transfer_stp"], peak_window_time
        )
        jfp_peak_br = np.trapezoid(
            peak_window["shoulder_energy_transfer_jfp"], peak_window_time
        )
        energy_rows.append(
            {
                "session_pitch": session_pitch,
                "shoulder_stp_fp_br": stp,
                "shoulder_jfp_fp_br": jfp,
                "shoulder_total_reconstructed": stp + jfp,
                "shoulder_stp_peak_br": stp_peak_br,
                "shoulder_total_peak_br": stp_peak_br + jfp_peak_br,
            }
        )

    columns = [
        "session_pitch",
        "shoulder_transfer_fp_br",
        "max_shoulder_internal_rotational_velo",
        "rotation_hip_shoulder_separation_fp",
        "max_shoulder_external_rotation",
        "max_torso_rotational_velo",
        "shoulder_horizontal_abduction_fp",
        "torso_anterior_tilt_br",
        "torso_anterior_tilt_mer",
        "torso_rotation_fp",
        "max_shoulder_horizontal_abduction",
    ]
    return (
        pitch.merge(pd.DataFrame(energy_rows), on="session_pitch", validate="one_to_one")
        .merge(poi[columns], on="session_pitch", validate="one_to_one")
        .merge(
            metadata[
                [
                    "session_pitch",
                    "session",
                    "session_mass_kg",
                    "session_height_m",
                    "pitch_speed_mph",
                ]
            ],
            on="session_pitch",
            validate="one_to_one",
        )
    )


def main():
    pitch = build_pitch_metrics()
    athlete = pitch.groupby("session", as_index=False).mean(numeric_only=True)
    inertia = pd.read_csv(
        ROOT / "data" / "poi" / "thorax_inertia_estimates.csv",
        usecols=["session", "rta_izz_kg_m2"],
    )
    athlete = athlete.merge(inertia, on="session", how="inner", validate="one_to_one")
    if len(athlete) != len(inertia):
        raise ValueError(
            f"Inertia join mismatch: athlete rows={len(athlete)}, inertia rows={len(inertia)}"
        )
    athlete["omega_peak_sq"] = athlete["omega_peak"] ** 2
    athlete["omega_br_sq"] = athlete["omega_br"] ** 2

    print(f"Pitch rows: {len(pitch)}; athlete rows: {len(athlete)}")
    print(
        "POI total vs reconstructed STP+JFP: "
        f"r={pitch['shoulder_transfer_fp_br'].corr(pitch['shoulder_total_reconstructed']):.6f}"
    )
    print(
        "delta omega vs integrated negative changes: "
        f"r={athlete['delta_omega'].corr(athlete['integrated_negative_domega']):.6f}"
    )
    print(
        f"Mean delta omega={athlete['delta_omega'].mean():.3f} deg/s; "
        f"mean fraction peak lost={athlete['fraction_peak_lost'].mean():.4%}"
    )

    targets = ["shoulder_stp_fp_br", "shoulder_transfer_fp_br"]
    predictors = ["delta_omega", "delta_omega_sq"]
    for target in targets:
        print(f"\nTarget: {target}")
        athlete[f"{target}_per_kg"] = athlete[target] / athlete["session_mass_kg"]
        for predictor in predictors:
            raw = correlation(athlete, predictor, target)
            partial = correlation(athlete, predictor, target, ("session_mass_kg",))
            ratio = correlation(athlete, predictor, f"{target}_per_kg")
            predictor_partial = correlation(
                athlete, predictor, f"{target}_per_kg", ("session_mass_kg",)
            )
            print(
                f"{predictor:16s} raw r={raw[1]: .4f}, p={raw[2]:.4f}; "
                f"partial-mass r={partial[1]: .4f}, p={partial[2]:.4f}; "
                f"J/kg r={ratio[1]: .4f}, p={ratio[2]:.4f}; "
                f"J/kg+mass r={predictor_partial[1]: .4f}, p={predictor_partial[2]:.4f}"
            )

    athlete["mass_delta_omega_sq"] = (
        athlete["session_mass_kg"] * athlete["delta_omega_sq"]
    )
    athlete["mass_height_sq_delta_omega_sq"] = (
        athlete["session_mass_kg"]
        * athlete["session_height_m"] ** 2
        * athlete["delta_omega_sq"]
    )
    athlete["mass_power_delta_omega_sq"] = (
        athlete["session_mass_kg"] ** 1.579441 * athlete["delta_omega_sq"]
    )
    athlete["delta_kz_j"] = (
        0.5
        * athlete["rta_izz_kg_m2"]
        * athlete["delta_omega_sq"]
        * (np.pi / 180.0) ** 2
    )
    print(
        "Reconstructed delta Kz J: "
        f"mean={athlete['delta_kz_j'].mean():.3f}, "
        f"median={athlete['delta_kz_j'].median():.3f}, "
        f"range={athlete['delta_kz_j'].min():.3f}..{athlete['delta_kz_j'].max():.3f}"
    )
    print("\nMass-weighted rotational-energy proxies")
    for target in targets:
        print(f"\nTarget: {target}")
        for label, baseline, proxy in [
            (
                "mass * delta omega squared",
                ["session_mass_kg"],
                "mass_delta_omega_sq",
            ),
            (
                "mass * height^2 * delta omega squared",
                ["session_mass_kg", "session_height_m"],
                "mass_height_sq_delta_omega_sq",
            ),
            (
                "mass^1.579 * delta omega squared",
                ["session_mass_kg"],
                "mass_power_delta_omega_sq",
            ),
            (
                "reconstructed 0.5 * Izz * delta omega squared",
                ["session_mass_kg"],
                "delta_kz_j",
            ),
        ]:
            raw = correlation(athlete, proxy, target)
            baseline_model = sm.OLS(
                athlete[target],
                sm.add_constant(athlete[baseline]),
            ).fit()
            full_model = sm.OLS(
                athlete[target],
                sm.add_constant(athlete[[*baseline, proxy]]),
            ).fit()
            print(
                f"{label:42s} raw_r={raw[1]: .4f}, "
                f"raw_R2={raw[1] ** 2:.4f}, raw_p={raw[2]:.4f}; "
                f"baseline_R2={baseline_model.rsquared:.4f}, "
                f"full_R2={full_model.rsquared:.4f}, "
                f"incremental_R2="
                f"{full_model.rsquared - baseline_model.rsquared:.4f}, "
                f"proxy_p={full_model.pvalues[proxy]:.4f}"
            )

    print("\nExact reconstructed delta Kz partial correlations")
    for target in [*targets, "shoulder_stp_peak_br", "shoulder_total_peak_br"]:
        raw = correlation(athlete, "delta_kz_j", target)
        partial_mass = correlation(
            athlete, "delta_kz_j", target, ("session_mass_kg",)
        )
        partial_mass_height = correlation(
            athlete,
            "delta_kz_j",
            target,
            ("session_mass_kg", "session_height_m"),
        )
        print(
            f"{target}: raw_r={raw[1]:.4f}, raw_p={raw[2]:.4f}; "
            f"partial_mass_r={partial_mass[1]:.4f}, p={partial_mass[2]:.4f}; "
            f"partial_mass_height_r={partial_mass_height[1]:.4f}, "
            f"p={partial_mass_height[2]:.4f}"
        )

    print("\nMass associations")
    for variable in [
        "shoulder_stp_fp_br",
        "shoulder_transfer_fp_br",
        "delta_omega",
        "delta_omega_sq",
    ]:
        result = correlation(athlete, "session_mass_kg", variable)
        print(f"{variable:30s} r={result[1]: .4f}, p={result[2]:.4f}")

    feature_sets = {
        "shoulder_stp_fp_br": [
            "max_shoulder_internal_rotational_velo",
            "rotation_hip_shoulder_separation_fp",
            "max_shoulder_external_rotation",
            "max_torso_rotational_velo",
            "shoulder_horizontal_abduction_fp",
            "delta_omega_sq",
        ],
        "shoulder_transfer_fp_br": [
            "shoulder_horizontal_abduction_fp",
            "rotation_hip_shoulder_separation_fp",
            "torso_anterior_tilt_br",
            "torso_anterior_tilt_mer",
            "max_shoulder_external_rotation",
            "torso_rotation_fp",
            "max_shoulder_horizontal_abduction",
            "max_torso_rotational_velo",
            "delta_omega_sq",
        ],
    }
    print("\nExploratory multivariable summaries")
    print("10-fold CV uses a fixed shuffled split; features were preselected on this dataset.")
    for target, features in feature_sets.items():
        print(f"\nTarget: {target}")
        for label, predictors in [
            ("mass only", ["session_mass_kg"]),
            ("listed features", features),
            ("mass + listed features", ["session_mass_kg", *features]),
        ]:
            r2, adjusted_r2, cross_validated_r2 = multivariable_summary(
                athlete, target, predictors
            )
            print(
                f"{label:24s} k={len(predictors):2d} "
                f"R2={r2:.4f} adjusted_R2={adjusted_r2:.4f} "
                f"10-fold_CV_R2={cross_validated_r2:.4f}"
            )

    reduced_total_base = [
        "shoulder_horizontal_abduction_fp",
        "rotation_hip_shoulder_separation_fp",
        "torso_anterior_tilt_br",
        "max_shoulder_external_rotation",
        "torso_rotation_fp",
    ]
    print("\nDeduplicated total-transfer models")
    print("Removed max shoulder HAb and MER trunk tilt because |r| > 0.80.")
    for label, final_feature in [
        ("retain max torso velocity", "max_torso_rotational_velo"),
        ("retain delta omega squared", "delta_omega_sq"),
    ]:
        predictors = [*reduced_total_base, final_feature]
        r2, adjusted_r2, cross_validated_r2 = multivariable_summary(
            athlete,
            "shoulder_transfer_fp_br",
            ["session_mass_kg", *predictors],
        )
        print(
            f"{label:28s} k={len(predictors) + 1:2d} "
            f"R2={r2:.4f} adjusted_R2={adjusted_r2:.4f} "
            f"10-fold_CV_R2={cross_validated_r2:.4f}"
        )

    print("\nSquared-speed decomposition")
    for label, squared_features in [
        ("peak squared", ["omega_peak_sq"]),
        ("delta squared", ["delta_omega_sq"]),
        ("peak squared + delta squared", ["omega_peak_sq", "delta_omega_sq"]),
        ("peak squared + BR squared", ["omega_peak_sq", "omega_br_sq"]),
    ]:
        predictors = [*reduced_total_base, *squared_features]
        r2, adjusted_r2, cross_validated_r2 = multivariable_summary(
            athlete,
            "shoulder_transfer_fp_br",
            ["session_mass_kg", *predictors],
        )
        print(
            f"{label:30s} k={len(predictors) + 1:2d} "
            f"R2={r2:.4f} adjusted_R2={adjusted_r2:.4f} "
            f"10-fold_CV_R2={cross_validated_r2:.4f}"
        )

    print("\nSquared-speed relationships with pitch speed")
    for predictor in [
        "omega_peak_sq",
        "omega_br_sq",
        "delta_omega_sq",
        "fraction_peak_lost",
    ]:
        raw = correlation(athlete, predictor, "pitch_speed_mph")
        partial = correlation(
            athlete,
            predictor,
            "pitch_speed_mph",
            ("session_mass_kg",),
        )
        print(
            f"{predictor:24s} raw_r={raw[1]: .4f} raw_p={raw[2]:.4f} "
            f"partial_mass_r={partial[1]: .4f} partial_mass_p={partial[2]:.4f}"
        )

    joint_predictors = ["session_mass_kg", "omega_peak_sq", "omega_br_sq"]
    standardized_x = athlete[joint_predictors].apply(
        lambda column: (column - column.mean()) / column.std(ddof=0)
    )
    standardized_y = (
        athlete["pitch_speed_mph"] - athlete["pitch_speed_mph"].mean()
    ) / athlete["pitch_speed_mph"].std(ddof=0)
    joint_speed_model = sm.OLS(
        standardized_y,
        sm.add_constant(standardized_x),
    ).fit()
    print(
        "Joint pitch-speed model: "
        f"R2={joint_speed_model.rsquared:.4f} "
        f"adjusted_R2={joint_speed_model.rsquared_adj:.4f}"
    )
    for predictor in joint_predictors:
        print(
            f"{predictor:24s} beta={joint_speed_model.params[predictor]: .4f} "
            f"p={joint_speed_model.pvalues[predictor]:.4f}"
        )


if __name__ == "__main__":
    main()
