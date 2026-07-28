from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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
                "delta_omega": total_drop,
                "integrated_negative_domega": integrated_negative,
                "delta_omega_sq": post_omega[0] ** 2 - post_omega[-1] ** 2,
                "fraction_peak_lost": total_drop / post_omega[0],
                "peak_to_br_duration": post_time[-1] - post_time[0],
            }
        )

    pitch = pd.DataFrame(rows)

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
        energy_rows.append(
            {
                "session_pitch": session_pitch,
                "shoulder_stp_fp_br": stp,
                "shoulder_jfp_fp_br": jfp,
                "shoulder_total_reconstructed": stp + jfp,
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
            metadata[["session_pitch", "session", "session_mass_kg"]],
            on="session_pitch",
            validate="one_to_one",
        )
    )


def main():
    pitch = build_pitch_metrics()
    athlete = pitch.groupby("session", as_index=False).mean(numeric_only=True)

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

    print("\nMass associations")
    for variable in [
        "shoulder_stp_fp_br",
        "shoulder_transfer_fp_br",
        "delta_omega",
        "delta_omega_sq",
    ]:
        result = correlation(athlete, "session_mass_kg", variable)
        print(f"{variable:30s} r={result[1]: .4f}, p={result[2]:.4f}")


if __name__ == "__main__":
    main()
