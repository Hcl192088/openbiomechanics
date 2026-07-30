from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
NODES = np.linspace(0.0, 100.0, 101)
N_PERMUTATIONS = 10000
RANDOM_SEED = 20260730


def residualize_matrix(values, covariate):
    design = np.column_stack([np.ones(len(covariate)), covariate])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def standardize_columns(values):
    centered = values - values.mean(axis=0)
    scale = centered.std(axis=0, ddof=1)
    if np.any(scale == 0):
        raise ValueError("A time-series node has zero between-athlete variance.")
    return centered / scale


def contiguous_true_regions(mask):
    padded = np.r_[False, mask, False].astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts, ends))


def main():
    energy = pd.read_csv(
        ROOT / "data" / "full_sig" / "energy_flow.csv",
        usecols=[
            "session_pitch",
            "time",
            "fp_poi_time",
            "MER_time",
            "BR_time",
            "thorax_dist_seg_pwr",
            "upper_arm_prox_seg_pwr",
            "shoulder_energy_transfer_jfp",
        ],
    ).dropna()
    velos = pd.read_csv(
        ROOT / "data" / "full_sig" / "joint_velos.csv",
        usecols=["session_pitch", "time", "torso_velo_z"],
    ).dropna()
    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    data = energy.merge(
        velos,
        on=["session_pitch", "time"],
        validate="one_to_one",
    )
    data["thorax_stp"] = (
        data["thorax_dist_seg_pwr"] + data["shoulder_energy_transfer_jfp"]
    )
    data["upper_arm_stp"] = (
        data["upper_arm_prox_seg_pwr"] - data["shoulder_energy_transfer_jfp"]
    )
    opposite = data["thorax_stp"] * data["upper_arm_stp"] < 0
    data["transfer_power"] = np.where(
        opposite,
        np.minimum(data["thorax_stp"].abs(), data["upper_arm_stp"].abs()),
        0.0,
    )
    data["upper_arm_limited_power"] = np.where(
        opposite & (data["upper_arm_stp"].abs() < data["thorax_stp"].abs()),
        data["transfer_power"],
        0.0,
    )
    data["thorax_limited_power"] = np.where(
        opposite & (data["thorax_stp"].abs() < data["upper_arm_stp"].abs()),
        data["transfer_power"],
        0.0,
    )

    curve_rows = []
    for session_pitch, group in data.groupby("session_pitch", sort=False):
        group = group[
            (group["time"] >= group["fp_poi_time"])
            & (group["time"] <= group["BR_time"])
        ].sort_values("time")
        if len(group) < 3:
            continue
        time = group["time"].to_numpy(float)
        duration = float(group["BR_time"].iloc[0] - group["fp_poi_time"].iloc[0])
        phase = 100 * (time - float(group["fp_poi_time"].iloc[0])) / duration
        omega = group["torso_velo_z"].abs().to_numpy(float)
        signed_deceleration = -np.gradient(omega, time)
        peak_index = int(np.argmax(omega))
        row = {
            "session_pitch": session_pitch,
            "peak_phase": float(phase[peak_index]),
            "mer_phase": float(
                100
                * (
                    group["MER_time"].iloc[0]
                    - group["fp_poi_time"].iloc[0]
                )
                / duration
            ),
        }
        variables = {
            "signed_deceleration": signed_deceleration,
            "transfer_power": group["transfer_power"].to_numpy(float),
            "upper_arm_limited_power": group[
                "upper_arm_limited_power"
            ].to_numpy(float),
            "thorax_limited_power": group[
                "thorax_limited_power"
            ].to_numpy(float),
        }
        for variable, values in variables.items():
            interpolated = np.interp(NODES, phase, values)
            row.update(
                {
                    f"{variable}_{node}": value
                    for node, value in enumerate(interpolated)
                }
            )
            postpeak_phase = (
                100
                * (time[peak_index:] - time[peak_index])
                / (time[-1] - time[peak_index])
            )
            postpeak_interpolated = np.interp(
                NODES,
                postpeak_phase,
                values[peak_index:],
            )
            row.update(
                {
                    f"postpeak_{variable}_{node}": value
                    for node, value in enumerate(postpeak_interpolated)
                }
            )
        curve_rows.append(row)

    pitch = pd.DataFrame(curve_rows).merge(
        metadata,
        on="session_pitch",
        validate="one_to_one",
    )
    athlete = pitch.groupby("session", as_index=False).mean(numeric_only=True)
    mass = athlete["session_mass_kg"].to_numpy(float)

    decel_columns = [f"signed_deceleration_{node}" for node in range(101)]
    decel_residual = residualize_matrix(
        athlete[decel_columns].to_numpy(float),
        mass,
    )
    decel_z = standardize_columns(decel_residual)

    power_variables = [
        "transfer_power",
        "upper_arm_limited_power",
        "thorax_limited_power",
    ]
    power_z = {}
    observed = {}
    for variable in power_variables:
        columns = [f"{variable}_{node}" for node in range(101)]
        residual = residualize_matrix(
            athlete[columns].to_numpy(float),
            mass,
        )
        power_z[variable] = standardize_columns(residual)
        observed[variable] = (
            decel_z * power_z[variable]
        ).sum(axis=0) / (len(athlete) - 1)

    rng = np.random.default_rng(RANDOM_SEED)
    max_abs_r = np.empty(N_PERMUTATIONS)
    for permutation_index in range(N_PERMUTATIONS):
        order = rng.permutation(len(athlete))
        max_abs_r[permutation_index] = max(
            np.abs(
                (decel_z * power_z[variable][order]).sum(axis=0)
                / (len(athlete) - 1)
            ).max()
            for variable in power_variables
        )
    threshold = float(np.quantile(max_abs_r, 0.95, method="higher"))

    print(
        f"Pitch rows={len(pitch)}; athlete rows={len(athlete)}; "
        f"peak torso-speed phase median={athlete['peak_phase'].median():.3f}%, "
        f"IQR=[{athlete['peak_phase'].quantile(.25):.3f}, "
        f"{athlete['peak_phase'].quantile(.75):.3f}]%; "
        f"MER median={athlete['mer_phase'].median():.3f}%"
    )
    print(
        f"Global max-|r| permutation threshold across 3 power curves: "
        f"signed deceleration={threshold:.4f}; "
        f"permutations={N_PERMUTATIONS}"
    )

    print("\nsigned deceleration")
    for variable in power_variables:
        r_curve = observed[variable]
        peak_index = int(np.argmax(np.abs(r_curve)))
        regions = contiguous_true_regions(np.abs(r_curve) >= threshold)
        print(
            f"{variable}: peak r={r_curve[peak_index]:.4f} "
            f"at {NODES[peak_index]:.1f}%"
        )
        if not regions:
            print("  No FWER-significant interval")
        for start, end in regions:
            pointwise_fwer_p = (
                1
                + np.sum(
                    max_abs_r >= np.max(np.abs(r_curve[start : end + 1]))
                )
            ) / (N_PERMUTATIONS + 1)
            print(
                f"  FWER interval {NODES[start]:.1f}-"
                f"{NODES[end]:.1f}%, max-stat p={pointwise_fwer_p:.6f}"
            )

    postpeak_decel_columns = [
        f"postpeak_signed_deceleration_{node}" for node in range(101)
    ]
    postpeak_decel_z = standardize_columns(
        residualize_matrix(
            athlete[postpeak_decel_columns].to_numpy(float),
            mass,
        )
    )
    postpeak_power_z = {}
    postpeak_observed = {}
    for variable in power_variables:
        columns = [f"postpeak_{variable}_{node}" for node in range(101)]
        postpeak_power_z[variable] = standardize_columns(
            residualize_matrix(
                athlete[columns].to_numpy(float),
                mass,
            )
        )
        postpeak_observed[variable] = (
            postpeak_decel_z * postpeak_power_z[variable]
        ).sum(axis=0) / (len(athlete) - 1)

    postpeak_max_abs_r = np.empty(N_PERMUTATIONS)
    rng = np.random.default_rng(RANDOM_SEED)
    for permutation_index in range(N_PERMUTATIONS):
        order = rng.permutation(len(athlete))
        postpeak_max_abs_r[permutation_index] = max(
            np.abs(
                (
                    postpeak_decel_z
                    * postpeak_power_z[variable][order]
                ).sum(axis=0)
                / (len(athlete) - 1)
            ).max()
            for variable in power_variables
        )
    postpeak_threshold = float(
        np.quantile(postpeak_max_abs_r, 0.95, method="higher")
    )
    print(
        "\nPeak-torso-speed to BR normalized analysis; "
        f"global max-|r| threshold={postpeak_threshold:.4f}"
    )
    for variable in power_variables:
        r_curve = postpeak_observed[variable]
        peak_index = int(np.argmax(np.abs(r_curve)))
        regions = contiguous_true_regions(np.abs(r_curve) >= postpeak_threshold)
        print(
            f"{variable}: peak r={r_curve[peak_index]:.4f} "
            f"at {NODES[peak_index]:.1f}% of peak-to-BR"
        )
        if not regions:
            print("  No FWER-significant interval")
        for start, end in regions:
            pointwise_fwer_p = (
                1
                + np.sum(
                    postpeak_max_abs_r
                    >= np.max(np.abs(r_curve[start : end + 1]))
                )
            ) / (N_PERMUTATIONS + 1)
            print(
                f"  FWER interval {NODES[start]:.1f}-"
                f"{NODES[end]:.1f}% of peak-to-BR, "
                f"max-stat p={pointwise_fwer_p:.6f}"
            )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = {
        "transfer_power": "Total STP transfer power",
        "upper_arm_limited_power": "Upper-arm-limited STP power",
        "thorax_limited_power": "Thorax-limited STP power",
    }
    peak_median = athlete["peak_phase"].median()
    peak_q1 = athlete["peak_phase"].quantile(0.25)
    peak_q3 = athlete["peak_phase"].quantile(0.75)
    mer_median = athlete["mer_phase"].median()
    for axis, variable in zip(axes, power_variables):
        axis.plot(NODES, observed[variable], color="#2457A7", linewidth=2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(threshold, color="#B33A3A", linestyle=":", linewidth=1)
        axis.axhline(-threshold, color="#B33A3A", linestyle=":", linewidth=1)
        axis.axvspan(peak_q1, peak_q3, color="#6A9BD8", alpha=0.14)
        axis.axvline(peak_median, color="#2457A7", linestyle="--", linewidth=1)
        axis.axvline(mer_median, color="#E08A00", linestyle="--", linewidth=1)
        axis.set_ylabel("Partial r")
        axis.set_title(labels[variable])
        axis.set_ylim(-0.55, 0.55)
    axes[-1].set_xlabel("Normalized FP–BR phase (%)")
    fig.suptitle(
        "Instantaneous trunk deceleration vs instantaneous shoulder STP power\n"
        "Athlete-level partial correlations controlling body mass; "
        "global max-statistic correction"
    )
    fig.tight_layout()
    output = ROOT / "imgs" / "instantaneous_trunk_deceleration_stp_correlation.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, variable in zip(axes, power_variables):
        axis.plot(
            NODES,
            postpeak_observed[variable],
            color="#2457A7",
            linewidth=2,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(
            postpeak_threshold,
            color="#B33A3A",
            linestyle=":",
            linewidth=1,
        )
        axis.axhline(
            -postpeak_threshold,
            color="#B33A3A",
            linestyle=":",
            linewidth=1,
        )
        axis.set_ylabel("Partial r")
        axis.set_title(labels[variable])
        axis.set_ylim(-0.55, 0.55)
    axes[-1].set_xlabel("Normalized peak-torso-speed to BR phase (%)")
    fig.suptitle(
        "Instantaneous trunk deceleration vs instantaneous shoulder STP power\n"
        "Peak torso speed to BR; athlete-level, body-mass adjusted, "
        "global max-statistic correction"
    )
    fig.tight_layout()
    postpeak_output = (
        ROOT / "imgs" / "postpeak_trunk_deceleration_stp_correlation.png"
    )
    fig.savefig(postpeak_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {postpeak_output}")


if __name__ == "__main__":
    main()
