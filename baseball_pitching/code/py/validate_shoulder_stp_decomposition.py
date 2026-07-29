from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]


def main():
    columns = [
        "session_pitch",
        "time",
        "fp_poi_time",
        "BR_time",
        "thorax_dist_seg_pwr",
        "upper_arm_prox_seg_pwr",
        "shoulder_energy_transfer_stp",
        "shoulder_energy_transfer_jfp",
        "shoulder_energy_generated",
    ]
    data = pd.read_csv(ROOT / "data" / "full_sig" / "energy_flow.csv", usecols=columns)
    data = data.dropna(subset=columns)

    # The exported *_seg_pwr fields are total segment power (SP = JFP + STP).
    # shoulder_energy_transfer_jfp is positive into the upper arm and negative
    # into the thorax, so subtract it from SP to recover each torque-power side.
    data["thorax_stp"] = (
        data["thorax_dist_seg_pwr"] + data["shoulder_energy_transfer_jfp"]
    )
    data["upper_arm_stp"] = (
        data["upper_arm_prox_seg_pwr"] - data["shoulder_energy_transfer_jfp"]
    )

    thorax = data["thorax_stp"]
    upper_arm = data["upper_arm_stp"]
    transfer_magnitude = np.minimum(thorax.abs(), upper_arm.abs())
    reconstructed = np.select(
        [
            (thorax < 0) & (upper_arm > 0),
            (thorax > 0) & (upper_arm < 0),
        ],
        [transfer_magnitude, -transfer_magnitude],
        default=0.0,
    )
    official = data["shoulder_energy_transfer_stp"]
    error = official - reconstructed

    print("Full-signal formula validation")
    print(f"Rows: {len(data)}; pitches: {data['session_pitch'].nunique()}")
    print(
        "STP reconstruction vs official: "
        f"r={official.corr(pd.Series(reconstructed, index=data.index)):.12f}, "
        f"MAE={error.abs().mean():.9f} W, "
        f"max_abs_error={error.abs().max():.9f} W"
    )
    generation_error = (
        data["shoulder_energy_generated"] - thorax - upper_arm
    ).abs()
    print(
        "Generation identity (thorax STP + upper-arm STP): "
        f"MAE={generation_error.mean():.9f} W, "
        f"max_abs_error={generation_error.max():.9f} W"
    )

    window = data[
        (data["time"] >= data["fp_poi_time"])
        & (data["time"] <= data["BR_time"])
    ].copy()
    active = window[window["thorax_stp"] * window["upper_arm_stp"] < 0].copy()
    active["thorax_smaller"] = (
        active["thorax_stp"].abs() < active["upper_arm_stp"].abs()
    )
    active["upper_arm_smaller"] = (
        active["upper_arm_stp"].abs() < active["thorax_stp"].abs()
    )
    active["transfer_magnitude"] = np.minimum(
        active["thorax_stp"].abs(),
        active["upper_arm_stp"].abs(),
    )
    active["thorax_to_arm"] = (
        (active["thorax_stp"] < 0) & (active["upper_arm_stp"] > 0)
    )
    active["thorax_limited_power"] = active["transfer_magnitude"].where(
        active["thorax_smaller"]
    )
    active["upper_arm_limited_power"] = active["transfer_magnitude"].where(
        active["upper_arm_smaller"]
    )

    per_pitch = active.groupby("session_pitch").agg(
        thorax_smaller_proportion=("thorax_smaller", "mean"),
        upper_arm_smaller_proportion=("upper_arm_smaller", "mean"),
        thorax_limited_mean_power=("thorax_limited_power", "mean"),
        upper_arm_limited_mean_power=("upper_arm_limited_power", "mean"),
        thorax_limited_power_sum=("thorax_limited_power", "sum"),
        upper_arm_limited_power_sum=("upper_arm_limited_power", "sum"),
    )
    arm_weighted = active.loc[
        active["upper_arm_smaller"], "transfer_magnitude"
    ].sum() / active["transfer_magnitude"].sum()

    print("\nFP (fp_poi_time) to BR bottleneck summary")
    print(
        f"All frames: {len(window)}; active STP-transfer frames: {len(active)} "
        f"({len(active) / len(window):.3%}); pitches: {active['session_pitch'].nunique()}"
    )
    print(
        "Frame weighted: "
        f"thorax smaller={active['thorax_smaller'].mean():.3%}, "
        f"upper arm smaller={active['upper_arm_smaller'].mean():.3%}"
    )
    print(
        "Pitch-mean proportion: "
        f"thorax smaller={per_pitch['thorax_smaller_proportion'].mean():.3%}, "
        f"upper arm smaller={per_pitch['upper_arm_smaller_proportion'].mean():.3%}"
    )
    print(
        "Pitch majority: "
        f"thorax={int((per_pitch['thorax_smaller_proportion'] > per_pitch['upper_arm_smaller_proportion']).sum())}, "
        f"upper arm={int((per_pitch['upper_arm_smaller_proportion'] > per_pitch['thorax_smaller_proportion']).sum())}, "
        f"ties={int((per_pitch['upper_arm_smaller_proportion'] == per_pitch['thorax_smaller_proportion']).sum())}"
    )
    print(
        "Transfer-magnitude weighted: "
        f"thorax smaller={1 - arm_weighted:.3%}, "
        f"upper arm smaller={arm_weighted:.3%}"
    )
    thorax_limited = active.loc[
        active["thorax_smaller"], "transfer_magnitude"
    ]
    arm_limited = active.loc[
        active["upper_arm_smaller"], "transfer_magnitude"
    ]
    print(
        "Conditional transfer power when limiting: "
        f"thorax mean={thorax_limited.mean():.3f} W, "
        f"median={thorax_limited.median():.3f} W; "
        f"upper arm mean={arm_limited.mean():.3f} W, "
        f"median={arm_limited.median():.3f} W"
    )
    print(
        "Upper-arm/thorax conditional mean-power ratio: "
        f"{arm_limited.mean() / thorax_limited.mean():.4f}"
    )
    print(f"Direction thorax -> upper arm: {active['thorax_to_arm'].mean():.3%}")

    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg", "pitch_speed_mph"],
    )
    velos = pd.read_csv(
        ROOT / "data" / "full_sig" / "joint_velos.csv",
        usecols=["session_pitch", "time", "torso_velo_z", "fp_poi_time", "BR_time"],
    )
    deceleration_rows = []
    for session_pitch, group in velos.groupby("session_pitch", sort=False):
        group = group[
            (group["time"] >= group["fp_poi_time"])
            & (group["time"] <= group["BR_time"])
        ].dropna(subset=["torso_velo_z"])
        if len(group) < 2:
            continue
        omega = group.sort_values("time")["torso_velo_z"].abs().to_numpy(float)
        peak_index = int(np.argmax(omega))
        deceleration_rows.append(
            {
                "session_pitch": session_pitch,
                "delta_omega_sq": omega[peak_index] ** 2 - omega[-1] ** 2,
            }
        )
    pitch_summary = (
        per_pitch.reset_index()
        .merge(metadata, on="session_pitch", validate="one_to_one")
        .merge(
            pd.DataFrame(deceleration_rows),
            on="session_pitch",
            validate="one_to_one",
        )
    )
    athlete = pitch_summary.groupby("session", as_index=False).agg(
        upper_arm_smaller_proportion=("upper_arm_smaller_proportion", "mean"),
        thorax_limited_mean_power=("thorax_limited_mean_power", "mean"),
        upper_arm_limited_mean_power=("upper_arm_limited_mean_power", "mean"),
        thorax_limited_power_sum=("thorax_limited_power_sum", "mean"),
        upper_arm_limited_power_sum=("upper_arm_limited_power_sum", "mean"),
        delta_omega_sq=("delta_omega_sq", "mean"),
        pitch_speed_mph=("pitch_speed_mph", "mean"),
        session_mass_kg=("session_mass_kg", "mean"),
    )
    athlete["total_limited_power_sum"] = (
        athlete["thorax_limited_power_sum"]
        + athlete["upper_arm_limited_power_sum"]
    )
    paired_power = stats.ttest_rel(
        athlete["upper_arm_limited_mean_power"],
        athlete["thorax_limited_mean_power"],
    )
    print(
        "Athlete-level paired conditional power: "
        f"thorax={athlete['thorax_limited_mean_power'].mean():.3f} W, "
        f"upper arm={athlete['upper_arm_limited_mean_power'].mean():.3f} W, "
        f"paired p={paired_power.pvalue:.4g}"
    )
    for power_variable in [
        "thorax_limited_mean_power",
        "upper_arm_limited_mean_power",
        "thorax_limited_power_sum",
        "upper_arm_limited_power_sum",
        "total_limited_power_sum",
        "upper_arm_smaller_proportion",
    ]:
        raw_power = stats.pearsonr(
            athlete["delta_omega_sq"],
            athlete[power_variable],
        )
        decel_residual = sm.OLS(
            athlete["delta_omega_sq"],
            sm.add_constant(athlete["session_mass_kg"]),
        ).fit().resid
        power_residual = sm.OLS(
            athlete[power_variable],
            sm.add_constant(athlete["session_mass_kg"]),
        ).fit().resid
        partial_power = stats.pearsonr(decel_residual, power_residual)
        print(
            f"delta omega squared vs {power_variable}: "
            f"raw r={raw_power.statistic:.4f}, p={raw_power.pvalue:.4f}; "
            f"partial-mass r={partial_power.statistic:.4f}, "
            f"p={partial_power.pvalue:.4f}"
        )
    athlete["upper_arm_majority"] = (
        athlete["upper_arm_smaller_proportion"] > 0.5
    ).astype(int)
    athlete["tie"] = athlete["upper_arm_smaller_proportion"] == 0.5
    comparison = athlete[~athlete["tie"]].copy()
    arm_group = comparison.loc[
        comparison["upper_arm_majority"] == 1, "pitch_speed_mph"
    ]
    thorax_group = comparison.loc[
        comparison["upper_arm_majority"] == 0, "pitch_speed_mph"
    ]
    welch = stats.ttest_ind(arm_group, thorax_group, equal_var=False)
    adjusted = sm.OLS(
        comparison["pitch_speed_mph"],
        sm.add_constant(
            comparison[["upper_arm_majority", "session_mass_kg"]]
        ),
    ).fit()
    arm_residual = sm.OLS(
        athlete["upper_arm_smaller_proportion"],
        sm.add_constant(athlete["session_mass_kg"]),
    ).fit().resid
    speed_residual = sm.OLS(
        athlete["pitch_speed_mph"],
        sm.add_constant(athlete["session_mass_kg"]),
    ).fit().resid
    raw_continuous = stats.pearsonr(
        athlete["upper_arm_smaller_proportion"],
        athlete["pitch_speed_mph"],
    )
    partial_continuous = stats.pearsonr(arm_residual, speed_residual)

    print("\nPitch-speed comparison by athlete-level bottleneck majority")
    for group_value, label in [(0, "thorax majority"), (1, "upper-arm majority")]:
        group = comparison[comparison["upper_arm_majority"] == group_value]
        print(
            f"{label:20s} n={len(group):2d}, "
            f"speed={group['pitch_speed_mph'].mean():.3f} mph, "
            f"mass={group['session_mass_kg'].mean():.3f} kg"
        )
    print(
        "Unadjusted upper-arm minus thorax speed difference: "
        f"{arm_group.mean() - thorax_group.mean():.3f} mph, "
        f"Welch p={welch.pvalue:.4f}"
    )
    group_ci = adjusted.conf_int().loc["upper_arm_majority"]
    print(
        "Mass-adjusted upper-arm minus thorax difference: "
        f"{adjusted.params['upper_arm_majority']:.3f} mph, "
        f"p={adjusted.pvalues['upper_arm_majority']:.4f}, "
        f"95% CI=[{group_ci.iloc[0]:.3f}, {group_ci.iloc[1]:.3f}]"
    )
    print(
        "Continuous upper-arm-smaller proportion vs speed: "
        f"raw r={raw_continuous.statistic:.4f}, p={raw_continuous.pvalue:.4f}; "
        f"partial-mass r={partial_continuous.statistic:.4f}, "
        f"p={partial_continuous.pvalue:.4f}"
    )


if __name__ == "__main__":
    main()
