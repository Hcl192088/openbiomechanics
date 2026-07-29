from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import spm1d


ROOT = Path(__file__).resolve().parents[2]
NODES = np.linspace(0.0, 100.0, 101)


def residualize(values, covariate):
    design = np.column_stack([np.ones(len(covariate)), covariate])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def cluster_summary(inference):
    rows = []
    for cluster in inference.clusters:
        endpoints = np.asarray(cluster.endpoints, dtype=float)
        rows.append(
            {
                "start_pct": float(endpoints[0]),
                "end_pct": float(endpoints[1]),
                "p": float(cluster.P),
            }
        )
    return rows


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
    energy["thorax_stp"] = (
        energy["thorax_dist_seg_pwr"] + energy["shoulder_energy_transfer_jfp"]
    )
    energy["upper_arm_stp"] = (
        energy["upper_arm_prox_seg_pwr"] - energy["shoulder_energy_transfer_jfp"]
    )
    opposite = energy["thorax_stp"] * energy["upper_arm_stp"] < 0
    energy["transfer_power"] = np.where(
        opposite,
        np.minimum(energy["thorax_stp"].abs(), energy["upper_arm_stp"].abs()),
        0.0,
    )
    energy["thorax_limited_power"] = np.where(
        opposite & (energy["thorax_stp"].abs() < energy["upper_arm_stp"].abs()),
        energy["transfer_power"],
        0.0,
    )
    energy["upper_arm_limited_power"] = np.where(
        opposite & (energy["upper_arm_stp"].abs() < energy["thorax_stp"].abs()),
        energy["transfer_power"],
        0.0,
    )

    velos = pd.read_csv(
        ROOT / "data" / "full_sig" / "joint_velos.csv",
        usecols=["session_pitch", "time", "fp_poi_time", "BR_time", "torso_velo_z"],
    ).dropna()
    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )

    curve_rows = []
    metric_rows = []
    for session_pitch, group in energy.groupby("session_pitch", sort=False):
        group = group[
            (group["time"] >= group["fp_poi_time"])
            & (group["time"] <= group["BR_time"])
        ].sort_values("time")
        if len(group) < 3:
            continue
        phase = (
            100
            * (group["time"] - group["fp_poi_time"])
            / (group["BR_time"] - group["fp_poi_time"])
        ).to_numpy(float)
        row = {"session_pitch": session_pitch}
        for variable in [
            "transfer_power",
            "thorax_limited_power",
            "upper_arm_limited_power",
        ]:
            row.update(
                {
                    f"{variable}_{node}": value
                    for node, value in zip(
                        range(101),
                        np.interp(NODES, phase, group[variable].to_numpy(float)),
                    )
                }
            )
        row["mer_phase"] = float(
            100
            * (group["MER_time"].iloc[0] - group["fp_poi_time"].iloc[0])
            / (group["BR_time"].iloc[0] - group["fp_poi_time"].iloc[0])
        )
        curve_rows.append(row)

    for session_pitch, group in velos.groupby("session_pitch", sort=False):
        group = group[
            (group["time"] >= group["fp_poi_time"])
            & (group["time"] <= group["BR_time"])
        ].sort_values("time")
        if len(group) < 3:
            continue
        omega = group["torso_velo_z"].abs().to_numpy(float)
        peak_index = int(np.argmax(omega))
        metric_rows.append(
            {
                "session_pitch": session_pitch,
                "delta_omega_sq": omega[peak_index] ** 2 - omega[-1] ** 2,
            }
        )

    pitch = (
        pd.DataFrame(curve_rows)
        .merge(pd.DataFrame(metric_rows), on="session_pitch", validate="one_to_one")
        .merge(metadata, on="session_pitch", validate="one_to_one")
    )
    athlete = pitch.groupby("session", as_index=False).mean(numeric_only=True)
    x_decel = athlete["delta_omega_sq"].to_numpy(float)
    x_mass = athlete["session_mass_kg"].to_numpy(float)
    design = np.column_stack(
        [
            np.ones(len(athlete)),
            stats.zscore(x_decel),
            stats.zscore(x_mass),
        ]
    )
    contrast = np.array([0.0, 1.0, 0.0])

    print(
        f"Pitch rows={len(pitch)}; athlete rows={len(athlete)}; "
        f"MER median={athlete['mer_phase'].median():.3f}%, "
        f"IQR=[{athlete['mer_phase'].quantile(.25):.3f}, "
        f"{athlete['mer_phase'].quantile(.75):.3f}]%"
    )

    results = {}
    for variable in [
        "transfer_power",
        "upper_arm_limited_power",
        "thorax_limited_power",
    ]:
        columns = [f"{variable}_{node}" for node in range(101)]
        y = athlete[columns].to_numpy(float)
        inference = spm1d.stats.glm(y, design, contrast).inference(
            alpha=0.05,
            two_tailed=True,
            interp=True,
        )
        decel_residual = residualize(x_decel, x_mass)
        y_residual = np.column_stack(
            [residualize(y[:, node], x_mass) for node in range(y.shape[1])]
        )
        partial_r = np.array(
            [
                stats.pearsonr(decel_residual, y_residual[:, node]).statistic
                for node in range(y.shape[1])
            ]
        )
        clusters = cluster_summary(inference)
        results[variable] = {
            "partial_r": partial_r,
            "inference": inference,
            "clusters": clusters,
        }
        peak_index = int(np.nanargmax(np.abs(partial_r)))
        print(
            f"\n{variable}: peak |partial r|={partial_r[peak_index]:.4f} "
            f"at {NODES[peak_index]:.1f}%"
        )
        if clusters:
            for cluster in clusters:
                cluster_mask = (
                    (NODES >= cluster["start_pct"])
                    & (NODES <= cluster["end_pct"])
                )
                print(
                    f"  SPM cluster {cluster['start_pct']:.2f}-"
                    f"{cluster['end_pct']:.2f}%, p={cluster['p']:.6f}, "
                    f"mean partial r={partial_r[cluster_mask].mean():.4f}"
                )
        else:
            print("  No SPM-significant cluster")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = {
        "transfer_power": "Total STP transfer power",
        "upper_arm_limited_power": "Upper-arm-limited STP power",
        "thorax_limited_power": "Thorax-limited STP power",
    }
    mer_median = athlete["mer_phase"].median()
    mer_q1 = athlete["mer_phase"].quantile(0.25)
    mer_q3 = athlete["mer_phase"].quantile(0.75)
    for axis, variable in zip(axes, labels):
        result = results[variable]
        axis.plot(NODES, result["partial_r"], color="#2457A7", linewidth=2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvspan(mer_q1, mer_q3, color="#E8A838", alpha=0.18)
        axis.axvline(mer_median, color="#E08A00", linestyle="--", linewidth=1.2)
        for cluster in result["clusters"]:
            axis.axvspan(
                cluster["start_pct"],
                cluster["end_pct"],
                color="#D44A4A",
                alpha=0.18,
            )
        axis.set_ylabel("Partial r")
        axis.set_title(labels[variable])
        axis.set_ylim(-0.55, 0.55)
    axes[-1].set_xlabel("Normalized FP–BR phase (%)")
    fig.suptitle(
        "Trunk rotational-energy-drop proxy vs shoulder STP power\n"
        "Athlete-level partial correlations controlling body mass"
    )
    fig.tight_layout()
    output = ROOT / "imgs" / "trunk_deceleration_stp_spm.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
