from pathlib import Path

import numpy as np
import pandas as pd


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

    per_pitch = active.groupby("session_pitch").agg(
        thorax_smaller_proportion=("thorax_smaller", "mean"),
        upper_arm_smaller_proportion=("upper_arm_smaller", "mean"),
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
    print(f"Direction thorax -> upper arm: {active['thorax_to_arm'].mean():.3%}")


if __name__ == "__main__":
    main()
