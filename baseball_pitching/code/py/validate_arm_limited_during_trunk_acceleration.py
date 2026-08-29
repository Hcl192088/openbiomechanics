"""Test whether arm-limited shoulder STP occurs while the trunk is accelerating."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("arm_limited_trunk_acceleration_outputs")


def add_acceleration(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time").copy()
    t = group.time.to_numpy(float)
    speed = np.linalg.norm(
        group[[f"torso_velo_{axis}" for axis in "xyz"]].to_numpy(float), axis=1
    )
    group["torso_speed_deg_s"] = speed
    group["torso_speed_accel_raw_deg_s2"] = np.gradient(speed, t)
    smoothed = savgol_filter(speed, 11, 3, mode="interp")
    group["torso_speed_accel_sg11_deg_s2"] = np.gradient(smoothed, t)
    return group


def summarize_pitch(group: pd.DataFrame) -> pd.Series:
    fp, mer, br = (float(group[c].iloc[0]) for c in ["fp_poi_time", "MER_time", "BR_time"])
    w = group[(group.time >= fp) & (group.time <= br)].copy()
    jfp = w.shoulder_energy_transfer_jfp
    thorax_stp = w.thorax_dist_seg_pwr + jfp
    arm_stp = w.upper_arm_prox_seg_pwr - jfp
    arm_limited = (
        (thorax_stp < 0) & (arm_stp > 0)
        & (arm_stp.abs() < thorax_stp.abs())
        & (w.shoulder_energy_transfer_stp > 0)
    )
    a = w[arm_limited].copy()
    if a.empty:
        empty = {"eligible_arm_limited": False}
        metric_names = [
            "frame_n", "torso_accelerating_fraction",
            "torso_accelerating_raw_fraction", "thorax_net_positive_fraction",
            "both_accel_and_net_positive_fraction",
            "stp_weighted_accelerating_fraction",
            "stp_weighted_net_positive_fraction", "stp_weighted_both_fraction",
            "mean_thorax_shoulder_stp_w", "mean_arm_shoulder_stp_w",
            "mean_thorax_net_seg_pwr_w", "mean_torso_accel_deg_s2",
        ]
        empty.update({f"{label}_{name}": np.nan for label in ["all", "pre_mer"] for name in metric_names})
        return pd.Series(empty)
    eligible = {"eligible_arm_limited": True}
    a["thorax_shoulder_stp_w"] = thorax_stp.loc[a.index]
    a["arm_shoulder_stp_w"] = arm_stp.loc[a.index]
    a["thorax_net_seg_pwr_w"] = (
        a.thorax_prox_seg_pwr + a.thorax_dist_seg_pwr + a.thorax_dist_glove_seg_pwr
    )
    pre_mer = a.time <= mer
    rows = {}
    for label, part in [("all", a), ("pre_mer", a[pre_mer])]:
        if part.empty:
            continue
        accel = part.torso_speed_accel_sg11_deg_s2 > 0
        raw_accel = part.torso_speed_accel_raw_deg_s2 > 0
        net_positive = part.thorax_net_seg_pwr_w > 0
        weights = part.shoulder_energy_transfer_stp.clip(lower=0)
        weight_sum = weights.sum()
        rows.update({
            f"{label}_frame_n": len(part),
            f"{label}_torso_accelerating_fraction": accel.mean(),
            f"{label}_torso_accelerating_raw_fraction": raw_accel.mean(),
            f"{label}_thorax_net_positive_fraction": net_positive.mean(),
            f"{label}_both_accel_and_net_positive_fraction": (accel & net_positive).mean(),
            f"{label}_stp_weighted_accelerating_fraction": weights[accel].sum() / weight_sum,
            f"{label}_stp_weighted_net_positive_fraction": weights[net_positive].sum() / weight_sum,
            f"{label}_stp_weighted_both_fraction": weights[accel & net_positive].sum() / weight_sum,
            f"{label}_mean_thorax_shoulder_stp_w": part.thorax_shoulder_stp_w.mean(),
            f"{label}_mean_arm_shoulder_stp_w": part.arm_shoulder_stp_w.mean(),
            f"{label}_mean_thorax_net_seg_pwr_w": part.thorax_net_seg_pwr_w.mean(),
            f"{label}_mean_torso_accel_deg_s2": part.torso_speed_accel_sg11_deg_s2.mean(),
        })
    return pd.Series({**eligible, **rows})


def main() -> None:
    energy_cols = [
        "session_pitch", "time", "fp_poi_time", "MER_time", "BR_time",
        "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
        "upper_arm_prox_seg_pwr", "thorax_dist_seg_pwr",
        "thorax_prox_seg_pwr", "thorax_dist_glove_seg_pwr",
    ]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=energy_cols).dropna()
    velocity = pd.read_csv(
        ROOT / "data/full_sig/joint_velos.csv",
        usecols=["session_pitch", "time", *[f"torso_velo_{a}" for a in "xyz"]],
    ).dropna()
    data = energy.merge(velocity, on=["session_pitch", "time"], validate="one_to_one")
    if len(data) != len(energy):
        raise ValueError("Exact time-key merge lost rows; no fallback permitted")
    data = data.groupby("session_pitch", group_keys=True).apply(
        add_acceleration, include_groups=False
    ).reset_index(level=0)
    pitch = data.groupby("session_pitch", sort=False).apply(
        summarize_pitch, include_groups=False
    ).reset_index()
    meta = pd.read_csv(
        ROOT / "data/metadata.csv", usecols=["session_pitch", "session"]
    )
    pitch = pitch.merge(meta, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411 or pitch.session.nunique() != 100:
        raise ValueError("Expected 411 pitches and 100 sessions")
    numeric = [c for c in pitch.columns if c not in ["session_pitch", "session"]]
    eligible_pitch = pitch[pitch.eligible_arm_limited].copy()
    numeric = [c for c in numeric if c != "eligible_arm_limited"]
    athlete = eligible_pitch.groupby("session", as_index=False)[numeric].mean()
    OUT.mkdir(exist_ok=True)
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    athlete.to_csv(OUT / "per_pitcher.csv", index=False)
    summary = athlete[numeric].agg(["mean", "median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).T
    summary.columns = ["mean", "median", "q1", "q3"]
    summary.to_csv(OUT / "pitcher_summary.csv")
    key = [
        "pre_mer_torso_accelerating_fraction",
        "pre_mer_torso_accelerating_raw_fraction",
        "pre_mer_thorax_net_positive_fraction",
        "pre_mer_both_accel_and_net_positive_fraction",
        "pre_mer_stp_weighted_accelerating_fraction",
        "pre_mer_stp_weighted_net_positive_fraction",
        "pre_mer_stp_weighted_both_fraction",
        "pre_mer_mean_thorax_shoulder_stp_w",
        "pre_mer_mean_arm_shoulder_stp_w",
        "pre_mer_mean_thorax_net_seg_pwr_w",
        "pre_mer_mean_torso_accel_deg_s2",
    ]
    print(
        f"coverage: rows={len(data)}, all_pitches={len(pitch)}, "
        f"eligible_pitches={len(eligible_pitch)}, eligible_pitchers={len(athlete)}"
    )
    print(summary.loc[key].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
