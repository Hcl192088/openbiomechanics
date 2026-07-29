"""Replicate the visible predictors in Sigman's pelvis-velocity ranking.

The source screenshot reports pitch-level Pearson correlations. Event values
here use the nearest sampled frame. Project FP is fp_poi_time; BR is BR_time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RNG_SEED = 20260729
N_BOOT = 2_000


def nearest_event_values(
    data: pd.DataFrame, value_columns: list[str], event_column: str
) -> pd.DataFrame:
    valid = data.dropna(subset=["session_pitch", "time", event_column]).copy()
    valid["_distance"] = (valid["time"] - valid[event_column]).abs()
    rows = valid.loc[valid.groupby("session_pitch")["_distance"].idxmin()]
    return rows.set_index("session_pitch")[value_columns]


def participant_id(session_pitch: pd.Series) -> pd.Series:
    return session_pitch.astype(str).str.split("_", n=1).str[0]


def cluster_bootstrap_ci(
    frame: pd.DataFrame, x: str, y: str, cluster: str
) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    groups = [
        (group[x].to_numpy(), group[y].to_numpy())
        for _, group in frame.groupby(cluster)
    ]
    estimates = np.empty(N_BOOT)
    for index in range(N_BOOT):
        sampled = rng.integers(0, len(groups), size=len(groups))
        sample_x = np.concatenate([groups[group_index][0] for group_index in sampled])
        sample_y = np.concatenate([groups[group_index][1] for group_index in sampled])
        estimates[index] = np.corrcoef(sample_x, sample_y)[0, 1]
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def main() -> None:
    angle_columns = [
        "session_pitch",
        "time",
        "fp_poi_time",
        "BR_time",
        "pelvis_angle_z",
        "rear_hip_angle_x",
        "rear_hip_angle_y",
        "rear_hip_angle_z",
    ]
    angles = pd.read_csv(DATA / "full_sig" / "joint_angles.csv", usecols=angle_columns)
    poi = pd.read_csv(
        DATA / "poi" / "poi_metrics.csv",
        usecols=[
            "session_pitch",
            "max_pelvis_rotational_velo",
            "stride_length",
            "lead_knee_extension_from_fp_to_br",
        ],
    ).set_index("session_pitch")

    fp = nearest_event_values(
        angles,
        [
            "pelvis_angle_z",
            "rear_hip_angle_x",
            "rear_hip_angle_y",
            "rear_hip_angle_z",
        ],
        "fp_poi_time",
    ).add_suffix("_fp")
    br = nearest_event_values(
        angles,
        [
            "pelvis_angle_z",
            "rear_hip_angle_x",
            "rear_hip_angle_y",
            "rear_hip_angle_z",
        ],
        "BR_time",
    ).add_suffix("_br")
    maxima = angles.groupby("session_pitch").agg(
        max_abs_rear_hip_angle_x=(
            "rear_hip_angle_x",
            lambda series: series.abs().max(),
        ),
        max_rear_hip_angle_y=("rear_hip_angle_y", "max"),
    )

    analysis = poi.join([fp, br, maxima], how="inner").reset_index()
    analysis["participant"] = participant_id(analysis["session_pitch"])
    analysis["pelvis_angle_z_fp_to_br"] = (
        analysis["pelvis_angle_z_br"] - analysis["pelvis_angle_z_fp"]
    )
    analysis["rear_hip_angle_z_fp_to_br"] = (
        analysis["rear_hip_angle_z_br"] - analysis["rear_hip_angle_z_fp"]
    )

    predictors = [
        ("Pelvis Angle Z at Ball Release", "pelvis_angle_z_br", 0.268),
        ("Max abs Rear Hip Angle X", "max_abs_rear_hip_angle_x", 0.257),
        ("Pelvis Angle Z (FP to BR)", "pelvis_angle_z_fp_to_br", 0.251),
        ("Rear Hip Angle Z (FP to BR)", "rear_hip_angle_z_fp_to_br", -0.243),
        ("Stride Length", "stride_length", -0.222),
        ("Rear Hip Angle Y at Ball Release", "rear_hip_angle_y_br", -0.207),
        ("Rear Hip Angle X at Foot Plant", "rear_hip_angle_x_fp", -0.189),
        (
            "Lead Knee Extension (FP to BR)",
            "lead_knee_extension_from_fp_to_br",
            0.188,
        ),
        ("Max Rear Hip Angle Y", "max_rear_hip_angle_y", -0.169),
    ]

    print(
        f"Analysis unit: pitch; pitches={len(analysis)}; "
        f"participants={analysis['participant'].nunique()}"
    )
    print("Outcome: max_pelvis_rotational_velo (POI)")
    print("FP: fp_poi_time; BR: BR_time; event sampling: nearest frame")
    print(f"CI: participant-cluster bootstrap, {N_BOOT:,} resamples")
    print()
    print(
        f"{'Predictor':43s} {'n':>4s} {'OBP r':>8s} "
        f"{'95% cluster CI':>21s} {'Source r':>9s} {'sign':>6s}"
    )
    for label, column, source_r in predictors:
        complete = analysis[
            [column, "max_pelvis_rotational_velo", "participant"]
        ].dropna()
        result = pearsonr(complete[column], complete["max_pelvis_rotational_velo"])
        low, high = cluster_bootstrap_ci(
            complete, column, "max_pelvis_rotational_velo", "participant"
        )
        same_sign = np.sign(result.statistic) == np.sign(source_r)
        print(
            f"{label:43s} {len(complete):4d} {result.statistic:+8.3f} "
            f"[{low:+.3f}, {high:+.3f}] {source_r:+9.3f} "
            f"{str(same_sign):>6s}"
        )

    print()
    print("Hip X+Z: not calculated; the screenshot does not define its formula.")


if __name__ == "__main__":
    main()
