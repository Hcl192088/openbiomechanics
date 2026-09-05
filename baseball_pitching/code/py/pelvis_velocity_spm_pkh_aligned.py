from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spm1d


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "baseball_pitching" / "data"
IMG_DIR = ROOT / "baseball_pitching" / "imgs"
OUT_DIR = ROOT / "baseball_pitching" / "code" / "py" / "pelvis_velocity_spm_pkh_outputs"

VELO_PATH = DATA_DIR / "full_sig" / "joint_velos.csv"
POI_PATH = DATA_DIR / "poi" / "poi_metrics.csv"
VALUE_COL = "pelvis_velo_z"
SPEED_COL = "pitch_speed_mph"
SAMPLE_RATE_HZ = 360
END_FRAME = 260


def unique_value(group: pd.DataFrame, column: str, session_pitch: str) -> float:
    values = group[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{session_pitch}: expected one {column}, found {len(values)}")
    return float(values[0])


def extract_matrix(velos: pd.DataFrame, speeds: pd.DataFrame) -> pd.DataFrame:
    speed_map = dict(zip(speeds["session_pitch"].astype(str), speeds[SPEED_COL]))
    rows: list[dict[str, float | str | int]] = []
    for session_pitch, group in velos.groupby("session_pitch", sort=False):
        session_pitch = str(session_pitch)
        if session_pitch not in speed_map:
            raise ValueError(f"{session_pitch}: missing pitch speed")
        ordered = group[["time", "pkh_time", "fp_poi_time", VALUE_COL]].dropna().sort_values("time")
        pkh = unique_value(ordered, "pkh_time", session_pitch)
        fp = unique_value(ordered, "fp_poi_time", session_pitch)
        if not pkh < fp:
            raise ValueError(f"{session_pitch}: expected pkh_time < fp_poi_time")
        pkh_idx = int((ordered["time"] - pkh).abs().idxmin())
        pkh_position = int(ordered.index.get_loc(pkh_idx))
        values = ordered[VALUE_COL].to_numpy(dtype=float)
        if pkh_position + END_FRAME >= len(values):
            raise ValueError(f"{session_pitch}: fewer than {END_FRAME + 1} samples after PKH")
        window = values[pkh_position : pkh_position + END_FRAME + 1]
        row: dict[str, float | str | int] = {
            "session_pitch": session_pitch,
            "pitcher": session_pitch.split("_")[0],
            SPEED_COL: float(speed_map[session_pitch]),
            "fp_frame_from_pkh": (fp - pkh) * SAMPLE_RATE_HZ,
        }
        row.update({f"frame_{frame}": value for frame, value in enumerate(window)})
        rows.append(row)
    return pd.DataFrame(rows)


def group_masks(data: pd.DataFrame, mode: str) -> tuple[pd.Series, pd.Series, str]:
    if mode == "90mph":
        return data[SPEED_COL] > 90, data[SPEED_COL] <= 90, ">90 mph vs <=90 mph"
    if mode == "quartile":
        q75 = data[SPEED_COL].quantile(0.75)
        q25 = data[SPEED_COL].quantile(0.25)
        return (
            data[SPEED_COL] >= q75,
            data[SPEED_COL] <= q25,
            f"top quartile >= {q75:.1f} vs bottom <= {q25:.1f} mph",
        )
    if mode == "p55_p45":
        q55 = data[SPEED_COL].quantile(0.55)
        q45 = data[SPEED_COL].quantile(0.45)
        return (
            data[SPEED_COL] >= q55,
            data[SPEED_COL] <= q45,
            f">=55th pct {q55:.1f} vs <=45th pct {q45:.1f} mph",
        )
    raise ValueError(f"unknown mode: {mode}")


def analyze_mode(data: pd.DataFrame, mode: str) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    high_mask, low_mask, label = group_masks(data, mode)
    columns = [f"frame_{frame}" for frame in range(END_FRAME + 1)]
    high = data.loc[high_mask, columns].to_numpy(dtype=float)
    low = data.loc[low_mask, columns].to_numpy(dtype=float)
    inference = spm1d.stats.ttest2(high, low, equal_var=False).inference(
        alpha=0.05,
        two_tailed=True,
        interp=True,
    )

    high_mean = high.mean(axis=0)
    low_mean = low.mean(axis=0)
    high_sem = high.std(axis=0, ddof=1) / np.sqrt(len(high))
    low_sem = low.std(axis=0, ddof=1) / np.sqrt(len(low))
    x_seconds = np.arange(END_FRAME + 1) / SAMPLE_RATE_HZ

    high_fp = data.loc[high_mask, "fp_frame_from_pkh"] / SAMPLE_RATE_HZ
    low_fp = data.loc[low_mask, "fp_frame_from_pkh"] / SAMPLE_RATE_HZ
    high_fp_q25, high_fp_median, high_fp_q75 = high_fp.quantile([0.25, 0.5, 0.75])
    low_fp_q25, low_fp_median, low_fp_q75 = low_fp.quantile([0.25, 0.5, 0.75])

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = IMG_DIR / f"pelvis_velocity_spm_pkh_{mode}.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
    axes[0].plot(x_seconds, high_mean, color="red", linewidth=2, label="High speed")
    axes[0].fill_between(x_seconds, high_mean - high_sem, high_mean + high_sem, color="red", alpha=0.2)
    axes[0].plot(x_seconds, low_mean, color="black", linewidth=2, label="Low speed")
    axes[0].fill_between(x_seconds, low_mean - low_sem, low_mean + low_sem, color="gray", alpha=0.25)
    axes[0].axhline(0, color="0.6", linestyle=":", linewidth=1)
    axes[0].axvspan(high_fp_q25, high_fp_q75, color="red", alpha=0.08)
    axes[0].axvline(high_fp_median, color="red", linestyle="--", linewidth=1.5, label="High FP median")
    axes[0].axvspan(low_fp_q25, low_fp_q75, color="black", alpha=0.06)
    axes[0].axvline(low_fp_median, color="black", linestyle="--", linewidth=1.5, label="Low FP median")
    axes[0].set_title(label, fontsize=10)
    axes[0].set_xlabel("Time from PKH (s)")
    axes[0].set_ylabel("Pelvis rotation velocity (deg/s)")
    axes[0].legend(fontsize=7, loc="upper left")

    inference.plot(ax=axes[1])
    axes[1].set_xlabel("Time from PKH (s)")
    tick_frames = np.arange(0, END_FRAME + 1, 40)
    axes[1].set_xticks(tick_frames, [f"{frame / SAMPLE_RATE_HZ:.2f}" for frame in tick_frames])
    axes[1].set_title("SPM{t} high - low", fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    cluster_rows: list[dict[str, float | int | str]] = []
    for cluster in inference.clusters:
        start_frame, end_frame = cluster.endpoints
        first_complete_frame = max(0, int(np.ceil(start_frame)))
        last_complete_frame = min(END_FRAME, int(np.floor(end_frame)))
        mean_difference = float(
            np.mean(high[:, first_complete_frame : last_complete_frame + 1])
            - np.mean(low[:, first_complete_frame : last_complete_frame + 1])
        )
        cluster_rows.append(
            {
                "mode": mode,
                "start_frame_from_pkh": float(start_frame),
                "end_frame_from_pkh": float(end_frame),
                "start_seconds_from_pkh": float(start_frame / SAMPLE_RATE_HZ),
                "end_seconds_from_pkh": float(end_frame / SAMPLE_RATE_HZ),
                "mean_velocity_difference_high_minus_low": mean_difference,
                "cluster_p": float(cluster.P),
            }
        )

    summary = {
        "mode": mode,
        "comparison": label,
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "high_speed_mean": float(data.loc[high_mask, SPEED_COL].mean()),
        "low_speed_mean": float(data.loc[low_mask, SPEED_COL].mean()),
        "high_fp_median_seconds_from_pkh": float(high_fp_median),
        "high_fp_q25_seconds_from_pkh": float(high_fp_q25),
        "high_fp_q75_seconds_from_pkh": float(high_fp_q75),
        "low_fp_median_seconds_from_pkh": float(low_fp_median),
        "low_fp_q25_seconds_from_pkh": float(low_fp_q25),
        "low_fp_q75_seconds_from_pkh": float(low_fp_q75),
        "zstar": float(inference.zstar),
        "n_clusters": int(len(inference.clusters)),
        "min_cluster_p": min((float(c.P) for c in inference.clusters), default=np.nan),
        "figure": str(figure_path.relative_to(ROOT)),
    }
    return summary, cluster_rows


def pitcher_level_sensitivity(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [f"frame_{frame}" for frame in range(END_FRAME + 1)]
    aggregations = {SPEED_COL: "mean", "fp_frame_from_pkh": "mean"}
    aggregations.update({column: "mean" for column in columns})
    pitcher_means = matrix.groupby("pitcher", as_index=False).agg(aggregations)

    summaries: list[dict[str, float | int | str]] = []
    clusters: list[dict[str, float | int | str]] = []
    for mode in ["90mph", "quartile", "p55_p45"]:
        high_mask, low_mask, label = group_masks(pitcher_means, mode)
        high = pitcher_means.loc[high_mask, columns].to_numpy(dtype=float)
        low = pitcher_means.loc[low_mask, columns].to_numpy(dtype=float)
        inference = spm1d.stats.ttest2(high, low, equal_var=False).inference(
            alpha=0.05,
            two_tailed=True,
            interp=True,
        )
        summaries.append(
            {
                "mode": mode,
                "comparison": label,
                "n_high_pitchers": len(high),
                "n_low_pitchers": len(low),
                "n_clusters": len(inference.clusters),
                "min_cluster_p": min((float(c.P) for c in inference.clusters), default=np.nan),
            }
        )
        for cluster in inference.clusters:
            start_frame, end_frame = cluster.endpoints
            first_complete_frame = max(0, int(np.ceil(start_frame)))
            last_complete_frame = min(END_FRAME, int(np.floor(end_frame)))
            clusters.append(
                {
                    "mode": mode,
                    "start_seconds_from_pkh": float(start_frame / SAMPLE_RATE_HZ),
                    "end_seconds_from_pkh": float(end_frame / SAMPLE_RATE_HZ),
                    "mean_velocity_difference_high_minus_low": float(
                        np.mean(high[:, first_complete_frame : last_complete_frame + 1])
                        - np.mean(low[:, first_complete_frame : last_complete_frame + 1])
                    ),
                    "cluster_p": float(cluster.P),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(clusters)


def main() -> None:
    velos = pd.read_csv(
        VELO_PATH,
        usecols=["session_pitch", "time", "pkh_time", "fp_poi_time", VALUE_COL],
    )
    speeds = pd.read_csv(POI_PATH, usecols=["session_pitch", SPEED_COL]).dropna().drop_duplicates("session_pitch")
    matrix = extract_matrix(velos, speeds)
    if len(matrix) != 411:
        raise ValueError(f"expected 411 pitches, analyzed {len(matrix)}")

    summaries: list[dict[str, float | int | str]] = []
    clusters: list[dict[str, float | int | str]] = []
    for mode in ["90mph", "quartile", "p55_p45"]:
        summary, mode_clusters = analyze_mode(matrix, mode)
        summaries.append(summary)
        clusters.extend(mode_clusters)

    summary_df = pd.DataFrame(summaries)
    cluster_df = pd.DataFrame(clusters)
    pitcher_summary_df, pitcher_cluster_df = pitcher_level_sensitivity(matrix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUT_DIR / "pelvis_velocity_pkh_frame_windows.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "pelvis_velocity_spm_pkh_summary.csv", index=False, encoding="utf-8-sig")
    cluster_df.to_csv(OUT_DIR / "pelvis_velocity_spm_pkh_clusters.csv", index=False, encoding="utf-8-sig")
    pitcher_summary_df.to_csv(
        OUT_DIR / "pelvis_velocity_spm_pkh_pitcher_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    pitcher_cluster_df.to_csv(
        OUT_DIR / "pelvis_velocity_spm_pkh_pitcher_clusters.csv", index=False, encoding="utf-8-sig"
    )

    report = [
        "# PKH-aligned pelvis rotation velocity SPM",
        "",
        f"Window: PKH +0 to +{END_FRAME} frames at {SAMPLE_RATE_HZ} Hz (0 to {END_FRAME / SAMPLE_RATE_HZ:.3f} s).",
        "FP timing uses `fp_poi_time` only. All 411 pitches have a complete window.",
        "",
        "## Summary",
        summary_df.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## SPM clusters",
        cluster_df.to_csv(index=False, lineterminator="\n").strip()
        if len(cluster_df)
        else "No significant clusters.",
        "",
        "## Pitcher-mean sensitivity",
        "Each pitcher is averaged first, then grouped by pitcher mean velocity. This avoids treating repeated pitches as independent subjects.",
        pitcher_summary_df.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Pitcher-mean SPM clusters",
        pitcher_cluster_df.to_csv(index=False, lineterminator="\n").strip()
        if len(pitcher_cluster_df)
        else "No significant clusters.",
        "",
    ]
    (OUT_DIR / "pelvis_velocity_spm_pkh_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print("\nClusters")
    print(cluster_df.to_string(index=False) if len(cluster_df) else "None")
    print("\nPitcher-mean sensitivity")
    print(pitcher_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
