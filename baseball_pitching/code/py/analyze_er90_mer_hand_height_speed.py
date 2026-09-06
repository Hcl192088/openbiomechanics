from __future__ import annotations

"""Analyse maximum throwing-hand vertical drop within ER90-to-MER and pitch speed."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "baseball_pitching" / "data"
FULL_SIG_DIR = DATA_DIR / "full_sig"
META_PATH = DATA_DIR / "metadata.csv"
POI_PATH = DATA_DIR / "poi" / "poi_metrics.csv"
OUT_DIR = ROOT / "baseball_pitching" / "code" / "py" / "er90_mer_hand_height_speed_outputs"
FIG_PATH = ROOT / "baseball_pitching" / "imgs" / "er90_mer_hand_height_drop_vs_speed.png"

ER90_DEG = 90.0
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_N = 5000

LANDMARK_COLS = [
    "session_pitch",
    "time",
    "hand_jc_z",
    "MER_time",
    "BR_time",
    "fp_poi_time",
]
ANGLE_COLS = [
    "session_pitch",
    "time",
    "shoulder_angle_z",
    "MER_time",
    "BR_time",
    "fp_poi_time",
]
META_COLS = ["session_pitch", "user", "session_height_m", "pitch_speed_mph"]
POI_COLS = ["session_pitch", "pitch_speed_mph", "max_shoulder_external_rotation"]


def read_full_signal(filename: str, usecols: list[str]) -> pd.DataFrame:
    path = FULL_SIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required active full-signal CSV: {path}")
    return pd.read_csv(path, usecols=usecols, dtype={"session_pitch": "string"})


def normalize_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["session_pitch"] = frame["session_pitch"].astype("string")
    if frame["session_pitch"].isna().any():
        raise ValueError("session_pitch contains missing values")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def assert_unique_key(frame: pd.DataFrame, label: str) -> None:
    duplicate = frame["session_pitch"].duplicated()
    if duplicate.any():
        examples = frame.loc[duplicate, "session_pitch"].astype(str).head(5).tolist()
        raise ValueError(f"{label} has duplicate session_pitch rows, examples={examples}")


def event_value(group: pd.DataFrame, column: str, pitch: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{pitch} has {len(values)} unique non-null {column} values")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError(f"{pitch} has non-finite {column}")
    return value


def interpolate_at(group: pd.DataFrame, column: str, time: float) -> float:
    data = group[["time", column]].dropna().sort_values("time")
    if len(data) < 2:
        return np.nan
    times = data["time"].to_numpy(dtype=float)
    values = data[column].to_numpy(dtype=float)
    if time < times[0] or time > times[-1]:
        return np.nan
    return float(np.interp(time, times, values))


def maximum_vertical_drawdown(
    group: pd.DataFrame, column: str, start: float, end: float
) -> tuple[float, float, float]:
    data = group[["time", column]].dropna().sort_values("time")
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    all_times = data["time"].to_numpy(dtype=float)
    all_values = data[column].to_numpy(dtype=float)
    if start < all_times[0] or end > all_times[-1] or end <= start:
        return np.nan, np.nan, np.nan

    interior = data.loc[(data["time"] > start) & (data["time"] < end)]
    times = np.concatenate(([start], interior["time"].to_numpy(dtype=float), [end]))
    values = np.concatenate(
        (
            [np.interp(start, all_times, all_values)],
            interior[column].to_numpy(dtype=float),
            [np.interp(end, all_times, all_values)],
        )
    )
    running_high = np.maximum.accumulate(values)
    drawdown = running_high - values
    end_index = int(np.argmax(drawdown))
    start_index = int(np.argmax(values[: end_index + 1]))
    return float(drawdown[end_index]), float(times[start_index]), float(times[end_index])


def find_ascending_er90_crossings(group: pd.DataFrame, start: float, end: float) -> list[float]:
    data = (
        group.loc[(group["time"] >= start) & (group["time"] <= end), ["time", "shoulder_angle_z"]]
        .dropna()
        .sort_values("time")
    )
    if len(data) < 2:
        return []

    times = data["time"].to_numpy(dtype=float)
    values = data["shoulder_angle_z"].to_numpy(dtype=float)
    crossings: list[float] = []
    for index, value in enumerate(values):
        if np.isclose(value, ER90_DEG, atol=1e-8) and (index == 0 or values[index - 1] < ER90_DEG):
            crossings.append(float(times[index]))
    for index in np.flatnonzero((values[:-1] < ER90_DEG) & (values[1:] > ER90_DEG)):
        left_time = times[index]
        right_time = times[index + 1]
        left_value = values[index]
        right_value = values[index + 1]
        crossings.append(
            float(left_time + (ER90_DEG - left_value) * (right_time - left_time) / (right_value - left_value))
        )

    crossings.sort()
    unique: list[float] = []
    for crossing in crossings:
        if not unique or abs(crossing - unique[-1]) > 1e-9:
            unique.append(crossing)
    return unique


def validate_speed_sources(metadata: pd.DataFrame, poi: pd.DataFrame) -> None:
    assert_unique_key(metadata, "metadata.csv")
    assert_unique_key(poi, "poi_metrics.csv")
    merged = metadata[["session_pitch", "pitch_speed_mph"]].merge(
        poi[["session_pitch", "pitch_speed_mph"]],
        on="session_pitch",
        how="outer",
        suffixes=("_metadata", "_poi"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("metadata.csv and poi_metrics.csv do not contain the same pitch keys")
    mismatch = (
        pd.to_numeric(merged["pitch_speed_mph_metadata"], errors="coerce")
        - pd.to_numeric(merged["pitch_speed_mph_poi"], errors="coerce")
    ).abs()
    if mismatch.isna().any() or (mismatch > 1e-9).any():
        raise ValueError("metadata.csv and poi_metrics.csv pitch speeds disagree")


def validate_mer_against_poi(angles: pd.DataFrame, poi: pd.DataFrame, pitch: str, mer_time: float) -> float:
    mer_angle = interpolate_at(angles, "shoulder_angle_z", mer_time)
    poi_value = float(poi.loc[pitch, "max_shoulder_external_rotation"])
    if not np.isfinite(mer_angle) or not np.isfinite(poi_value):
        raise ValueError(f"{pitch} has missing MER angle validation value")
    if abs(mer_angle - poi_value) > 1e-3:
        raise ValueError(f"{pitch} MER angle mismatch: full_signal={mer_angle:.6f}, poi={poi_value:.6f}")
    return mer_angle


def linear_summary(data: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    clean = data[[x_col, y_col]].dropna()
    if len(clean) < 3 or clean[x_col].nunique() < 2:
        return {"n": int(len(clean)), "slope": np.nan, "intercept": np.nan, "r": np.nan, "r2": np.nan, "p": np.nan}
    result = stats.linregress(clean[x_col].to_numpy(dtype=float), clean[y_col].to_numpy(dtype=float))
    return {
        "n": int(len(clean)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "r2": float(result.rvalue**2),
        "p": float(result.pvalue),
    }


def rank_summary(data: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    clean = data[[x_col, y_col]].dropna()
    result = stats.spearmanr(clean[x_col], clean[y_col])
    return {"n": int(len(clean)), "rho": float(result.statistic), "p": float(result.pvalue)}


def clustered_ols(data: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> dict[str, float | int]:
    clean = data[[x_col, y_col, group_col]].dropna()
    if len(clean) < 3 or clean[x_col].nunique() < 2 or clean[group_col].nunique() < 2:
        return {"n": int(len(clean)), "n_groups": int(clean[group_col].nunique()), "slope": np.nan, "se": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    design = sm.add_constant(clean[[x_col]].astype(float), has_constant="add")
    fit = sm.OLS(clean[y_col].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": clean[group_col].astype(str)}
    )
    ci = fit.conf_int(alpha=0.05).loc[x_col]
    return {
        "n": int(len(clean)),
        "n_groups": int(clean[group_col].nunique()),
        "slope": float(fit.params[x_col]),
        "se": float(fit.bse[x_col]),
        "p": float(fit.pvalues[x_col]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
    }


def cluster_bootstrap(data: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> dict[str, float | int]:
    clean = data[[x_col, y_col, group_col]].dropna()
    grouped = [group for _, group in clean.groupby(group_col, sort=True)]
    if len(grouped) < 2:
        raise ValueError("Need at least two pitcher groups for clustered bootstrap")
    x_arrays = [group[x_col].to_numpy(dtype=float) for group in grouped]
    y_arrays = [group[y_col].to_numpy(dtype=float) for group in grouped]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    slopes: list[float] = []
    correlations: list[float] = []
    n_groups = len(grouped)
    for _ in range(BOOTSTRAP_N):
        chosen = rng.integers(0, n_groups, size=n_groups)
        x = np.concatenate([x_arrays[index] for index in chosen])
        y = np.concatenate([y_arrays[index] for index in chosen])
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = float(np.dot(x_centered, x_centered))
        x_sd = float(np.sqrt(np.dot(x_centered, x_centered)))
        y_sd = float(np.sqrt(np.dot(y_centered, y_centered)))
        if denominator <= 0 or x_sd <= 0 or y_sd <= 0:
            continue
        slopes.append(float(np.dot(x_centered, y_centered) / denominator))
        correlations.append(float(np.dot(x_centered, y_centered) / (x_sd * y_sd)))
    if not slopes:
        raise ValueError("Cluster bootstrap produced no valid resamples")
    return {
        "n_groups": int(n_groups),
        "n_bootstrap": int(len(slopes)),
        "slope_ci_low": float(np.percentile(slopes, 2.5)),
        "slope_ci_high": float(np.percentile(slopes, 97.5)),
        "r_ci_low": float(np.percentile(correlations, 2.5)),
        "r_ci_high": float(np.percentile(correlations, 97.5)),
    }


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    landmarks = normalize_key(read_full_signal("landmarks.csv", LANDMARK_COLS))
    angles = normalize_key(read_full_signal("joint_angles.csv", ANGLE_COLS))
    metadata = normalize_key(pd.read_csv(META_PATH, usecols=META_COLS, dtype={"session_pitch": "string"}))
    poi = normalize_key(pd.read_csv(POI_PATH, usecols=POI_COLS, dtype={"session_pitch": "string"}))
    require_columns(landmarks, LANDMARK_COLS, "landmarks.csv")
    require_columns(angles, ANGLE_COLS, "joint_angles.csv")
    require_columns(metadata, META_COLS, "metadata.csv")
    require_columns(poi, POI_COLS, "poi_metrics.csv")
    validate_speed_sources(metadata, poi)

    pitch_keys = set(metadata["session_pitch"].astype(str))
    for frame, label in ((landmarks, "landmarks.csv"), (angles, "joint_angles.csv")):
        if set(frame["session_pitch"].astype(str)) != pitch_keys:
            raise ValueError(f"{label} pitch keys do not match metadata.csv")

    landmark_groups = {str(key): group.sort_values("time") for key, group in landmarks.groupby("session_pitch", sort=False)}
    angle_groups = {str(key): group.sort_values("time") for key, group in angles.groupby("session_pitch", sort=False)}
    poi_by_pitch = poi.set_index("session_pitch")
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []

    for _, meta_row in metadata.sort_values("session_pitch").iterrows():
        pitch = str(meta_row["session_pitch"])
        landmark_group = landmark_groups[pitch]
        angle_group = angle_groups[pitch]
        fp = event_value(angle_group, "fp_poi_time", pitch)
        mer = event_value(angle_group, "MER_time", pitch)
        br = event_value(angle_group, "BR_time", pitch)
        for event in ("fp_poi_time", "MER_time", "BR_time"):
            landmark_event = event_value(landmark_group, event, pitch)
            angle_event = {"fp_poi_time": fp, "MER_time": mer, "BR_time": br}[event]
            if abs(landmark_event - angle_event) > 1e-9:
                raise ValueError(f"{pitch} {event} mismatch between landmarks and joint_angles")
        if not (fp < mer < br):
            raise ValueError(f"{pitch} violates fp_poi_time < MER_time < BR_time")

        mer_angle = validate_mer_against_poi(angle_group, poi_by_pitch, pitch, mer)
        crossings = find_ascending_er90_crossings(angle_group, fp, mer)
        if len(crossings) != 1:
            exclusions.append(
                {
                    "session_pitch": pitch,
                    "user": str(meta_row["user"]),
                    "reason": "expected exactly one ascending shoulder_angle_z=90 crossing",
                    "ascending_crossing_count": int(len(crossings)),
                    "pitch_speed_mph": float(meta_row["pitch_speed_mph"]),
                }
            )
            continue

        er90 = crossings[0]
        if not (fp < er90 < mer):
            raise ValueError(f"{pitch} violates fp_poi_time < ER90 < MER_time")
        hand_er90 = interpolate_at(landmark_group, "hand_jc_z", er90)
        hand_mer = interpolate_at(landmark_group, "hand_jc_z", mer)
        max_drop_m, max_drop_start, max_drop_end = maximum_vertical_drawdown(
            landmark_group, "hand_jc_z", er90, mer
        )
        if not all(np.isfinite(value) for value in (hand_er90, hand_mer, max_drop_m, max_drop_start, max_drop_end)):
            raise ValueError(f"{pitch} cannot calculate hand_jc_z maximum drop within ER90-to-MER")
        body_height = float(meta_row["session_height_m"])
        if not np.isfinite(body_height) or body_height <= 0:
            raise ValueError(f"{pitch} has invalid session_height_m")
        rows.append(
            {
                "session_pitch": pitch,
                "user": str(meta_row["user"]),
                "session_height_m": body_height,
                "pitch_speed_mph": float(meta_row["pitch_speed_mph"]),
                "fp_poi_time": fp,
                "ER90_time": er90,
                "MER_time": mer,
                "BR_time": br,
                "ER90_to_MER_duration_s": mer - er90,
                "shoulder_angle_z_at_MER_deg": mer_angle,
                "hand_jc_z_at_ER90_m": hand_er90,
                "hand_jc_z_at_MER_m": hand_mer,
                "max_hand_height_drop_m": max_drop_m,
                "max_hand_height_drop_cm": max_drop_m * 100.0,
                "max_hand_height_drop_pct_body_height": max_drop_m / body_height * 100.0,
                "max_hand_height_drop_start_time": max_drop_start,
                "max_hand_height_drop_end_time": max_drop_end,
                "max_hand_height_drop_duration_s": max_drop_end - max_drop_start,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("session_pitch").reset_index(drop=True)
    exclusions_frame = pd.DataFrame(exclusions)
    if metrics.empty:
        raise ValueError("No pitch passed the ER90 reconstruction and hand-height interpolation checks")

    height_col = "max_hand_height_drop_cm"
    metrics["height_drop_cm_within_pitcher"] = metrics[height_col] - metrics.groupby("user")[height_col].transform("mean")
    metrics["speed_mph_within_pitcher"] = metrics["pitch_speed_mph"] - metrics.groupby("user")["pitch_speed_mph"].transform("mean")
    raw_linear = linear_summary(metrics, height_col, "pitch_speed_mph")
    raw_rank = rank_summary(metrics, height_col, "pitch_speed_mph")
    clustered = clustered_ols(metrics, height_col, "pitch_speed_mph", "user")
    bootstrap = cluster_bootstrap(metrics, height_col, "pitch_speed_mph", "user")
    within_linear = linear_summary(metrics, "height_drop_cm_within_pitcher", "speed_mph_within_pitcher")
    within_clustered = clustered_ols(metrics, "height_drop_cm_within_pitcher", "speed_mph_within_pitcher", "user")
    pitcher_means = metrics.groupby("user", as_index=False).agg(
        height_mean_cm=(height_col, "mean"),
        speed_mean_mph=("pitch_speed_mph", "mean"),
        n_pitches=("session_pitch", "count"),
    )
    pitcher_mean = linear_summary(pitcher_means, "height_mean_cm", "speed_mean_mph")

    summary: dict[str, object] = {
        "definition": {
            "ER90": "unique ascending crossing of shoulder_angle_z=90 degrees between fp_poi_time and MER_time",
            "MER": "full-signal MER_time; validated against poi_metrics.max_shoulder_external_rotation",
            "height": "hand_jc_z, the throwing-hand landmark/joint-center vertical position; positive global z is upward",
            "maximum_drop": "max over ER90<=s<=t<=MER of hand_jc_z(s)-hand_jc_z(t); not the ER90-to-MER endpoint difference",
            "speed": "metadata.pitch_speed_mph, exact-match validated against poi_metrics.pitch_speed_mph",
        },
        "data": {
            "source_pitch_count": int(len(metadata)),
            "valid_pitch_count": int(len(metrics)),
            "excluded_pitch_count": int(len(exclusions_frame)),
            "pitcher_count_valid": int(metrics["user"].nunique()),
            "excluded_reasons": exclusions_frame["reason"].value_counts().to_dict() if not exclusions_frame.empty else {},
        },
        "maximum_drop": {
            "mean_cm": float(metrics[height_col].mean()),
            "sd_cm": float(metrics[height_col].std(ddof=1)),
            "median_cm": float(metrics[height_col].median()),
            "q25_cm": float(metrics[height_col].quantile(0.25)),
            "q75_cm": float(metrics[height_col].quantile(0.75)),
            "min_cm": float(metrics[height_col].min()),
            "max_cm": float(metrics[height_col].max()),
            "percent_positive_drop": float((metrics[height_col] > 0).mean() * 100.0),
            "mean_pct_body_height": float(metrics["max_hand_height_drop_pct_body_height"].mean()),
        },
        "raw_pitch_level": {"linear": raw_linear, "spearman": raw_rank},
        "pitcher_clustered": {"ols": clustered, "bootstrap": bootstrap},
        "within_pitcher_centered": {"linear": within_linear, "clustered_ols": within_clustered},
        "pitcher_mean_level": {"linear": pitcher_mean},
    }
    return metrics, exclusions_frame, summary


def make_plot(metrics: pd.DataFrame, summary: dict[str, object]) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = summary["raw_pitch_level"]["linear"]
    bootstrap = summary["pitcher_clustered"]["bootstrap"]
    x_values = metrics["max_hand_height_drop_cm"]
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(x_values, metrics["pitch_speed_mph"], s=28, alpha=0.62, color="#2f6f9f", edgecolors="white", linewidths=0.35)
    ax.plot(
        x_line,
        raw["intercept"] + raw["slope"] * x_line,
        color="#c43c39",
        linewidth=2,
        label=f"OLS slope={raw['slope']:.2f} mph/cm",
    )
    ax.axvline(0, color="0.55", linewidth=1, linestyle="--")
    ax.set_xlabel("Maximum throwing-hand vertical drop within ER90 to MER (cm)")
    ax.set_ylabel("Pitch speed (mph)")
    ax.set_title(
        "Maximum Throwing-Hand Vertical Drop within ER90-to-MER vs Pitch Speed\n"
        f"n={len(metrics)} pitches, {metrics['user'].nunique()} pitchers; raw r={raw['r']:.3f}; "
        f"cluster-bootstrap slope 95% CI [{bootstrap['slope_ci_low']:.2f}, {bootstrap['slope_ci_high']:.2f}]"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics, exclusions, summary = build_metrics()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "er90_mer_hand_height_speed_metrics.csv"
    exclusions_path = OUT_DIR / "er90_mer_exclusions.csv"
    summary_path = OUT_DIR / "er90_mer_hand_height_speed_summary.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    exclusions.to_csv(exclusions_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(metrics, summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print(f"metrics={metrics_path}")
    print(f"exclusions={exclusions_path}")
    print(f"summary={summary_path}")
    print(f"figure={FIG_PATH}")


if __name__ == "__main__":
    main()
