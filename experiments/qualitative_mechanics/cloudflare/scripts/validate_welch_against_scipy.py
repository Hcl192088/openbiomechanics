#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate Worker Welch p-values against SciPy using local labeled data."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scipy import stats


CLOUDFLARE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = CLOUDFLARE_DIR.parent
REPO_ROOT = CLOUDFLARE_DIR.parents[2]
LABELS_LONG_PATH = EXPERIMENT_DIR / "labels_long.csv"
POI_PATH = REPO_ROOT / "baseball_pitching" / "data" / "poi" / "poi_metrics.csv"

FIELD_VALUE_ORDER = {
    "hip_shoulder_separation": ["good", "average", "bad", "unclear"],
    "lower_body_dominance": ["glute", "mixed", "quad", "unclear"],
    "direction": ["good", "bad", "unclear"],
    "shoulder_horizontal_abduction": ["good", "average", "bad", "unclear"],
    "torso_velo_z": ["fast", "slow", "unclear"],
    "hip_extension": ["good", "bad", "unclear"],
    "heel_connection": ["connected", "early_extension", "unclear"],
    "drift": ["good", "average", "bad", "unclear"],
}

PILOT_FIELD_METRICS = {
    "hip_shoulder_separation": [
        "pitch_speed_mph",
        "max_rotation_hip_shoulder_separation",
        "rotation_hip_shoulder_separation_fp",
    ],
    "shoulder_horizontal_abduction": [
        "pitch_speed_mph",
        "shoulder_horizontal_abduction_fp",
        "max_shoulder_horizontal_abduction",
    ],
    "torso_velo_z": ["pitch_speed_mph", "max_torso_rotational_velo"],
    "hip_extension": [
        "pitch_speed_mph",
        "pelvis_rotation_fp",
        "rotation_hip_shoulder_separation_fp",
        "max_rotation_hip_shoulder_separation",
        "max_torso_rotational_velo",
        "cog_velo_pkh",
        "stride_length",
        "stride_angle",
        "max_rear_hip_flexion",
        "max_rear_hip_internal_rotation_velo",
        "rear_hip_transfer_pkh_fp",
        "rear_hip_generation_pkh_fp",
        "rear_hip_absorption_pkh_fp",
        "lead_hip_transfer_fp_br",
        "lead_hip_generation_fp_br",
        "lead_hip_absorption_fp_br",
        "lead_knee_extension_from_fp_to_br",
        "lead_knee_extension_angular_velo_fp",
        "lead_grf_x_max",
        "lead_grf_y_max",
        "lead_grf_z_max",
        "rear_grf_x_max",
        "rear_grf_y_max",
        "rear_grf_z_max",
    ],
    "direction": ["pitch_speed_mph", "stride_length", "stride_angle", "max_cog_velo_x"],
    "heel_connection": [
        "pitch_speed_mph",
        "lead_knee_extension_from_fp_to_br",
        "lead_knee_extension_angular_velo_fp",
        "lead_grf_z_max",
    ],
    "drift": ["pitch_speed_mph", "cog_velo_pkh", "max_cog_velo_x", "stride_angle"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ordered_values(field: str, values: list[str]) -> list[str]:
    preferred = FIELD_VALUE_ORDER.get(field, [])
    ordered = [value for value in preferred if value in values]
    ordered.extend(sorted(value for value in values if value not in set(ordered)))
    return ordered


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def cohen_d(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    pooled = (((len(left) - 1) * variance(left)) + ((len(right) - 1) * variance(right))) / (len(left) + len(right) - 2)
    return (mean(left) - mean(right)) / math.sqrt(pooled) if pooled > 0 else None


def worker_welch(left: list[float], right: list[float]) -> dict[str, float] | None:
    if len(left) < 2 or len(right) < 2:
        return None
    left_var = variance(left)
    right_var = variance(right)
    se_squared = (left_var / len(left)) + (right_var / len(right))
    if se_squared <= 0:
        return None
    t_value = (mean(left) - mean(right)) / math.sqrt(se_squared)
    numerator = se_squared**2
    denominator = ((left_var / len(left)) ** 2 / (len(left) - 1)) + ((right_var / len(right)) ** 2 / (len(right) - 1))
    df = numerator / denominator
    p_value = 2 * (1 - student_t_cdf(abs(t_value), df))
    return {
        "t": t_value,
        "df": df,
        "p": p_value,
        "mean_diff": mean(left) - mean(right),
        "cohen_d": cohen_d(left, right),
    }


def student_t_cdf(t_value: float, df: float) -> float:
    x_value = df / (df + (t_value * t_value))
    ib = regularized_incomplete_beta(x_value, df / 2, 0.5)
    return 1 - (0.5 * ib)


def regularized_incomplete_beta(x_value: float, a_value: float, b_value: float) -> float:
    if x_value <= 0:
        return 0.0
    if x_value >= 1:
        return 1.0
    bt = math.exp(
        math.lgamma(a_value + b_value)
        - math.lgamma(a_value)
        - math.lgamma(b_value)
        + (a_value * math.log(x_value))
        + (b_value * math.log(1 - x_value))
    )
    if x_value < (a_value + 1) / (a_value + b_value + 2):
        return (bt * beta_continued_fraction(x_value, a_value, b_value)) / a_value
    return 1 - ((bt * beta_continued_fraction(1 - x_value, b_value, a_value)) / b_value)


def beta_continued_fraction(x_value: float, a_value: float, b_value: float) -> float:
    max_iterations = 100
    epsilon = 3e-7
    fp_min = 1e-30
    qab = a_value + b_value
    qap = a_value + 1
    qam = a_value - 1
    c_value = 1.0
    d_value = 1 - (qab * x_value / qap)
    if abs(d_value) < fp_min:
        d_value = fp_min
    d_value = 1 / d_value
    h_value = d_value
    for m_value in range(1, max_iterations + 1):
        m2 = 2 * m_value
        aa = (m_value * (b_value - m_value) * x_value) / ((qam + m2) * (a_value + m2))
        d_value = 1 + (aa * d_value)
        if abs(d_value) < fp_min:
            d_value = fp_min
        c_value = 1 + (aa / c_value)
        if abs(c_value) < fp_min:
            c_value = fp_min
        d_value = 1 / d_value
        h_value *= d_value * c_value
        aa = -((a_value + m_value) * (qab + m_value) * x_value) / ((a_value + m2) * (qap + m2))
        d_value = 1 + (aa * d_value)
        if abs(d_value) < fp_min:
            d_value = fp_min
        c_value = 1 + (aa / c_value)
        if abs(c_value) < fp_min:
            c_value = fp_min
        d_value = 1 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1) < epsilon:
            break
    return h_value


def parse_float(row: dict[str, str], metric: str, session_pitch: str) -> float | None:
    raw = row.get(metric, "")
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {metric} for {session_pitch}: {raw!r}") from exc


def main() -> None:
    labels = read_csv(LABELS_LONG_PATH)
    poi = {row["session_pitch"]: row for row in read_csv(POI_PATH)}
    comparisons = []
    max_p_delta = 0.0
    max_t_delta = 0.0
    for field, metrics in PILOT_FIELD_METRICS.items():
        field_labels = [
            row for row in labels
            if row["item_name"] == field and row["label_value"] not in {"", "unclear"} and row["skipped"].lower() != "true"
        ]
        values = ordered_values(field, sorted({row["label_value"] for row in field_labels}))
        if len(values) != 2:
            continue
        for metric in metrics:
            grouped: dict[str, list[float]] = {value: [] for value in values}
            for label in field_labels:
                poi_row = poi.get(label["session_pitch"])
                if poi_row is None:
                    continue
                metric_value = parse_float(poi_row, metric, label["session_pitch"])
                if metric_value is not None:
                    grouped[label["label_value"]].append(metric_value)
            left, right = grouped[values[0]], grouped[values[1]]
            if len(left) < 2 or len(right) < 2:
                continue
            worker = worker_welch(left, right)
            scipy_result = stats.ttest_ind(left, right, equal_var=False)
            if worker is None:
                continue
            p_delta = abs(worker["p"] - float(scipy_result.pvalue))
            t_delta = abs(worker["t"] - float(scipy_result.statistic))
            max_p_delta = max(max_p_delta, p_delta)
            max_t_delta = max(max_t_delta, t_delta)
            comparisons.append((field, metric, values[0], len(left), values[1], len(right), worker["p"], scipy_result.pvalue, p_delta))

    print(f"comparisons={len(comparisons)}")
    print(f"max_p_delta={max_p_delta:.12g}")
    print(f"max_t_delta={max_t_delta:.12g}")
    for row in comparisons[:10]:
        field, metric, left_name, left_n, right_name, right_n, worker_p, scipy_p, p_delta = row
        print(
            f"{field}.{metric}: {left_name} n={left_n} vs {right_name} n={right_n} "
            f"worker_p={worker_p:.8f} scipy_p={scipy_p:.8f} delta={p_delta:.3g}"
        )
    if max_p_delta > 1e-6 or max_t_delta > 1e-10:
        raise RuntimeError("Welch validation exceeded tolerance.")


if __name__ == "__main__":
    main()
