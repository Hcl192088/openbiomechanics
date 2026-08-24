"""Build a time-ordered association network from all analyzable POI metrics.

Edges are associations, not causal effects. Direction is imposed from explicit
event names or reconstructed peak timing; it is not learned causal direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FULL_SIG_DIR = DATA_DIR / "full_sig"
POI_PATH = DATA_DIR / "poi" / "poi_metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "poi_motion_network_outputs"
FIGURE_PATH = ROOT / "imgs" / "all_poi_motion_relationship_network.png"

MIN_N = 250
MIN_ABS_WITHIN_R = 0.15
FDR_ALPHA = 0.05

STAGE_LABELS = {
    0: "PKH",
    1: "PKH-FP",
    2: "FP",
    3: "FP-MER",
    4: "MER / FP-BR aggregate",
    5: "MER-BR",
    6: "BR",
    7: "BR-MIR",
    8: "MIR",
    9: "Outcome",
}

NON_NODE_COLUMNS = {
    "session_pitch",
    "session",
    "p_throws",
    "pitch_type",
    "fp_poi_time",
    "fp_poi_range_ms",
    "fp_poi_n",
}

LABELS = {
    "pitch_speed_mph": "Pitch velocity",
    "max_shoulder_internal_rotational_velo": "Peak shoulder IR velocity",
    "max_elbow_extension_velo": "Peak elbow extension velocity",
    "max_torso_rotational_velo": "Peak torso rotation velocity",
    "max_rotation_hip_shoulder_separation": "Peak hip-shoulder separation (+)",
    "max_elbow_flexion": "Peak elbow flexion (+)",
    "max_shoulder_external_rotation": "Maximum shoulder ER (+)",
    "max_cog_velo_x": "Peak COG forward velocity",
    "torso_rotation_min": "Peak torso counterrotation",
    "max_pelvis_rotational_velo": "Peak pelvis rotation velocity",
    "elbow_varus_moment": "Peak elbow varus moment",
    "shoulder_internal_rotation_moment": "Peak shoulder IR moment",
    "cog_velo_pkh": "COG velocity at PKH",
    "stride_length": "Stride length",
    "stride_angle": "Stride angle (cross-body +)",
    "arm_slot": "Arm slot",
    "timing_peak_torso_to_peak_pelvis_rot_velo": "Pelvis-to-torso peak timing",
    "max_shoulder_horizontal_abduction": "Peak shoulder horizontal abduction (+)",
    "peak_rfd_rear": "Peak rear RFD",
    "peak_rfd_lead": "Peak lead RFD",
    "max_rear_hip_internal_rotation_velo": "Peak rear-hip IR velocity",
    "max_rear_hip_flexion": "Peak rear-hip flexion (-)",
}

SIGN_SUFFIX = {
    "elbow_flexion_fp": " (flexion +)",
    "elbow_pronation_fp": " (pronation +)",
    "rotation_hip_shoulder_separation_fp": " (separation +)",
    "shoulder_horizontal_abduction_fp": " (horizontal abduction +)",
    "shoulder_abduction_fp": " (adduction + / abduction -)",
    "shoulder_external_rotation_fp": " (external rotation +)",
    "torso_anterior_tilt_fp": " (flexion +)",
    "torso_anterior_tilt_mer": " (flexion +)",
    "torso_anterior_tilt_br": " (flexion +)",
    "pelvis_anterior_tilt_fp": " (anterior tilt +)",
}

# target -> (full-signal file, source column, extreme operation, search window)
PEAK_MAPPINGS = {
    "max_shoulder_internal_rotational_velo": ("joint_velos.csv", "shoulder_velo_z", "max", "all"),
    "max_elbow_extension_velo": ("joint_velos.csv", "elbow_velo_x", "max", "all"),
    "max_torso_rotational_velo": ("joint_velos.csv", "torso_velo_z", "max", "all"),
    "max_rotation_hip_shoulder_separation": ("joint_angles.csv", "torso_pelvis_angle_z", "max", "all"),
    "max_elbow_flexion": ("joint_angles.csv", "elbow_angle_x", "max", "fp_br"),
    "max_shoulder_external_rotation": ("joint_angles.csv", "shoulder_angle_z", "max", "all"),
    "lead_knee_extension_angular_velo_max": ("joint_velos.csv", "lead_knee_velo_x", "max", "fp_br"),
    "torso_rotation_min": ("joint_angles.csv", "torso_angle_z", "min", "all"),
    "max_pelvis_rotational_velo": ("joint_velos.csv", "pelvis_velo_z", "max", "all"),
    "elbow_varus_moment": ("forces_moments.csv", "elbow_moment_y", "max", "all"),
    "shoulder_internal_rotation_moment": ("forces_moments.csv", "shoulder_upper_arm_moment_z", "max", "all"),
    "max_shoulder_horizontal_abduction": ("joint_angles.csv", "shoulder_angle_x", "max", "all"),
    "max_rear_hip_internal_rotation_velo": ("joint_velos.csv", "rear_hip_velo_z", "negative_min", "all"),
    "max_rear_hip_flexion": ("joint_angles.csv", "rear_hip_angle_x", "negative_min", "all"),
}


def readable_label(column: str) -> str:
    if column in LABELS:
        return LABELS[column]
    text = column.replace("_", " ")
    replacements = {
        " fp br": " FP-BR",
        " pkh fp": " PKH-FP",
        " fp": " at FP",
        " mer": " at MER",
        " br": " at BR",
        " velo": " velocity",
        " grf": " GRF",
        " cog": " COG",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.strip().title()
    return text + SIGN_SUFFIX.get(column, "")


def phase_coordinate(time: pd.Series, pkh: pd.Series, fp: pd.Series, mer: pd.Series, br: pd.Series, mir: pd.Series) -> pd.Series:
    """Map clock time to PKH=0, FP=2, MER=4, BR=6, MIR=8."""
    out = pd.Series(np.nan, index=time.index, dtype=float)
    valid1 = time.notna() & pkh.notna() & fp.notna() & (fp > pkh) & (time <= fp)
    out.loc[valid1] = 2 * (time.loc[valid1] - pkh.loc[valid1]) / (fp.loc[valid1] - pkh.loc[valid1])
    valid2 = time.notna() & fp.notna() & mer.notna() & (mer > fp) & (time > fp) & (time <= mer)
    out.loc[valid2] = 2 + 2 * (time.loc[valid2] - fp.loc[valid2]) / (mer.loc[valid2] - fp.loc[valid2])
    valid3 = time.notna() & mer.notna() & br.notna() & (br > mer) & (time > mer) & (time <= br)
    out.loc[valid3] = 4 + 2 * (time.loc[valid3] - mer.loc[valid3]) / (br.loc[valid3] - mer.loc[valid3])
    valid4 = time.notna() & br.notna() & mir.notna() & (mir > br) & (time > br)
    out.loc[valid4] = 6 + 2 * (time.loc[valid4] - br.loc[valid4]) / (mir.loc[valid4] - br.loc[valid4])
    return out.clip(0, 8)


def extreme_rows(data: pd.DataFrame, signal: str, operation: str, window: str = "all") -> pd.DataFrame:
    if window == "fp_br":
        data = data[(data["time"] >= data["fp_poi_time"]) & (data["time"] <= data["BR_time"])]
    grouped = data.groupby("session_pitch", sort=False)[signal]
    idx = grouped.idxmin() if operation in {"min", "negative_min"} else grouped.idxmax()
    rows = data.loc[idx.dropna().astype(int), [
        "session_pitch", "time", signal, "pkh_time", "fp_poi_time", "MER_time", "BR_time", "MIR_time"
    ]].copy()
    rows["derived_value"] = -rows[signal] if operation == "negative_min" else rows[signal]
    rows["phase_coordinate"] = phase_coordinate(
        rows["time"], rows["pkh_time"], rows["fp_poi_time"], rows["MER_time"], rows["BR_time"], rows["MIR_time"]
    )
    return rows


def load_peak_timing(poi: pd.DataFrame) -> pd.DataFrame:
    results = []
    by_file: dict[str, list[tuple[str, str, str]]] = {}
    for target, spec in PEAK_MAPPINGS.items():
        if target in poi.columns:
            by_file.setdefault(spec[0], []).append((target, spec[1], spec[2], spec[3]))

    event_cols = ["session_pitch", "time", "pkh_time", "fp_poi_time", "MER_time", "BR_time", "MIR_time"]
    for filename, specs in by_file.items():
        usecols = event_cols + sorted({signal for _, signal, _, _ in specs})
        data = pd.read_csv(FULL_SIG_DIR / filename, usecols=usecols)
        for target, signal, operation, window in specs:
            rows = extreme_rows(data, signal, operation, window)
            comparison = poi[["session_pitch", target]].merge(
                rows[["session_pitch", "derived_value", "phase_coordinate"]], on="session_pitch", how="inner"
            ).dropna()
            median_phase = float(comparison["phase_coordinate"].median())
            results.append({
                "column": target,
                "timing_method": f"full_signal:{filename}:{signal}:{operation}:window={window}",
                "timing_confidence": "validated_mapping",
                "timing_n": int(comparison["phase_coordinate"].notna().sum()),
                "median_phase_coordinate": median_phase,
                "nearest_stage_index": int(np.clip(np.floor(median_phase + 0.5), 0, 8)),
                "value_match_r": float(comparison[target].corr(comparison["derived_value"])),
                "median_abs_value_error": float((comparison[target] - comparison["derived_value"]).abs().median()),
            })

    # GRF component/magnitude peaks and angle-at-magnitude timing.
    force_cols = event_cols + [
        "rear_force_x", "rear_force_y", "rear_force_z", "lead_force_x", "lead_force_y", "lead_force_z"
    ]
    force = pd.read_csv(FULL_SIG_DIR / "force_plate.csv", usecols=force_cols)
    force["rear_force_mag"] = np.sqrt(sum(force[f"rear_force_{axis}"] ** 2 for axis in "xyz"))
    force["lead_force_mag"] = np.sqrt(sum(force[f"lead_force_{axis}"] ** 2 for axis in "xyz"))
    grf_specs = {
        "rear_grf_x_max": ("rear_force_x", "max"),
        "rear_grf_y_max": ("rear_force_y", "max"),
        "rear_grf_z_max": ("rear_force_z", "max"),
        "rear_grf_mag_max": ("rear_force_mag", "max"),
        "lead_grf_x_max": ("lead_force_x", "max"),
        "lead_grf_y_max": ("lead_force_y", "max"),
        "lead_grf_z_max": ("lead_force_z", "max"),
        "lead_grf_mag_max": ("lead_force_mag", "max"),
    }
    peak_rows: dict[str, pd.DataFrame] = {}
    for target, (signal, operation) in grf_specs.items():
        if target not in poi.columns:
            continue
        rows = extreme_rows(force, signal, operation)
        peak_rows[target] = rows
        comparison = poi[["session_pitch", target]].merge(
            rows[["session_pitch", "derived_value", "phase_coordinate"]], on="session_pitch", how="inner"
        ).dropna()
        median_phase = float(comparison["phase_coordinate"].median())
        results.append({
            "column": target,
            "timing_method": f"full_signal:force_plate.csv:{signal}:max",
            "timing_confidence": "validated_mapping",
            "timing_n": int(comparison["phase_coordinate"].notna().sum()),
            "median_phase_coordinate": median_phase,
            "nearest_stage_index": int(np.clip(np.floor(median_phase + 0.5), 0, 8)),
            "value_match_r": float(comparison[target].corr(comparison["derived_value"])),
            "median_abs_value_error": float((comparison[target] - comparison["derived_value"]).abs().median()),
        })

    for angle_target, mag_target in {
        "rear_grf_angle_at_max": "rear_grf_mag_max",
        "lead_grf_angle_at_max": "lead_grf_mag_max",
    }.items():
        if angle_target in poi.columns and mag_target in peak_rows:
            coords = peak_rows[mag_target]["phase_coordinate"].dropna()
            median_phase = float(coords.median())
            results.append({
                "column": angle_target,
                "timing_method": f"same_time_as:{mag_target}",
                "timing_confidence": "definition_linked",
                "timing_n": int(len(coords)),
                "median_phase_coordinate": median_phase,
                "nearest_stage_index": int(np.clip(np.floor(median_phase + 0.5), 0, 8)),
                "value_match_r": np.nan,
                "median_abs_value_error": np.nan,
            })

    # RFD source algorithm is unavailable. Use the timing of maximum positive
    # derivative of GRF magnitude as a labeled timing proxy, not a reconstruction.
    force = force.sort_values(["session_pitch", "time"])
    for side, target in [("rear", "peak_rfd_rear"), ("lead", "peak_rfd_lead")]:
        if target not in poi.columns:
            continue
        force[f"{side}_rfd_proxy"] = force.groupby("session_pitch")[f"{side}_force_mag"].diff() / force.groupby("session_pitch")["time"].diff()
        rows = extreme_rows(force, f"{side}_rfd_proxy", "max")
        coords = rows["phase_coordinate"].dropna()
        median_phase = float(coords.median())
        results.append({
            "column": target,
            "timing_method": f"proxy:max_d({side}_grf_magnitude)/dt",
            "timing_confidence": "proxy_unverified_poi_algorithm",
            "timing_n": int(len(coords)),
            "median_phase_coordinate": median_phase,
            "nearest_stage_index": int(np.clip(np.floor(median_phase + 0.5), 0, 8)),
            "value_match_r": np.nan,
            "median_abs_value_error": np.nan,
        })

    # COG forward velocity timing from the derivative of center-of-mass X.
    if "max_cog_velo_x" in poi.columns:
        landmark_cols = event_cols + ["centerofmass_x"]
        landmarks = pd.read_csv(FULL_SIG_DIR / "landmarks.csv", usecols=landmark_cols)
        landmarks = landmarks.sort_values(["session_pitch", "time"])
        landmarks["cog_velo_x"] = landmarks.groupby("session_pitch")["centerofmass_x"].diff() / landmarks.groupby("session_pitch")["time"].diff()
        rows = extreme_rows(landmarks, "cog_velo_x", "max")
        comparison = poi[["session_pitch", "max_cog_velo_x"]].merge(
            rows[["session_pitch", "derived_value", "phase_coordinate"]], on="session_pitch", how="inner"
        ).dropna()
        median_phase = float(comparison["phase_coordinate"].median())
        results.append({
            "column": "max_cog_velo_x",
            "timing_method": "derived:landmarks:centerofmass_x:d/dt:max",
            "timing_confidence": "derived_mapping",
            "timing_n": int(comparison["phase_coordinate"].notna().sum()),
            "median_phase_coordinate": median_phase,
            "nearest_stage_index": int(np.clip(np.floor(median_phase + 0.5), 0, 8)),
            "value_match_r": float(comparison["max_cog_velo_x"].corr(comparison["derived_value"])),
            "median_abs_value_error": float((comparison["max_cog_velo_x"] - comparison["derived_value"]).abs().median()),
        })

    return pd.DataFrame(results)


def explicit_stage(column: str) -> tuple[int, str] | None:
    if column == "pitch_speed_mph":
        return 9, "outcome"
    if column == "cog_velo_pkh":
        return 0, "explicit_pkh"
    if column.endswith("_pkh_fp"):
        return 1, "explicit_pkh_fp_window"
    if column.endswith("_fp"):
        return 2, "explicit_fp"
    if column in {"stride_length", "stride_angle"}:
        return 2, "definition_at_foot_plant"
    if column == "timing_peak_torso_to_peak_pelvis_rot_velo":
        return 3, "pelvis_to_torso_peak_interval"
    if column.endswith("_fp_br") or column == "lead_knee_extension_from_fp_to_br":
        return 4, "explicit_fp_br_window_midpoint"
    if column.endswith("_mer"):
        return 4, "explicit_mer"
    if column.endswith("_br") or column == "arm_slot":
        return 6, "explicit_br_or_release_definition"
    return None


def build_nodes(poi: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    timing_lookup = timing.set_index("column").to_dict("index") if not timing.empty else {}
    rows = []
    for column in poi.columns:
        if column in NON_NODE_COLUMNS or not pd.api.types.is_numeric_dtype(poi[column]):
            continue
        explicit = explicit_stage(column)
        if explicit is not None:
            stage_index, method = explicit
            timing_info = {
                "timing_method": method,
                "timing_confidence": "explicit_definition",
                "timing_n": int(poi[column].notna().sum()),
                "median_phase_coordinate": float(stage_index),
                "value_match_r": np.nan,
                "median_abs_value_error": np.nan,
            }
        elif column in timing_lookup:
            timing_info = timing_lookup[column]
            stage_index = int(timing_info["nearest_stage_index"])
        else:
            stage_index = 9
            timing_info = {
                "timing_method": "unresolved_no_event_or_full_signal_mapping",
                "timing_confidence": "unresolved",
                "timing_n": 0,
                "median_phase_coordinate": np.nan,
                "value_match_r": np.nan,
                "median_abs_value_error": np.nan,
            }
        rows.append({
            "column": column,
            "label": readable_label(column),
            "stage_index": stage_index,
            "stage": STAGE_LABELS[stage_index],
            "non_missing_n": int(poi[column].notna().sum()),
            **{k: v for k, v in timing_info.items() if k != "nearest_stage_index"},
        })
    nodes = pd.DataFrame(rows)
    nodes["stage_order"] = nodes.groupby("stage_index").cumcount()
    return nodes


def bh_fdr(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=p_values.index)


def safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def analyze_pair(df: pd.DataFrame, source: str, target: str) -> dict[str, float]:
    pair = df[["session", source, target]].dropna().copy()
    raw_r, raw_p = safe_pearson(pair[source], pair[target])
    pair["source_within"] = pair[source] - pair.groupby("session")[source].transform("mean")
    pair["target_within"] = pair[target] - pair.groupby("session")[target].transform("mean")
    within_r, within_pearson_p = safe_pearson(pair["source_within"], pair["target_within"])

    def clustered(x: pd.Series, y: pd.Series, add_constant: bool) -> tuple[float, float]:
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        if not (sx > 0 and sy > 0):
            return np.nan, np.nan
        zx, zy = (x - x.mean()) / sx, (y - y.mean()) / sy
        design = sm.add_constant(zx) if add_constant else zx
        model = sm.OLS(zy, design).fit(cov_type="cluster", cov_kwds={"groups": pair["session"]})
        return float(model.params.iloc[-1]), float(model.pvalues.iloc[-1])

    cluster_beta, cluster_p = clustered(pair[source], pair[target], True)
    _, within_cluster_p = clustered(pair["source_within"], pair["target_within"], False)
    return {
        "n": int(len(pair)),
        "sessions": int(pair["session"].nunique()),
        "raw_r": raw_r,
        "raw_p": raw_p,
        "cluster_beta": cluster_beta,
        "cluster_p": cluster_p,
        "within_r": within_r,
        "within_pearson_p": within_pearson_p,
        "within_cluster_p": within_cluster_p,
    }


def build_edges(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in nodes.itertuples():
        for target in nodes.itertuples():
            if source.stage_index >= target.stage_index:
                continue
            rows.append({
                "source": source.column,
                "source_label": source.label,
                "source_stage": source.stage,
                "target": target.column,
                "target_label": target.label,
                "target_stage": target.stage,
                **analyze_pair(df, source.column, target.column),
            })
    edges = pd.DataFrame(rows)
    edges["within_q"] = bh_fdr(edges["within_cluster_p"])
    edges["cluster_q"] = bh_fdr(edges["cluster_p"])
    edges["selected"] = (
        (edges["n"] >= MIN_N)
        & (edges["within_r"].abs() >= MIN_ABS_WITHIN_R)
        & (edges["within_q"] < FDR_ALPHA)
    )
    return edges


def plot_network(nodes: pd.DataFrame, selected: pd.DataFrame) -> None:
    active_stages = sorted(nodes["stage_index"].unique())
    stage_x = {stage_index: order * 4.7 for order, stage_index in enumerate(active_stages)}
    max_nodes = int(nodes.groupby("stage_index").size().max())
    positions = {}
    fig, ax = plt.subplots(figsize=(34, 24), facecolor="#08101f")
    ax.set_facecolor("#08101f")
    for stage_index in active_stages:
        stage_nodes = nodes[nodes.stage_index == stage_index]
        y_values = np.linspace(max_nodes - 1, 0, len(stage_nodes))
        for y, row in zip(y_values, stage_nodes.itertuples()):
            positions[row.column] = (stage_x[stage_index], float(y))

    for edge in selected.sort_values("within_r", key=lambda s: s.abs()).itertuples():
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        color = "#50d6b1" if edge.within_r > 0 else "#f06f6f"
        width = 0.55 + 4.2 * min(abs(edge.within_r), 0.7)
        alpha = 0.25 + 0.55 * min(abs(edge.within_r) / 0.5, 1.0)
        ax.annotate("", xy=(x2 - 0.9, y2), xytext=(x1 + 0.9, y1), arrowprops={
            "arrowstyle": "-|>", "color": color, "lw": width, "alpha": alpha,
            "connectionstyle": "arc3,rad=0.04", "shrinkA": 2, "shrinkB": 2,
        }, zorder=1)

    confidence_colors = {
        "explicit_definition": "#58739a",
        "validated_mapping": "#5f9f89",
        "definition_linked": "#5f9f89",
        "derived_mapping": "#a48656",
        "proxy_unverified_poi_algorithm": "#b36b6b",
        "unresolved": "#b36b6b",
    }
    for row in nodes.itertuples():
        x, y = positions[row.column]
        ax.text(x, y, row.label, ha="center", va="center", fontsize=7.1, color="#edf4ff",
                bbox={"boxstyle": "round,pad=0.42", "facecolor": "#17243b",
                      "edgecolor": confidence_colors.get(row.timing_confidence, "#58739a"), "linewidth": 0.9}, zorder=3)

    for stage_index in active_stages:
        ax.text(stage_x[stage_index], max_nodes + 0.7, STAGE_LABELS[stage_index], ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#f0c96a")
    legend = [
        Line2D([0], [0], color="#50d6b1", lw=3, label="Positive within-session association"),
        Line2D([0], [0], color="#f06f6f", lw=3, label="Negative within-session association"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor="#b36b6b", markersize=9,
               label="Timing proxy or unresolved", linestyle="None"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3,
              frameon=False, labelcolor="#dce8fa")
    ax.text(0, -1.5,
            f"All numeric POI metrics except IDs and FP reconstruction QC. Associations only; arrows encode timing order, not causality. "
            f"Selected: |within-session r| >= {MIN_ABS_WITHIN_R:.2f}, cluster-robust BH-FDR q < {FDR_ALPHA:.2f}, n >= {MIN_N}.",
            color="#aab8cc", fontsize=9)
    ax.set_xlim(-2.0, max(stage_x.values()) + 2.0)
    ax.set_ylim(-1.8, max_nodes + 1.4)
    ax.axis("off")
    fig.suptitle("All-POI Motion Relationship Network", color="#f5f8ff", fontsize=22, fontweight="bold", y=0.99)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    poi = pd.read_csv(POI_PATH)
    timing = load_peak_timing(poi)
    nodes = build_nodes(poi, timing)
    unresolved = nodes[nodes.timing_confidence == "unresolved"]
    if not unresolved.empty:
        raise ValueError(f"POI timing classification unresolved: {unresolved.column.tolist()}")
    edges = build_edges(poi, nodes)
    selected = edges.loc[edges.selected].sort_values("within_r", key=lambda s: s.abs(), ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUTPUT_DIR / "poi_motion_network_nodes.csv", index=False)
    timing.to_csv(OUTPUT_DIR / "poi_motion_network_peak_timing.csv", index=False)
    edges.to_csv(OUTPUT_DIR / "poi_motion_network_all_edges.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "poi_motion_network_selected_edges.csv", index=False)
    summary = {
        "source": str(POI_PATH),
        "rows": int(len(poi)),
        "sessions": int(poi["session"].nunique()),
        "poi_columns_total": int(len(poi.columns)),
        "excluded_non_node_columns": sorted(NON_NODE_COLUMNS),
        "nodes": int(len(nodes)),
        "tested_time_ordered_pairs": int(len(edges)),
        "selected_edges": int(len(selected)),
        "stage_counts": {STAGE_LABELS[int(k)]: int(v) for k, v in nodes.groupby("stage_index").size().items()},
        "timing_confidence_counts": {str(k): int(v) for k, v in nodes.timing_confidence.value_counts().items()},
        "selection": {"minimum_n": MIN_N, "minimum_absolute_within_session_r": MIN_ABS_WITHIN_R, "bh_fdr_q": FDR_ALPHA},
        "direction_warning": "Arrows encode explicit/reconstructed timing order and are not estimated causal direction.",
    }
    (OUTPUT_DIR / "poi_motion_network_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_network(nodes, selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPeak timing classification:\n", timing.to_string(index=False))


if __name__ == "__main__":
    main()
