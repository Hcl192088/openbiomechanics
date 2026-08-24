"""Build an exploratory, time-ordered POI motion-association network.

Edges are associations, not causal effects.  Direction is imposed only by the
predefined pitching-phase order; it is not learned from the data.
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
POI_PATH = ROOT / "data" / "poi" / "poi_metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "poi_motion_network_outputs"
FIGURE_PATH = ROOT / "imgs" / "poi_motion_relationship_network.png"

MIN_N = 250
MIN_ABS_WITHIN_R = 0.15
FDR_ALPHA = 0.05

# Phase ordering is project knowledge, not inferred directionality.
STAGES = [
    (
        "Drive / stride",
        [
            ("rear_grf_mag_max", "Rear GRF magnitude"),
            ("rear_grf_z_max", "Rear vertical GRF"),
            ("max_rear_hip_internal_rotation_velo", "Rear hip IR velocity"),
            ("max_rear_hip_flexion", "Rear hip flexion"),
            ("cog_velo_pkh", "COG velocity at PKH"),
            ("stride_length", "Stride length"),
            ("stride_angle", "Stride angle"),
        ],
    ),
    (
        "Foot plant",
        [
            ("pelvis_anterior_tilt_fp", "Pelvis anterior tilt (+)"),
            ("pelvis_rotation_fp", "Pelvis rotation"),
            ("torso_anterior_tilt_fp", "Torso flexion (+) at FP"),
            ("torso_rotation_fp", "Torso rotation"),
            ("rotation_hip_shoulder_separation_fp", "Hip-shoulder separation"),
            ("shoulder_horizontal_abduction_fp", "Shoulder horizontal abduction (+)"),
            ("shoulder_external_rotation_fp", "Shoulder external rotation"),
            ("lead_knee_extension_angular_velo_fp", "Lead-knee extension velocity"),
        ],
    ),
    (
        "Rotation / MER",
        [
            ("max_pelvis_rotational_velo", "Peak pelvis rotation velocity"),
            ("max_torso_rotational_velo", "Peak torso rotation velocity"),
            ("timing_peak_torso_to_peak_pelvis_rot_velo", "Pelvis-to-torso peak timing"),
            ("max_rotation_hip_shoulder_separation", "Peak hip-shoulder separation"),
            ("max_shoulder_external_rotation", "Maximum shoulder ER"),
            ("torso_anterior_tilt_mer", "Torso flexion (+) at MER"),
            ("torso_rotation_mer", "Torso rotation at MER"),
        ],
    ),
    (
        "Acceleration / release",
        [
            ("lead_knee_extension_from_fp_to_br", "Lead-knee extension FP-BR"),
            ("max_shoulder_internal_rotational_velo", "Shoulder IR velocity"),
            ("max_elbow_extension_velo", "Elbow extension velocity"),
            ("torso_anterior_tilt_br", "Torso flexion (+) at BR"),
            ("torso_rotation_br", "Torso rotation at BR"),
            ("arm_slot", "Arm slot"),
        ],
    ),
    (
        "Outcomes",
        [
            ("pitch_speed_mph", "Pitch velocity"),
            ("elbow_varus_moment", "Elbow varus moment"),
            ("shoulder_internal_rotation_moment", "Shoulder IR moment"),
        ],
    ),
]


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values with monotonicity enforcement."""
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

    within_sx = pair["source_within"].std(ddof=1)
    within_sy = pair["target_within"].std(ddof=1)
    if within_sx > 0 and within_sy > 0:
        within_x = pair["source_within"] / within_sx
        within_y = pair["target_within"] / within_sy
        within_model = sm.OLS(within_y, within_x).fit(
            cov_type="cluster", cov_kwds={"groups": pair["session"]}
        )
        within_cluster_p = float(within_model.pvalues.iloc[0])
    else:
        within_cluster_p = np.nan

    # Cluster-robust association on globally standardized values. This is kept
    # as a diagnostic alongside the primary within-session edge screen.
    sx = pair[source].std(ddof=1)
    sy = pair[target].std(ddof=1)
    if sx > 0 and sy > 0:
        x = (pair[source] - pair[source].mean()) / sx
        y = (pair[target] - pair[target].mean()) / sy
        model = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type="cluster", cov_kwds={"groups": pair["session"]}
        )
        cluster_beta = float(model.params.iloc[1])
        cluster_p = float(model.pvalues.iloc[1])
    else:
        cluster_beta = np.nan
        cluster_p = np.nan

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


def build_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    for stage_index, (stage, metrics) in enumerate(STAGES):
        for order, (column, label) in enumerate(metrics):
            node_rows.append(
                {
                    "column": column,
                    "label": label,
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_order": order,
                    "non_missing_n": int(df[column].notna().sum()),
                }
            )
    nodes = pd.DataFrame(node_rows)

    edge_rows = []
    for _, source in nodes.iterrows():
        for _, target in nodes.iterrows():
            if source.stage_index >= target.stage_index:
                continue
            stats = analyze_pair(df, source.column, target.column)
            edge_rows.append(
                {
                    "source": source.column,
                    "source_label": source.label,
                    "source_stage": source.stage,
                    "target": target.column,
                    "target_label": target.label,
                    "target_stage": target.stage,
                    **stats,
                }
            )
    edges = pd.DataFrame(edge_rows)
    edges["within_q"] = bh_fdr(edges["within_cluster_p"])
    edges["cluster_q"] = bh_fdr(edges["cluster_p"])
    edges["selected"] = (
        (edges["n"] >= MIN_N)
        & (edges["within_r"].abs() >= MIN_ABS_WITHIN_R)
        & (edges["within_q"] < FDR_ALPHA)
    )
    return nodes, edges


def plot_network(nodes: pd.DataFrame, selected: pd.DataFrame) -> None:
    stage_count = len(STAGES)
    max_nodes = max(len(metrics) for _, metrics in STAGES)
    positions: dict[str, tuple[float, float]] = {}
    fig, ax = plt.subplots(figsize=(22, 12), facecolor="#08101f")
    ax.set_facecolor("#08101f")

    for _, row in nodes.iterrows():
        stage_nodes = nodes[nodes.stage_index == row.stage_index]
        y_values = np.linspace(max_nodes - 1, 0, len(stage_nodes))
        y = float(y_values[int(row.stage_order)])
        x = float(row.stage_index * 4.1)
        positions[row.column] = (x, y)

    # Draw weaker edges first so stronger relationships remain visible.
    for _, edge in selected.sort_values("within_r", key=lambda s: s.abs()).iterrows():
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        color = "#50d6b1" if edge.within_r > 0 else "#f06f6f"
        width = 0.8 + 5.2 * min(abs(edge.within_r), 0.7)
        alpha = 0.35 + 0.55 * min(abs(edge.within_r) / 0.5, 1.0)
        ax.annotate(
            "",
            xy=(x2 - 0.85, y2),
            xytext=(x1 + 0.85, y1),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": width,
                "alpha": alpha,
                "connectionstyle": "arc3,rad=0.08",
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=1,
        )

    for _, row in nodes.iterrows():
        x, y = positions[row.column]
        ax.text(
            x,
            y,
            row.label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#edf4ff",
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": "#17243b",
                "edgecolor": "#58739a",
                "linewidth": 0.9,
            },
            zorder=3,
        )

    for stage_index, (stage, _) in enumerate(STAGES):
        ax.text(
            stage_index * 4.1,
            max_nodes + 0.35,
            stage,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#f0c96a",
        )

    legend = [
        Line2D([0], [0], color="#50d6b1", lw=3, label="Positive within-session association"),
        Line2D([0], [0], color="#f06f6f", lw=3, label="Negative within-session association"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=2, frameon=False, labelcolor="#dce8fa")
    ax.text(
        0,
        -1.35,
        (
            f"Exploratory associations only; arrows encode predefined phase order, not causality. "
            f"Selected: |within-session r| >= {MIN_ABS_WITHIN_R:.2f}, BH-FDR q < {FDR_ALPHA:.2f}, n >= {MIN_N}."
        ),
        color="#aab8cc",
        fontsize=9,
    )
    ax.set_xlim(-1.8, (stage_count - 1) * 4.1 + 1.8)
    ax.set_ylim(-1.65, max_nodes + 1.0)
    ax.axis("off")
    fig.suptitle(
        "POI Motion Relationship Network",
        color="#f5f8ff",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(POI_PATH)
    required = {"session"} | {column for _, metrics in STAGES for column, _ in metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required POI columns: {missing}")

    nodes, edges = build_results(df)
    selected = edges.loc[edges.selected].sort_values("within_r", key=lambda s: s.abs(), ascending=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUTPUT_DIR / "poi_motion_network_nodes.csv", index=False)
    edges.to_csv(OUTPUT_DIR / "poi_motion_network_all_edges.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "poi_motion_network_selected_edges.csv", index=False)

    summary = {
        "source": str(POI_PATH),
        "rows": int(len(df)),
        "sessions": int(df["session"].nunique()),
        "nodes": int(len(nodes)),
        "tested_time_ordered_pairs": int(len(edges)),
        "selected_edges": int(len(selected)),
        "selection": {
            "minimum_n": MIN_N,
            "minimum_absolute_within_session_r": MIN_ABS_WITHIN_R,
            "bh_fdr_q": FDR_ALPHA,
        },
        "direction_warning": "Arrows encode predefined phase order and are not estimated causal direction.",
    }
    (OUTPUT_DIR / "poi_motion_network_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_network(nodes, selected)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not selected.empty:
        print("\nSelected edges:")
        print(
            selected[
                ["source", "target", "n", "raw_r", "cluster_beta", "within_r", "within_q"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
