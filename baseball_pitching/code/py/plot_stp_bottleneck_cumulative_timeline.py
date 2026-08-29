"""Plot cumulative STP energy by limiting side over normalized FP-BR time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid


ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = Path(__file__).resolve().with_name("stp_bottleneck_cumulative_timeline.csv")
OUT_FIG = ROOT / "imgs" / "stp_bottleneck_cumulative_timeline.png"
GRID = np.linspace(0, 1, 101)


def thorax_probability(series: pd.Series) -> float:
    valid = series.notna().sum()
    return float((series == 1).sum() / valid) if valid else np.nan


def summarize(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time")
    fp, mer, br = (float(group[c].iloc[0]) for c in ["fp_poi_time", "MER_time", "BR_time"])
    w = group[(group.time >= fp) & (group.time <= br)].copy()
    phase = (w.time.to_numpy(float) - fp) / (br - fp)
    jfp = w.shoulder_energy_transfer_jfp.to_numpy(float)
    thorax = w.thorax_dist_seg_pwr.to_numpy(float) + jfp
    arm = w.upper_arm_prox_seg_pwr.to_numpy(float) - jfp
    stp = w.shoulder_energy_transfer_stp.to_numpy(float)
    active = (thorax * arm < 0) & (stp > 0)
    thorax_limited = active & (np.abs(thorax) <= np.abs(arm))
    arm_limited = active & ~thorax_limited
    thorax_power = np.where(thorax_limited, stp, 0)
    arm_power = np.where(arm_limited, stp, 0)
    thorax_cum = cumulative_trapezoid(thorax_power, w.time, initial=0)
    arm_cum = cumulative_trapezoid(arm_power, w.time, initial=0)
    torso_speed = np.abs(w["torso_velo_z"].to_numpy(float))
    torso_peak_phase = float(phase[np.argmax(torso_speed)])

    nearest = np.abs(phase[:, None] - GRID[None, :]).argmin(axis=0)
    state = np.where(thorax_limited, 1.0, np.where(arm_limited, -1.0, np.nan))
    return pd.DataFrame({
        "phase": GRID,
        "thorax_cumulative_j": np.interp(GRID, phase, thorax_cum),
        "arm_cumulative_j": np.interp(GRID, phase, arm_cum),
        "limiting_state": state[nearest],
        "mer_phase": (mer - fp) / (br - fp),
        "torso_peak_phase": torso_peak_phase,
    })


def main() -> None:
    columns = [
        "session_pitch", "time", "fp_poi_time", "MER_time", "BR_time",
        "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr",
        "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
    ]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=columns).dropna()
    velocity = pd.read_csv(
        ROOT / "data/full_sig/joint_velos.csv",
        usecols=["session_pitch", "time", *[f"torso_velo_{a}" for a in "xyz"]],
    ).dropna()
    energy = energy.merge(velocity, on=["session_pitch", "time"], validate="one_to_one")
    rows = []
    for pitch, group in energy.groupby("session_pitch", sort=False):
        part = summarize(group)
        part["session_pitch"] = pitch
        rows.append(part)
    pitch_curve = pd.concat(rows, ignore_index=True)
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session"])
    pitch_curve = pitch_curve.merge(meta, on="session_pitch", validate="many_to_one")
    session_curve = pitch_curve.groupby(["session", "phase"], as_index=False).agg(
        thorax_cumulative_j=("thorax_cumulative_j", "mean"),
        arm_cumulative_j=("arm_cumulative_j", "mean"),
        thorax_limiting_probability=("limiting_state", thorax_probability),
    )
    curve = session_curve.groupby("phase", as_index=False).agg(
        thorax_cumulative_j=("thorax_cumulative_j", "mean"),
        arm_cumulative_j=("arm_cumulative_j", "mean"),
        thorax_limiting_probability=("thorax_limiting_probability", "mean"),
    )
    curve["arm_cumulative_signed_j"] = -curve.arm_cumulative_j
    curve["signed_bottleneck_balance_j"] = (
        curve.thorax_cumulative_j - curve.arm_cumulative_j
    )
    curve.to_csv(OUT_CSV, index=False)

    mer_by_pitch = pitch_curve.groupby("session_pitch").mer_phase.first()
    mer_median = float(mer_by_pitch.median())
    mer_q1, mer_q3 = (float(mer_by_pitch.quantile(q)) for q in [.25, .75])
    torso_peak_by_pitch = pitch_curve.groupby("session_pitch").torso_peak_phase.first()
    torso_peak_median = float(torso_peak_by_pitch.median())
    torso_peak_q1, torso_peak_q3 = (
        float(torso_peak_by_pitch.quantile(q)) for q in [.25, .75]
    )
    transition = pd.read_csv(
        Path(__file__).resolve().with_name("stp_final_bottleneck_transition_outputs") / "per_pitch.csv"
    )
    transition_median = float(
        transition.loc[transition.eligible_final_A_to_T, "final_t_phase_fp_br"].median()
    )

    x = curve.phase.to_numpy(float) * 100
    thorax = curve.thorax_cumulative_j.to_numpy(float)
    arm_signed = curve.arm_cumulative_signed_j.to_numpy(float)
    total_cumulative = thorax - arm_signed
    probability = curve.thorax_limiting_probability.to_numpy(float)

    fig, ax = plt.subplots(figsize=(12, 7))
    points = np.column_stack([x, total_cumulative])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_probability = np.nanmean(
        np.column_stack([probability[:-1], probability[1:]]), axis=1
    )
    colors = np.where(segment_probability >= .5, "#d95f02", "#1b9e77")
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=4))
    ax.autoscale_view()
    ax.axvspan(torso_peak_q1 * 100, torso_peak_q3 * 100, color="0.45", alpha=.10,
               label="Peak torso rotation speed IQR")
    ax.axvline(torso_peak_median * 100, color="0.35", linestyle=":", linewidth=2,
               label=f"Peak torso rotation speed median ({torso_peak_median*100:.1f}%)")
    ax.axvline(transition_median * 100, color="#6a3d9a", linestyle="--", linewidth=2,
               label=f"Final sustained A→T transition ({transition_median*100:.1f}%)")
    ax.axvspan(mer_q1 * 100, mer_q3 * 100, color="#377eb8", alpha=.12, label="MER IQR")
    ax.axvline(mer_median * 100, color="#377eb8", linestyle="-.", linewidth=2,
               label=f"MER median ({mer_median*100:.1f}%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Normalized FP→BR time (%)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean cumulative positive STP energy (J)")
    ax.set_title("Shoulder STP cumulative energy and limiting side (100 pitchers)\nGreen: upper-arm bottleneck majority; orange: thorax bottleneck majority")
    ymin = ax.get_ylim()[0]
    ax.text(0, ymin, " FP", va="bottom", ha="left", fontweight="bold")
    ax.text(100, ymin, "BR ", va="bottom", ha="right", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles = [
        Line2D([0], [0], color="#1b9e77", lw=4, label="Upper-arm bottleneck majority"),
        Line2D([0], [0], color="#d95f02", lw=4, label="Thorax bottleneck majority"),
        *handles,
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"coverage: pitches={pitch_curve.session_pitch.nunique()}, sessions={session_curve.session.nunique()}")
    print(f"transition={transition_median*100:.2f}%, MER={mer_median*100:.2f}% [{mer_q1*100:.2f}, {mer_q3*100:.2f}]")
    print(f"peak torso rotation speed={torso_peak_median*100:.2f}% [{torso_peak_q1*100:.2f}, {torso_peak_q3*100:.2f}]")
    print(f"final total cumulative={total_cumulative[-1]:.2f} J; thorax-limited={thorax[-1]:.2f} J; arm-limited={-arm_signed[-1]:.2f} J")
    print(f"saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
