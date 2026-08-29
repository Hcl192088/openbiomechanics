"""Plot cumulative STP energy by limiting side over normalized FP-BR time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    nearest = np.abs(phase[:, None] - GRID[None, :]).argmin(axis=0)
    state = np.where(thorax_limited, 1.0, np.where(arm_limited, -1.0, np.nan))
    return pd.DataFrame({
        "phase": GRID,
        "thorax_cumulative_j": np.interp(GRID, phase, thorax_cum),
        "arm_cumulative_j": np.interp(GRID, phase, arm_cum),
        "limiting_state": state[nearest],
        "mer_phase": (mer - fp) / (br - fp),
    })


def main() -> None:
    columns = [
        "session_pitch", "time", "fp_poi_time", "MER_time", "BR_time",
        "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr",
        "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
    ]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=columns).dropna()
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
    transition = pd.read_csv(
        Path(__file__).resolve().with_name("stp_final_bottleneck_transition_outputs") / "per_pitch.csv"
    )
    transition_median = float(
        transition.loc[transition.eligible_final_A_to_T, "final_t_phase_fp_br"].median()
    )

    x = curve.phase.to_numpy(float) * 100
    thorax = curve.thorax_cumulative_j.to_numpy(float)
    arm_signed = curve.arm_cumulative_signed_j.to_numpy(float)
    balance = curve.signed_bottleneck_balance_j.to_numpy(float)
    balance_min_index = int(np.nanargmin(balance))
    balance_min_phase = x[balance_min_index]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, balance, color="black", linewidth=3,
            label="Signed cumulative bottleneck energy")
    ax.axhline(0, color="0.35", linewidth=.8)
    ymin, ymax = ax.get_ylim()
    ax.axvline(transition_median * 100, color="#6a3d9a", linestyle="--", linewidth=2,
               label=f"Final sustained A→T transition ({transition_median*100:.1f}%)")
    ax.scatter([balance_min_phase], [balance[balance_min_index]], color="black", s=42, zorder=5)
    ax.annotate(
        f"Cumulative balance turns upward\n({balance_min_phase:.0f}%)",
        (balance_min_phase, balance[balance_min_index]), xytext=(-125, 28),
        textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=9,
    )
    ax.axvspan(mer_q1 * 100, mer_q3 * 100, color="#377eb8", alpha=.12, label="MER IQR")
    ax.axvline(mer_median * 100, color="#377eb8", linestyle="-.", linewidth=2,
               label=f"MER median ({mer_median*100:.1f}%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Normalized FP→BR time (%)")
    ax.set_ylabel("Signed cumulative bottleneck STP energy (J)")
    ax.set_title("Shoulder STP cumulative bottleneck balance (100 pitchers)\nDownward slope: upper-arm bottleneck; upward slope: thorax bottleneck")
    ax.text(0, ymin, " FP", va="bottom", ha="left", fontweight="bold")
    ax.text(100, ymin, "BR ", va="bottom", ha="right", fontweight="bold")
    ax.annotate("Upper-arm bottleneck dominates added STP\n(line slopes downward)",
                (35, np.interp(35, x, balance)), xytext=(-80, 38),
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#1b9e77"})
    ax.annotate("Thorax bottleneck dominates added STP\n(line slopes upward)",
                (84, np.interp(84, x, balance)), xytext=(20, -35),
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#d95f02"})
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"coverage: pitches={pitch_curve.session_pitch.nunique()}, sessions={session_curve.session.nunique()}")
    print(f"transition={transition_median*100:.2f}%, MER={mer_median*100:.2f}% [{mer_q1*100:.2f}, {mer_q3*100:.2f}]")
    print(f"cumulative balance minimum={balance_min_phase:.2f}%")
    print(f"final thorax cumulative={thorax[-1]:.2f} J, arm cumulative={-arm_signed[-1]:.2f} J, balance={balance[-1]:.2f} J")
    print(f"saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
