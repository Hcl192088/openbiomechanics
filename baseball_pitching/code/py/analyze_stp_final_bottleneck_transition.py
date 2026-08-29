"""Analyze the last sustained upper-arm-to-thorax STP bottleneck transition."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("stp_final_bottleneck_transition_outputs")


def integrate_positive(time: np.ndarray, power: np.ndarray) -> float:
    return float(np.trapezoid(np.maximum(power, 0), time))


def summarize(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time").copy()
    fp, mer, br = (float(group[c].iloc[0]) for c in ["fp_poi_time", "MER_time", "BR_time"])
    w = group[(group.time >= fp) & (group.time <= br)].copy()
    jfp = w.shoulder_energy_transfer_jfp
    thorax = w.thorax_dist_seg_pwr + jfp
    arm = w.upper_arm_prox_seg_pwr - jfp
    active = (thorax * arm < 0) & (w.shoulder_energy_transfer_stp > 0)
    a = w.loc[active].copy()
    a["state"] = np.where(
        thorax.loc[active].abs() <= arm.loc[active].abs(), "T", "A"
    )
    if a.empty:
        raise ValueError(f"No active STP frames for {group.name}")
    states = a.state.to_numpy()
    run_start = np.r_[True, states[1:] != states[:-1]]
    starts = np.flatnonzero(run_start)
    run_states = states[starts]
    final_t_start = np.nan
    eligible = run_states[-1] == "T" and np.any(run_states[:-1] == "A")
    if eligible:
        final_t_start = float(a.time.iloc[starts[-1]])

    t = w.time.to_numpy(float)
    stp = w.shoulder_energy_transfer_stp.to_numpy(float)
    fp_mer = w.time <= mer
    total_energy = integrate_positive(t, stp)
    active_array = active.to_numpy(float)
    active_duration = float(np.trapezoid(active_array, t))
    before_energy = after_energy = np.nan
    if eligible:
        before = t <= final_t_start
        after = t >= final_t_start
        before_energy = integrate_positive(t[before], stp[before]) if before.sum() >= 2 else 0
        after_energy = integrate_positive(t[after], stp[after]) if after.sum() >= 2 else 0
    return pd.Series({
        "eligible_final_A_to_T": eligible,
        "run_pattern": "->".join(run_states),
        "switch_count": len(run_states) - 1,
        "final_t_start_time": final_t_start,
        "final_t_phase_fp_br": (final_t_start - fp) / (br - fp) if eligible else np.nan,
        "final_t_minus_mer_ms": (final_t_start - mer) * 1000 if eligible else np.nan,
        "final_t_before_mer": final_t_start <= mer if eligible else np.nan,
        "stp_positive_fp_br_j": total_energy,
        "stp_active_duration_s": active_duration,
        "stp_active_mean_power_w": total_energy / active_duration,
        "stp_positive_fp_mer_j": integrate_positive(t[fp_mer], stp[fp_mer]),
        "stp_before_final_t_j": before_energy,
        "stp_after_final_t_j": after_energy,
        "stp_after_final_t_share": after_energy / total_energy if eligible and total_energy > 0 else np.nan,
    })


def partial_result(data: pd.DataFrame, outcome: str) -> dict:
    clean = data[["final_t_phase_fp_br", outcome, "session_mass_kg"]].dropna()
    x = sm.add_constant(clean[["session_mass_kg"]])
    rx = sm.OLS(clean.final_t_phase_fp_br, x).fit().resid
    ry = sm.OLS(clean[outcome], x).fit().resid
    r = float(np.corrcoef(rx, ry)[0, 1])
    fit = sm.OLS(ry, sm.add_constant(rx)).fit()
    return {"outcome": outcome, "n": len(clean), "partial_r": r, "p": float(fit.pvalues.iloc[1])}


def main() -> None:
    columns = [
        "session_pitch", "time", "fp_poi_time", "MER_time", "BR_time",
        "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr",
        "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
    ]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=columns).dropna()
    pitch = energy.groupby("session_pitch", sort=False).apply(
        summarize, include_groups=False
    ).reset_index()
    meta = pd.read_csv(
        ROOT / "data/metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    pitch = pitch.merge(meta, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411 or pitch.session.nunique() != 100:
        raise ValueError("Expected 411 pitches and 100 sessions")
    eligible = pitch[pitch.eligible_final_A_to_T].copy()
    athlete = eligible.groupby("session", as_index=False).agg(
        session_mass_kg=("session_mass_kg", "first"),
        eligible_pitch_n=("session_pitch", "size"),
        final_t_phase_fp_br=("final_t_phase_fp_br", "mean"),
        final_t_minus_mer_ms=("final_t_minus_mer_ms", "mean"),
        final_t_before_mer_fraction=("final_t_before_mer", "mean"),
        stp_positive_fp_br_j=("stp_positive_fp_br_j", "mean"),
        stp_positive_fp_mer_j=("stp_positive_fp_mer_j", "mean"),
        stp_active_duration_s=("stp_active_duration_s", "mean"),
        stp_active_mean_power_w=("stp_active_mean_power_w", "mean"),
        stp_after_final_t_share=("stp_after_final_t_share", "mean"),
    )
    partial = pd.DataFrame([
        partial_result(athlete, "stp_positive_fp_br_j"),
        partial_result(athlete, "stp_positive_fp_mer_j"),
        partial_result(athlete, "stp_active_duration_s"),
        partial_result(athlete, "stp_active_mean_power_w"),
        partial_result(athlete, "stp_after_final_t_share"),
    ])
    OUT.mkdir(exist_ok=True)
    pitch.to_csv(OUT / "per_pitch.csv", index=False)
    athlete.to_csv(OUT / "per_pitcher.csv", index=False)
    partial.to_csv(OUT / "partial_correlations.csv", index=False)
    print(f"coverage: pitches={len(pitch)}, sessions={pitch.session.nunique()}")
    print(f"eligible pitches={len(eligible)} ({len(eligible)/len(pitch):.1%}), eligible sessions={len(athlete)}")
    print("final sustained thorax transition phase median/IQR:",
          eligible.final_t_phase_fp_br.median(),
          eligible.final_t_phase_fp_br.quantile(.25),
          eligible.final_t_phase_fp_br.quantile(.75))
    print("relative to MER ms median/IQR:",
          eligible.final_t_minus_mer_ms.median(),
          eligible.final_t_minus_mer_ms.quantile(.25),
          eligible.final_t_minus_mer_ms.quantile(.75))
    print("before MER:", eligible.final_t_before_mer.mean())
    print("STP remaining share median/IQR:",
          eligible.stp_after_final_t_share.median(),
          eligible.stp_after_final_t_share.quantile(.25),
          eligible.stp_after_final_t_share.quantile(.75))
    print("\nPartial correlations controlling mass")
    print(partial.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
