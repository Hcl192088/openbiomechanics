"""Which limiting side of shoulder STP transfer tracks total transfer and pitch speed?

Per pitch, positive official shoulder STP transfer power
(`shoulder_energy_transfer_stp > 0`, with thorax/arm STP of opposite sign) is
split by which side has the smaller magnitude:

    thorax_stp = thorax_dist_seg_pwr + shoulder_energy_transfer_jfp
    arm_stp    = upper_arm_prox_seg_pwr - shoulder_energy_transfer_jfp
    thorax-limited: |thorax_stp| <= |arm_stp|

Energy is integrated from fp_poi_time to BR_time (and FP->MER as a window
sensitivity). Athlete level = pitches averaged within `session`.
Outcomes: official POI `shoulder_transfer_fp_br` and `pitch_speed_mph`.
Because both parts are components of total transfer, the main summary is a
covariance decomposition: share of between-athlete total-STP variance carried
by each part (shares sum to 100%).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("stp_limited_side_vs_total_transfer_outputs")
WINDOWS = {"fp_br": "BR_time", "fp_mer": "MER_time"}


def summarize(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("time")
    fp = group["fp_poi_time"].dropna().unique()
    out = {}
    for name, end_col in WINDOWS.items():
        end = group[end_col].dropna().unique()
        if len(fp) != 1 or len(end) != 1:
            raise ValueError(f"Ambiguous FP/{end_col} for {group.name}")
        w = group[(group["time"] >= fp[0]) & (group["time"] <= end[0])]
        t = w["time"].to_numpy(float)
        stp = w["shoulder_energy_transfer_stp"].to_numpy(float)
        jfp = w["shoulder_energy_transfer_jfp"].to_numpy(float)
        thorax_stp = w["thorax_dist_seg_pwr"].to_numpy(float) + jfp
        arm_stp = w["upper_arm_prox_seg_pwr"].to_numpy(float) - jfp
        valid = (thorax_stp * arm_stp < 0) & (stp > 0)
        thorax_limited = valid & (np.abs(thorax_stp) <= np.abs(arm_stp))
        arm_limited = valid & ~thorax_limited
        out[f"{name}_thorax_limited_j"] = float(np.trapezoid(np.where(thorax_limited, stp, 0.0), t))
        out[f"{name}_arm_limited_j"] = float(np.trapezoid(np.where(arm_limited, stp, 0.0), t))
    return pd.Series(out)


def partial_r(df: pd.DataFrame, y: str, x: str, controls: list[str]) -> float:
    design = np.column_stack([np.ones(len(df)), df[controls].to_numpy()])

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    return float(np.corrcoef(resid(df[x].to_numpy(float)), resid(df[y].to_numpy(float)))[0, 1])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cols = ["session_pitch", "time", "fp_poi_time", "MER_time", "BR_time",
            "shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp",
            "thorax_dist_seg_pwr", "upper_arm_prox_seg_pwr"]
    energy = pd.read_csv(ROOT / "data/full_sig/energy_flow.csv", usecols=cols)
    pitch = energy.groupby("session_pitch", sort=False).apply(summarize, include_groups=False).reset_index()

    poi = pd.read_csv(ROOT / "data/poi/poi_metrics.csv",
                      usecols=["session_pitch", "shoulder_transfer_fp_br", "pitch_speed_mph"])
    meta = pd.read_csv(ROOT / "data/metadata.csv", usecols=["session_pitch", "session", "session_mass_kg"])
    pitch = pitch.merge(poi, on="session_pitch", validate="one_to_one").merge(
        meta, on="session_pitch", validate="one_to_one")
    if len(pitch) != 411 or pitch["session"].nunique() != 100:
        raise ValueError(f"Expected 411 pitches / 100 sessions, got {len(pitch)} / {pitch['session'].nunique()}")
    pitch.to_csv(OUT / "per_pitch.csv", index=False)

    athlete = pitch.groupby("session").mean(numeric_only=True)
    athlete.to_csv(OUT / "per_athlete.csv")

    rows = []
    for win in WINDOWS:
        parts = {"thorax_limited": f"{win}_thorax_limited_j", "arm_limited": f"{win}_arm_limited_j"}
        stp_total = athlete[list(parts.values())].sum(axis=1)
        # residualize on body mass for the mass-adjusted covariance decomposition
        design = np.column_stack([np.ones(len(athlete)), athlete["session_mass_kg"]])
        res = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
        total_res = res(stp_total.to_numpy())
        for side, col in parts.items():
            part = athlete[col]
            rows.append({
                "window": win,
                "side": side,
                "mean_j": part.mean(),
                "share_of_mean_stp": part.mean() / stp_total.mean(),
                "sd_j": part.std(),
                "share_of_stp_variance": np.cov(part, stp_total)[0, 1] / stp_total.var(),
                "share_of_stp_variance_mass_adjusted": np.cov(res(part.to_numpy()), total_res)[0, 1] / total_res.var(ddof=1),
                "r_with_shoulder_transfer_fp_br": part.corr(athlete["shoulder_transfer_fp_br"]),
                "partial_r_shoulder_transfer_mass": partial_r(athlete, "shoulder_transfer_fp_br", col, ["session_mass_kg"]),
                "r_with_pitch_speed": part.corr(athlete["pitch_speed_mph"]),
                "partial_r_pitch_speed_mass": partial_r(athlete, "pitch_speed_mph", col, ["session_mass_kg"]),
            })
        rows.append({
            "window": win, "side": "parts_correlation",
            "mean_j": np.nan, "share_of_mean_stp": np.nan, "sd_j": np.nan,
            "share_of_stp_variance": np.nan, "share_of_stp_variance_mass_adjusted": np.nan,
            "r_with_shoulder_transfer_fp_br": athlete[parts["thorax_limited"]].corr(athlete[parts["arm_limited"]]),
            "partial_r_shoulder_transfer_mass": np.nan, "r_with_pitch_speed": np.nan,
            "partial_r_pitch_speed_mass": np.nan,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "limited_side_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
