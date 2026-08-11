from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "baseball_pitching" / "data"
OUTPUT = Path(__file__).with_name("shoulder_horizontal_abduction_br_speed_results.csv")


def sample_at_br(group: pd.DataFrame) -> float:
    br = group["BR_time"].iloc[0]
    if pd.isna(br):
        return np.nan
    values = group[["time", "shoulder_angle_x"]].dropna().sort_values("time")
    if len(values) < 2 or br < values["time"].iloc[0] or br > values["time"].iloc[-1]:
        return np.nan
    return float(np.interp(br, values["time"], values["shoulder_angle_x"]))


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    poi_path = DATA / "poi" / "poi_metrics.csv"
    joint_path = DATA / "full_sig" / "joint_angles.csv"
    poi_cols = [
        "session_pitch",
        "session",
        "pitch_speed_mph",
        "shoulder_horizontal_abduction_fp",
    ]
    joint_cols = ["session_pitch", "time", "BR_time", "shoulder_angle_x"]

    poi = pd.read_csv(poi_path, usecols=poi_cols)
    joint = pd.read_csv(joint_path, usecols=joint_cols)
    br_angle = (
        joint.groupby("session_pitch", sort=False)
        .apply(sample_at_br, include_groups=False)
        .rename("shoulder_horizontal_abduction_br")
        .reset_index()
    )
    data = poi.merge(br_angle, on="session_pitch", how="left", validate="one_to_one").dropna()
    if len(data) != 411:
        raise RuntimeError(f"Expected 411 complete pitches, found {len(data)}")

    data["br_group"] = np.where(
        data["shoulder_horizontal_abduction_br"] > 0,
        "BR horizontal abduction (>0 deg)",
        "BR neutral/adduction (<=0 deg)",
    )
    abd = data.loc[data["shoulder_horizontal_abduction_br"] > 0, "pitch_speed_mph"]
    non_abd = data.loc[data["shoulder_horizontal_abduction_br"] <= 0, "pitch_speed_mph"]
    if min(len(abd), len(non_abd)) < 2:
        raise RuntimeError("The predefined 0-degree split produced an insufficient group")

    pearson = stats.pearsonr(data["shoulder_horizontal_abduction_br"], data["pitch_speed_mph"])
    welch = stats.ttest_ind(abd, non_abd, equal_var=False)
    mann_whitney = stats.mannwhitneyu(abd, non_abd, alternative="two-sided")

    # Cluster-robust inference accounts for repeated pitches from the same pitcher/session.
    continuous = smf.ols(
        "pitch_speed_mph ~ shoulder_horizontal_abduction_br", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": data["session"]})
    adjusted = smf.ols(
        "pitch_speed_mph ~ I(shoulder_horizontal_abduction_br > 0) + shoulder_horizontal_abduction_fp",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["session"]})
    raw_group_clustered = smf.ols(
        "pitch_speed_mph ~ I(shoulder_horizontal_abduction_br > 0)", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": data["session"]})

    # Pitcher fixed effects isolate within-pitcher differences between pitches.
    within = smf.ols(
        "pitch_speed_mph ~ I(shoulder_horizontal_abduction_br > 0) + shoulder_horizontal_abduction_fp + C(session)",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["session"]})

    fp_abducted = data.loc[data["shoulder_horizontal_abduction_fp"] > 0].copy()
    maintained = smf.ols(
        "pitch_speed_mph ~ I(shoulder_horizontal_abduction_br > 0) + shoulder_horizontal_abduction_fp",
        data=fp_abducted,
    ).fit(cov_type="cluster", cov_kwds={"groups": fp_abducted["session"]})

    threshold_mph = 140.0 / 1.609344
    fast = data.loc[data["pitch_speed_mph"] >= threshold_mph].copy()
    if fast.empty:
        raise RuntimeError("No pitches met the predefined 140 km/h threshold")

    group_summary = (
        data.groupby("br_group", sort=False)["pitch_speed_mph"]
        .agg(n="size", mean_mph="mean", sd_mph="std", median_mph="median")
        .reset_index()
    )
    group_summary.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    group_term = "I(shoulder_horizontal_abduction_br > 0)[T.True]"
    print(f"n pitches={len(data)}, n sessions={data['session'].nunique()}")
    print(group_summary.to_string(index=False))
    print(f"BR angle mean={data['shoulder_horizontal_abduction_br'].mean():.3f}, "
          f"SD={data['shoulder_horizontal_abduction_br'].std():.3f}, "
          f"range=[{data['shoulder_horizontal_abduction_br'].min():.3f}, "
          f"{data['shoulder_horizontal_abduction_br'].max():.3f}]")
    print(f"Pearson r={pearson.statistic:.4f}, p={pearson.pvalue:.6g}")
    print(f"Cluster-robust continuous beta={continuous.params['shoulder_horizontal_abduction_br']:.4f} mph/deg, "
          f"p={continuous.pvalues['shoulder_horizontal_abduction_br']:.6g}, "
          f"95% CI={tuple(continuous.conf_int().loc['shoulder_horizontal_abduction_br'].round(4))}")
    print(f"Raw group difference={abd.mean() - non_abd.mean():.4f} mph, "
          f"Welch p={welch.pvalue:.6g}, Cohen d={cohens_d(abd, non_abd):.4f}, "
          f"Mann-Whitney p={mann_whitney.pvalue:.6g}")
    print(f"Raw group cluster-robust difference={raw_group_clustered.params[group_term]:.4f} mph, "
          f"p={raw_group_clustered.pvalues[group_term]:.6g}, "
          f"95% CI={tuple(raw_group_clustered.conf_int().loc[group_term].round(4))}")
    print(f"Adjusted group difference={adjusted.params[group_term]:.4f} mph, "
          f"p={adjusted.pvalues[group_term]:.6g}, "
          f"95% CI={tuple(adjusted.conf_int().loc[group_term].round(4))}")
    print(f"Within-pitcher adjusted difference={within.params[group_term]:.4f} mph, "
          f"p={within.pvalues[group_term]:.6g}, "
          f"95% CI={tuple(within.conf_int().loc[group_term].round(4))}")
    print(f"FP-abducted subset n={len(fp_abducted)} "
          f"(maintained={int((fp_abducted['shoulder_horizontal_abduction_br'] > 0).sum())}, "
          f"not maintained={int((fp_abducted['shoulder_horizontal_abduction_br'] <= 0).sum())})")
    print(f"FP-abducted maintained adjusted difference={maintained.params[group_term]:.4f} mph, "
          f"p={maintained.pvalues[group_term]:.6g}, "
          f"95% CI={tuple(maintained.conf_int().loc[group_term].round(4))}")
    print(f">=140 km/h threshold={threshold_mph:.5f} mph, n={len(fast)}, "
          f"sessions={fast['session'].nunique()}")
    print(f">=140 km/h BR angle mean={fast['shoulder_horizontal_abduction_br'].mean():.4f}, "
          f"SD={fast['shoulder_horizontal_abduction_br'].std():.4f}, "
          f"median={fast['shoulder_horizontal_abduction_br'].median():.4f}, "
          f"IQR=[{fast['shoulder_horizontal_abduction_br'].quantile(.25):.4f}, "
          f"{fast['shoulder_horizontal_abduction_br'].quantile(.75):.4f}], "
          f"range=[{fast['shoulder_horizontal_abduction_br'].min():.4f}, "
          f"{fast['shoulder_horizontal_abduction_br'].max():.4f}]")
    print(f">=140 km/h BR abduction >0 n={int((fast['shoulder_horizontal_abduction_br'] > 0).sum())}, "
          f"proportion={(fast['shoulder_horizontal_abduction_br'] > 0).mean():.4%}")


if __name__ == "__main__":
    main()
