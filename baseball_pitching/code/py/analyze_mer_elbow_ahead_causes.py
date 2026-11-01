"""Why do some pitchers have the elbow ahead of the shoulder line at MER ("pushing", Tread)?

Builds on analyze_tread_plane_components_speed.py (run it first; reads its per-pitch output).

1. How much of the geometric elbow-ahead angle (upper arm vs glove->throwing shoulder line, in the
   plane perpendicular to the trunk axis) is glenohumeral horizontal adduction (shoulder_angle_x).
   A scapular/shoulder-girdle term was not computed: thorax_ap - thorax_prox runs almost parallel
   to the shoulder line, so thorax_ap cannot be used as an anterior axis without a definition.
2. Candidate causes from Tread's "7 causes of pushing" that this dataset can measure, screened at
   the pitcher-mean level (the speed signal was between-pitcher), plus how the elbow-ahead -> speed
   slope changes when each candidate is added.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "baseball_pitching" / "data"
HERE = Path(__file__).parent
BASE = HERE / "tread_plane_components_metrics.csv"

Y = "upperarm_in_plane_mer"
# Tread cause -> measurable proxy (POI unless noted).
CANDIDATES = {
    "torso_rotation_fp": "#4 flying open: torso rotation at FP",
    "rotation_hip_shoulder_separation_fp": "#4 flying open: hip-shoulder separation at FP",
    "shoulder_horizontal_abduction_fp": "arm position at FP: horizontal abduction",
    "shoulder_external_rotation_fp": "arm position at FP: external rotation",
    "max_shoulder_external_rotation": "#6/#7 layback: max external rotation",
    "elbow_flexion_mer": "#7 fake layback: elbow flexion at MER",
    "shoulder_abd_mer": "#1 elbow climbs: shoulder abduction at MER",
    "torso_anterior_tilt_mer": "#3 linear/flexion: torso anterior tilt at MER",
    "torso_lateral_tilt_mer": "torso lateral tilt at MER",
    "torso_rotation_mer": "torso rotation at MER",
    "torso_rotation_br": "torso rotation at BR",
    "torso_rotation_mer_to_br": "torso rotation change MER to BR",
    "glove_shoulder_horizontal_abduction_fp": "glove arm horizontal abduction at FP",
    "fp_to_mer_ms": "timing: FP to MER duration",
    "max_torso_rotational_velo": "torso rotational velocity peak",
    "session_height_m": "body height",
    "session_mass_kg": "body mass",
}
BODY = ("session_height_m", "session_mass_kg")
DERIVED = ("fp_to_mer_ms", "torso_rotation_mer_to_br")


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def main() -> None:
    base = pd.read_csv(BASE)
    poi_cols = ["session_pitch", "p_throws"] + [c for c in CANDIDATES if c not in base.columns and c not in DERIVED]
    poi = pd.read_csv(DATA / "poi" / "poi_metrics.csv", usecols=poi_cols)
    events = (pd.read_csv(DATA / "full_sig" / "joint_angles.csv", usecols=["session_pitch", "MER_time", "fp_poi_time"])
                .drop_duplicates())
    data = (base.merge(poi, on="session_pitch", validate="one_to_one")
                .merge(events, on="session_pitch", validate="one_to_one"))
    data["fp_to_mer_ms"] = (data["MER_time"] - data["fp_poi_time"]) * 1000
    data["torso_rotation_mer_to_br"] = data["torso_rotation_br"] - data["torso_rotation_mer"]
    if len(data) != 411 or data[list(CANDIDATES) + [Y]].isna().any().any():
        raise RuntimeError("Missing pitches or candidate values")
    print(f"n={len(data)}, sessions={data['session'].nunique()}, "
          f"left-handed pitches={int((data['p_throws'] == 'L').sum())}")

    gh = smf.ols(f"{Y} ~ shoulder_hab_mer", data=data).fit()
    print(f"\n{Y} ~ shoulder_hab_mer: R2={gh.rsquared:.3f}, slope={gh.params['shoulder_hab_mer']:+.3f}")

    pm = data.groupby("session")[list(CANDIDATES) + [Y, "upperarm_in_plane_br", "pitch_speed_mph"]].mean()
    base_fit = smf.ols(f"pitch_speed_mph ~ {Y} + session_height_m + session_mass_kg", data=pm).fit()
    print(f"\nPitcher-mean speed ~ elbow-ahead (+height/mass): beta={base_fit.params[Y]:+.3f} mph/deg, "
          f"p={base_fit.pvalues[Y]:.3g}")
    rows = []
    for c, label in CANDIDATES.items():
        r_y = stats.pearsonr(pm[c], pm[Y])
        within = smf.ols(f"{Y} ~ {c} + C(session)", data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["session"]})
        row = {"candidate": label, "r_elbow_ahead": r_y.statistic, "p": r_y.pvalue,
               "r_speed": stats.pearsonr(pm[c], pm["pitch_speed_mph"]).statistic,
               "within_slope": within.params[c], "within_p": within.pvalues[c]}
        if c not in BODY:
            adj = smf.ols(f"pitch_speed_mph ~ {Y} + {c} + session_height_m + session_mass_kg", data=pm).fit()
            row["elbow_ahead_beta_adj"] = adj.params[Y]
            row["elbow_ahead_p_adj"] = adj.pvalues[Y]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("p")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))

    picked = [c for c, label in CANDIDATES.items()
              if c not in BODY and table.set_index("candidate").loc[label, "p"] < 0.05]
    # torso_rotation_br and the MER->BR change are collinear with torso_rotation_mer; keep MER only.
    picked = [c for c in picked if c not in ("torso_rotation_br", "torso_rotation_mer_to_br")]
    mv = smf.ols(f"{Y} ~ " + " + ".join(f"zscore({c})" for c in picked), data=pm).fit()
    print(f"\nPitcher-mean model for elbow-ahead (standardized predictors): R2={mv.rsquared:.3f}, "
          f"adj R2={mv.rsquared_adj:.3f}")
    print(mv.summary().tables[1])

    print("\nTorso rotation at BR vs elbow-ahead and speed:")
    for c in ["torso_rotation_mer", "torso_rotation_br", "torso_rotation_mer_to_br"]:
        print(f"  {c}: pitch median={data[c].median():+.1f}, IQR=[{data[c].quantile(.25):+.1f}, "
              f"{data[c].quantile(.75):+.1f}]")
        for target in ["upperarm_in_plane_mer", "upperarm_in_plane_br", "pitch_speed_mph"]:
            r = stats.pearsonr(pm[c], pm[target])
            within = smf.ols(f"{target} ~ {c} + C(session)", data=data).fit(
                cov_type="cluster", cov_kwds={"groups": data["session"]})
            print(f"    vs {target}: pitcher-mean r={r.statistic:+.3f} p={r.pvalue:.3g}; "
                  f"within slope={within.params[c]:+.3f} p={within.pvalues[c]:.3g}")
    adj = smf.ols("pitch_speed_mph ~ torso_rotation_br + torso_rotation_mer + session_height_m + session_mass_kg",
                  data=pm).fit()
    print(f"  pitcher-mean speed ~ BR + MER torso rotation (+height/mass): "
          f"BR {adj.params['torso_rotation_br']:+.3f} p={adj.pvalues['torso_rotation_br']:.3g}; "
          f"MER {adj.params['torso_rotation_mer']:+.3f} p={adj.pvalues['torso_rotation_mer']:.3g}")


if __name__ == "__main__":
    main()
