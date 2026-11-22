"""Within-athlete check: on pitches where an athlete leans forward over the hips
faster than usual, are shoulder JFP / transfer and pitch speed higher?

Inputs (per pitch, produced upstream, FP->BR window, gate STP + JFP > 0):
  trunk_base_rel_pelvis_sources_outputs/per_pitch.csv
      vx_sagittal_lean        forward velocity of trunk base from sagittal lean (m/s)
      jfp_sagittal_lean       its JFP contribution (J)
      line_forward_lean_deg_br  pelvis->trunk-base line lean at BR (deg)
  shoulder_transfer_component_variance_outputs/per_pitch.csv
      jfp                     official JFP on gated frames (J)
      positive_total_j        positive shoulder transfer (J)
  POI pitch_speed_mph

Model per (predictor, outcome), 411 pitches / 100 athletes:
  y ~ x_within + x_between + (1 | session)
  x_between = athlete mean of x, x_within = x - athlete mean.
Coefficients reported per 1 within-athlete SD and per 1 between-athlete SD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "forward_lean_within_athlete_outputs"
PREDICTORS = ["vx_sagittal_lean", "jfp_sagittal_lean", "line_forward_lean_deg_br"]
OUTCOMES = ["pitch_speed_mph", "jfp", "positive_total_j"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    lean = pd.read_csv(HERE / "trunk_base_rel_pelvis_sources_outputs/per_pitch.csv",
                       usecols=["session_pitch", "session", *PREDICTORS, "pitch_speed_mph"])
    transfer = pd.read_csv(HERE / "shoulder_transfer_component_variance_outputs/per_pitch.csv",
                           usecols=["session_pitch", "jfp", "positive_total_j"])
    d = lean.merge(transfer, on="session_pitch", validate="one_to_one")
    if len(d) != 411 or d["session"].nunique() != 100:
        raise ValueError("Expected 411 pitches / 100 athletes")

    rows = []
    for x in PREDICTORS:
        d[f"{x}_between"] = d.groupby("session")[x].transform("mean")
        d[f"{x}_within"] = d[x] - d[f"{x}_between"]
        sd_w = d[f"{x}_within"].std()
        sd_b = d.groupby("session")[x].mean().std()
        for y in OUTCOMES:
            fit = smf.mixedlm(f"{y} ~ {x}_within + {x}_between", d, groups=d["session"]).fit(reml=True)
            for part, sd in [("within", sd_w), ("between", sd_b)]:
                term = f"{x}_{part}"
                lo, hi = fit.conf_int().loc[term]
                rows.append({
                    "predictor": x, "outcome": y, "level": part,
                    "predictor_sd": sd,
                    "effect_per_sd": fit.params[term] * sd,
                    "ci_low_per_sd": lo * sd, "ci_high_per_sd": hi * sd,
                    "p": fit.pvalues[term],
                })
            # within-athlete correlation as a plain descriptive check
            rows[-2]["within_r"] = float(np.corrcoef(d[f"{x}_within"], d[y] - d.groupby("session")[y].transform("mean"))[0, 1])
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "within_between_mixed_models.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
