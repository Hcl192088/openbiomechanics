"""Shoulder-hip separation vs pitch speed, split by height (< 6 ft vs >= 6 ft).

Questions:
1. How strong is the separation-speed association in each height group?
2. Is the difference between groups supported by a height-group x separation
   interaction test?
3. Full-sample pitch-level r, to reconcile values quoted in different notes.

Independent unit for inference: athlete (pitches averaged within `user`).
FP separation uses the POI field `rotation_hip_shoulder_separation_fp`, which is
the project FP (`fp_poi_time`). Source files are read only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_FILE = BASE_DIR / "data" / "metadata.csv"
POI_FILE = BASE_DIR / "data" / "poi" / "poi_metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "separation_speed_by_height_outputs"
HEIGHT_CUTOFF_M = 1.8288  # 6 ft
BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 20261122

OUTCOME = "pitch_speed_mph"
COVARIATES = ["session_height_m", "session_mass_kg", "age_yrs"]
EXPOSURES = {
    "separation_at_fp_poi": "rotation_hip_shoulder_separation_fp",
    "maximum_separation": "max_rotation_hip_shoulder_separation",
}


def load_data() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_FILE)
    poi = pd.read_csv(POI_FILE)
    needed_poi = ["session_pitch", OUTCOME, *EXPOSURES.values()]
    missing = [c for c in needed_poi if c not in poi.columns]
    if missing:
        raise KeyError(f"POI file missing columns: {missing}")
    meta = meta[["user", "session_pitch", *COVARIATES]]
    return poi[needed_poi].merge(meta, on="session_pitch", how="inner", validate="one_to_one")


def residualize(df: pd.DataFrame, col: str) -> np.ndarray:
    design = np.column_stack([np.ones(len(df)), df[COVARIATES].to_numpy()])
    beta, *_ = np.linalg.lstsq(design, df[col].to_numpy(), rcond=None)
    return df[col].to_numpy() - design @ beta


def partial_r(df: pd.DataFrame, x: str) -> float:
    return float(np.corrcoef(residualize(df, x), residualize(df, OUTCOME))[0, 1])


def bootstrap_ci(df: pd.DataFrame, x: str, rng: np.random.Generator) -> tuple[float, float]:
    xs = df[x].to_numpy()
    ys = df[OUTCOME].to_numpy()
    n = len(df)
    rs = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        rs[i] = np.corrcoef(xs[idx], ys[idx])[0, 1]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    pitches = load_data().dropna()

    athletes = pitches.groupby("user", as_index=False).mean(numeric_only=True)
    athletes["short"] = (athletes["session_height_m"] < HEIGHT_CUTOFF_M).astype(int)
    groups = {
        "under_6ft": athletes[athletes["short"] == 1],
        "6ft_and_over": athletes[athletes["short"] == 0],
        "all": athletes,
    }

    rows = []
    for exposure_name, x in EXPOSURES.items():
        rows.append(
            {
                "level": "pitch",
                "group": "all",
                "exposure": exposure_name,
                "n_athletes": pitches["user"].nunique(),
                "n_pitches": len(pitches),
                "r": pitches[x].corr(pitches[OUTCOME]),
                "r_ci_low": np.nan,
                "r_ci_high": np.nan,
                "partial_r_height_mass_age": partial_r(pitches, x),
            }
        )
        for group_name, df in groups.items():
            lo, hi = bootstrap_ci(df, x, rng)
            rows.append(
                {
                    "level": "athlete_mean",
                    "group": group_name,
                    "exposure": exposure_name,
                    "n_athletes": len(df),
                    "n_pitches": int(pitches["user"].isin(df["user"]).sum()),
                    "r": df[x].corr(df[OUTCOME]),
                    "r_ci_low": lo,
                    "r_ci_high": hi,
                    "partial_r_height_mass_age": partial_r(df, x),
                }
            )
    correlations = pd.DataFrame(rows)

    interaction_rows = []
    for exposure_name, x in EXPOSURES.items():
        df = athletes.copy()
        df["X"] = (df[x] - df[x].mean()) / df[x].std()
        fit = smf.ols(f"{OUTCOME} ~ X * short + " + " + ".join(COVARIATES), df).fit(cov_type="HC3")
        ci = fit.conf_int().loc["X:short"]
        interaction_rows.append(
            {
                "exposure": exposure_name,
                "n_athletes": len(df),
                "slope_mph_per_sd_6ft_and_over": fit.params["X"],
                "interaction_mph_per_sd_under_6ft_extra": fit.params["X:short"],
                "interaction_ci_low": ci[0],
                "interaction_ci_high": ci[1],
                "interaction_p_hc3": fit.pvalues["X:short"],
            }
        )
    interactions = pd.DataFrame(interaction_rows)

    correlations.to_csv(OUTPUT_DIR / "correlations_by_height.csv", index=False)
    interactions.to_csv(OUTPUT_DIR / "height_interaction.csv", index=False)
    pd.set_option("display.width", 200)
    print(correlations.round(3).to_string(index=False))
    print()
    print(interactions.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
