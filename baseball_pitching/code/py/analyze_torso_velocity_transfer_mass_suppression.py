"""Does body mass mask the torso-velocity vs energy-transfer relationship?

Athlete-level (pitches averaged within `user`) correlations between
`max_torso_rotational_velo` and FP-BR energy transfer outcomes, raw and
partial (controlling body mass, and body mass + height). POI transfer fields
use the official FP->BR window. Source files are read only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
POI_FILE = BASE_DIR / "data" / "poi" / "poi_metrics.csv"
METADATA_FILE = BASE_DIR / "data" / "metadata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "torso_velocity_transfer_mass_suppression_outputs"

PREDICTOR = "max_torso_rotational_velo"
OUTCOMES = [
    "shoulder_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
    "pelvis_lumbar_transfer_fp_br",
    "shoulder_transfer_per_kg",
    "pitch_speed_mph",
]
SIZE = ["session_mass_kg", "session_height_m"]


def partial_r(df: pd.DataFrame, y: str, x: str, controls: list[str]) -> float:
    design = np.column_stack([np.ones(len(df)), df[controls].to_numpy()])

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    return float(np.corrcoef(resid(df[x].to_numpy()), resid(df[y].to_numpy()))[0, 1])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    poi = pd.read_csv(POI_FILE)
    meta = pd.read_csv(METADATA_FILE)[["session_pitch", "user", *SIZE]]
    needed = [PREDICTOR, "shoulder_transfer_fp_br", "thorax_distal_transfer_fp_br",
              "pelvis_lumbar_transfer_fp_br", "pitch_speed_mph"]
    missing = [c for c in needed if c not in poi.columns]
    if missing:
        raise KeyError(f"POI file missing columns: {missing}")

    data = poi[["session_pitch", *needed]].merge(meta, on="session_pitch", how="inner", validate="one_to_one")
    athletes = data.groupby("user").mean(numeric_only=True).dropna()
    athletes["shoulder_transfer_per_kg"] = athletes["shoulder_transfer_fp_br"] / athletes["session_mass_kg"]

    rows = [
        {"outcome": f"size:{s}", "n_athletes": len(athletes), "r": athletes[PREDICTOR].corr(athletes[s]),
         "partial_r_mass": np.nan, "partial_r_mass_height": np.nan}
        for s in SIZE
    ]
    for y in OUTCOMES:
        rows.append(
            {
                "outcome": y,
                "n_athletes": len(athletes),
                "r": athletes[PREDICTOR].corr(athletes[y]),
                "partial_r_mass": partial_r(athletes, y, PREDICTOR, ["session_mass_kg"]),
                "partial_r_mass_height": partial_r(athletes, y, PREDICTOR, SIZE),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "torso_velocity_transfer_correlations.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
