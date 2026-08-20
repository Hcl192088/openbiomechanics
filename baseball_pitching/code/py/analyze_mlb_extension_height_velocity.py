#!/usr/bin/env python3
"""
2025 MLB Statcast: same-height extension and four-seam velocity analysis.

The unit of analysis is one pitcher-season.  Statcast data are downloaded as
pitcher-level summaries for four-seam fastballs (FF), and player height is
joined by MLBAM player_id from the MLB Stats API.

Primary model:
    four-seam release speed ~ exact recorded height (inch fixed effects)
                              + release extension (feet)

This is deliberately a source-facing script: a failed download or a missing
required source field stops the run instead of silently substituting another
data source or definition.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "baseball_pitching" / "data"
IMG_DIR = ROOT / "baseball_pitching" / "imgs"
SAVANT_BASE = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_PLAYERS_BASE = "https://statsapi.mlb.com/api/v1/sports/1/players"


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "openbiomechanics-mlb-extension-analysis/1.0"})
    with urlopen(request, timeout=180) as response:
        return response.read()


def download_csv(url: str) -> pd.DataFrame:
    payload = download_bytes(url)
    if not payload:
        raise RuntimeError(f"Empty CSV response: {url}")
    return pd.read_csv(io.BytesIO(payload))


def download_json(url: str) -> dict:
    payload = download_bytes(url)
    if not payload:
        raise RuntimeError(f"Empty JSON response: {url}")
    return json.loads(payload.decode("utf-8"))


def build_savant_url(season: int, min_pitches: int) -> str:
    params = {
        "all": "true",
        "hfSea": f"{season}|",
        "hfGT": "R|",
        "hfPT": "FF|",
        "game_date_gt": f"{season}-03-01",
        "game_date_lt": f"{season}-11-01",
        "player_type": "pitcher",
        "group_by": "name",
        "min_pitches": str(min_pitches),
        "min_results": "0",
        "sort_col": "release_extension",
        "sort_order": "desc",
    }
    return f"{SAVANT_BASE}?{urlencode(params)}"


def parse_height_inches(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)['’]\s*(\d+)[\"”]?\s*", str(value))
    if not match:
        return np.nan
    feet, inches = (int(part) for part in match.groups())
    return float(feet * 12 + inches)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required source columns: {missing}")


def get_source_data(season: int, min_pitches: int) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    savant_url = build_savant_url(season, min_pitches)
    players_url = f"{MLB_PLAYERS_BASE}?season={season}"

    savant = download_csv(savant_url)
    require_columns(
        savant,
        {"player_id", "player_name", "pitches", "velocity", "release_extension"},
        "Baseball Savant CSV",
    )
    savant.to_csv(DATA_DIR / f"mlb_statcast_{season}_ff_pitcher_summary.csv", index=False)

    player_payload = download_json(players_url)
    if "people" not in player_payload or not isinstance(player_payload["people"], list):
        raise RuntimeError("MLB Stats API response has no usable people list")
    players = pd.json_normalize(player_payload["people"])
    require_columns(players, {"id", "fullName", "height", "primaryPosition.name"}, "MLB Stats API")
    players = players.rename(
        columns={
            "id": "player_id",
            "fullName": "statsapi_name",
            "height": "height_raw",
            "primaryPosition.name": "position",
        }
    )
    players["height_in"] = players["height_raw"].map(parse_height_inches)
    players[["player_id", "statsapi_name", "height_raw", "height_in", "position"]].to_csv(
        DATA_DIR / f"mlb_statsapi_{season}_players.csv", index=False
    )
    return savant, players, savant_url, players_url


def fit_model(data: pd.DataFrame, formula: str):
    return smf.ols(formula, data=data).fit(cov_type="HC3")


def model_row(model, term: str, label: str) -> dict[str, object]:
    ci = model.conf_int().loc[term]
    return {
        "model": label,
        "term": term,
        "n": int(model.nobs),
        "coefficient_mph": float(model.params[term]),
        "ci95_low_mph": float(ci.iloc[0]),
        "ci95_high_mph": float(ci.iloc[1]),
        "p_value": float(model.pvalues[term]),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
    }


def create_figure(data: pd.DataFrame, fixed_model, season: int, min_pitches: int) -> Path:
    observed_heights = sorted(data["height_in"].dropna().unique())
    reference_height = min(observed_heights, key=lambda value: abs(value - data["height_in"].median()))
    x_grid = np.linspace(data["extension_ft"].min(), data["extension_ft"].max(), 100)
    prediction_frame = pd.DataFrame({"height_in": reference_height, "extension_ft": x_grid})
    prediction = fixed_model.predict(prediction_frame)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(
        data["extension_ft"],
        data["fastball_velocity_mph"],
        c=data["height_in"],
        cmap="viridis",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.plot(
        x_grid,
        prediction,
        color="#c0392b",
        linewidth=2.2,
        label=f"Height fixed effect: {int(reference_height // 12)}' {int(reference_height % 12)}\"",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Recorded height (in)")
    ax.set_xlabel("Release extension (ft)")
    ax.set_ylabel("Four-seam release speed (mph)")
    ax.set_title(f"2025 MLB: extension and four-seam velocity\nFF pitches >= {min_pitches} per pitcher")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    output = IMG_DIR / "mlb_2025_extension_height_adjusted_velocity.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def run(season: int, min_pitches: int) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    savant, players, savant_url, players_url = get_source_data(season, min_pitches)
    savant["player_id"] = pd.to_numeric(savant["player_id"], errors="coerce")
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    merged = savant.merge(
        players[["player_id", "statsapi_name", "height_raw", "height_in", "position"]],
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    merged["pitches"] = pd.to_numeric(merged["pitches"], errors="coerce")
    merged["velocity"] = pd.to_numeric(merged["velocity"], errors="coerce")
    merged["release_extension"] = pd.to_numeric(merged["release_extension"], errors="coerce")
    merged = merged.rename(
        columns={
            "pitches": "ff_pitch_count",
            "velocity": "fastball_velocity_mph",
            "release_extension": "extension_ft",
        }
    )

    analysis_columns = [
        "player_id",
        "player_name",
        "statsapi_name",
        "position",
        "height_raw",
        "height_in",
        "ff_pitch_count",
        "fastball_velocity_mph",
        "extension_ft",
    ]
    analysis = merged[analysis_columns].copy()
    analysis["height_in"] = pd.to_numeric(analysis["height_in"], errors="coerce")
    required = ["height_in", "ff_pitch_count", "fastball_velocity_mph", "extension_ft"]
    missing_counts = analysis[required].isna().sum().to_dict()
    usable = analysis.dropna(subset=required).copy()
    if usable.empty:
        raise RuntimeError(f"No usable merged rows. Missing counts: {missing_counts}")
    if usable["height_in"].nunique() < 2:
        raise RuntimeError("Fewer than two recorded height values remain after merge")

    analysis.to_csv(DATA_DIR / f"mlb_{season}_ff_extension_height_velocity_merged.csv", index=False)

    fixed_model = fit_model(
        usable,
        "fastball_velocity_mph ~ C(height_in) + extension_ft",
    )
    linear_model = fit_model(
        usable,
        "fastball_velocity_mph ~ height_in + extension_ft",
    )
    fixed_result = model_row(fixed_model, "extension_ft", "exact_height_fixed_effect")
    linear_result = model_row(linear_model, "extension_ft", "linear_height_adjustment")

    q1, q3 = usable["extension_ft"].quantile([0.25, 0.75])
    fixed_result["extension_iqr_ft"] = float(q3 - q1)
    fixed_result["predicted_q3_minus_q1_mph"] = float(fixed_model.params["extension_ft"] * (q3 - q1))
    fixed_result["height_groups"] = int(usable["height_in"].nunique())
    fixed_result["height_groups_with_two_or_more_pitchers"] = int(
        (usable.groupby("height_in").size() >= 2).sum()
    )

    results = pd.DataFrame([fixed_result, linear_result])
    results.to_csv(DATA_DIR / f"mlb_{season}_extension_height_velocity_results.csv", index=False)
    figure = create_figure(usable, fixed_model, season, min_pitches)

    report = {
        "season": season,
        "regular_season_only": True,
        "pitch_type": "FF",
        "minimum_ff_pitches_per_pitcher": min_pitches,
        "savant_rows_downloaded": int(len(savant)),
        "merged_rows": int(len(analysis)),
        "usable_pitchers": int(len(usable)),
        "missing_counts_before_drop": {key: int(value) for key, value in missing_counts.items()},
        "height_groups": int(usable["height_in"].nunique()),
        "height_groups_with_two_or_more_pitchers": int(
            (usable.groupby("height_in").size() >= 2).sum()
        ),
        "extension_q1_ft": float(q1),
        "extension_q3_ft": float(q3),
        "primary_model": fixed_result,
        "sensitivity_model": linear_result,
        "savant_url": savant_url,
        "mlb_statsapi_url": players_url,
        "figure": str(figure.relative_to(ROOT)),
    }
    with (DATA_DIR / f"mlb_{season}_extension_height_velocity_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--min-pitches", type=int, default=100)
    args = parser.parse_args()
    if args.min_pitches <= 0:
        raise SystemExit("--min-pitches must be positive")
    run(args.season, args.min_pitches)


if __name__ == "__main__":
    main()
