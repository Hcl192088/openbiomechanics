from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "hp_obp.csv"


def summarize_numeric(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "non_null": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "zero": int(numeric.eq(0).sum()),
        "negative": int(numeric.lt(0).sum()),
        "min": numeric.min(),
        "p01": numeric.quantile(0.01),
        "p05": numeric.quantile(0.05),
        "median": numeric.median(),
        "p95": numeric.quantile(0.95),
        "p99": numeric.quantile(0.99),
        "max": numeric.max(),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    print(f"rows={len(data)} columns={len(data.columns)}")
    print(f"athletes={data['athlete_uid'].nunique(dropna=True)}")
    print(f"missing_athlete_uid={data['athlete_uid'].isna().sum()}")
    print(f"full_row_duplicates={data.duplicated().sum()}")

    print("\nKEY NUMERIC FIELDS")
    fields = [
        "pitch_speed_mph",
        "peak_power_[w]_mean_cmj",
        "peak_power_/_bm_[w/kg]_mean_cmj",
        "jump_height_(imp-mom)_[cm]_mean_cmj",
        "body_weight_[lbs]",
    ]
    print(pd.DataFrame({field: summarize_numeric(data[field]) for field in fields}).T.to_string())

    print("\nPITCH SPEED FLAGS")
    speed = pd.to_numeric(data["pitch_speed_mph"], errors="coerce")
    for label, mask in {
        "equal_zero": speed.eq(0),
        "between_0_and_40": speed.gt(0) & speed.lt(40),
        "between_40_and_60": speed.ge(40) & speed.lt(60),
        "above_100": speed.gt(100),
    }.items():
        print(f"{label}={mask.sum()}")

    print("\nPITCH SPEED BY PLAYING LEVEL")
    level_summary = data.assign(_speed=speed).groupby("playing_level")["_speed"].agg(
        count="count", min="min", median="median", max="max"
    )
    print(level_summary.sort_values("median").to_string())

    print("\nPITCH SPEED BINS BY PLAYING LEVEL")
    speed_bins = pd.Series(pd.NA, index=data.index, dtype="string")
    speed_bins.loc[speed.eq(0)] = "0"
    speed_bins.loc[speed.gt(0) & speed.lt(40)] = "(0,40)"
    speed_bins.loc[speed.ge(40) & speed.lt(60)] = "[40,60)"
    speed_bins.loc[speed.ge(60) & speed.lt(70)] = "[60,70)"
    speed_bins.loc[speed.ge(70) & speed.lt(80)] = "[70,80)"
    speed_bins.loc[speed.ge(80) & speed.lt(90)] = "[80,90)"
    speed_bins.loc[speed.ge(90)] = "90+"
    print(pd.crosstab(data["playing_level"], speed_bins, dropna=False).to_string())

    print("\nDUPLICATE GRAINS")
    grains = {
        "athlete_test_date": ["athlete_uid", "test_date"],
        "athlete_pitching_session_date": ["athlete_uid", "pitching_session_date"],
        "athlete_test_and_pitching_date": ["athlete_uid", "test_date", "pitching_session_date"],
    }
    for label, columns in grains.items():
        complete = data.dropna(subset=columns)
        duplicate_rows = complete.duplicated(columns, keep=False)
        duplicate_groups = complete.loc[duplicate_rows].groupby(columns, dropna=False).ngroups
        print(f"{label}: duplicate_rows={duplicate_rows.sum()} duplicate_groups={duplicate_groups}")

    print("\nDUPLICATE ATHLETE/TEST/PITCHING DATE ROWS")
    exact_grain = ["athlete_uid", "test_date", "pitching_session_date"]
    exact_duplicates = data.duplicated(exact_grain, keep=False) & data[exact_grain].notna().all(axis=1)
    duplicate_display = exact_grain + [
        "playing_level",
        "pitch_speed_mph",
        "peak_power_[w]_mean_cmj",
        "body_weight_[lbs]",
    ]
    print(data.loc[exact_duplicates, duplicate_display].sort_values(exact_grain).to_string(index=False))

    print("\nCMJ INTERNAL CONSISTENCY")
    power = pd.to_numeric(data["peak_power_[w]_mean_cmj"], errors="coerce")
    relative_power = pd.to_numeric(data["peak_power_/_bm_[w/kg]_mean_cmj"], errors="coerce")
    weight_lb = pd.to_numeric(data["body_weight_[lbs]"], errors="coerce")
    implied_weight_lb = power / relative_power * 2.2046226218
    weight_difference = (implied_weight_lb - weight_lb).abs()
    comparable = power.notna() & relative_power.notna() & weight_lb.notna()
    print(f"comparable_rows={comparable.sum()}")
    print(f"abs_weight_difference_gt_1lb={(weight_difference[comparable] > 1).sum()}")
    print(f"max_abs_weight_difference_lb={weight_difference[comparable].max():.3f}")

    print("\nROWS PER ATHLETE")
    rows_per_athlete = data.groupby("athlete_uid", dropna=True).size()
    print(rows_per_athlete.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

    print("\nZERO-SPEED ROWS")
    zero_columns = [
        "athlete_uid",
        "playing_level",
        "test_date",
        "pitching_session_date",
        "pitch_speed_mph",
        "pitch_speed_mph_group",
        "peak_power_[w]_mean_cmj",
    ]
    print(data.loc[speed.eq(0), zero_columns].to_string(index=False))

    print("\nMISSINGNESS >= 25%")
    missing = data.isna().mean().sort_values(ascending=False)
    print((missing[missing.ge(0.25)] * 100).round(1).to_string())


if __name__ == "__main__":
    main()
