from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "high_performance" / "data" / "hp_obp.csv"
OUTPUT = ROOT / "baseball_pitching" / "imgs" / "cmj_power_level_user_prediction.png"
REPORT = ROOT / "baseball_pitching" / "data" / "hp_pitch_speed_prediction" / "cmj_level_user_prediction.csv"

POWER_COL = "peak_power_[w]_mean_cmj"
SPEED_COL = "pitch_speed_mph"
LEVEL_COL = "playing_level"
USER_POWER_W = 4312.0
USER_SPEED_MPH = 114.0 / 1.609344

PANELS = [
    ("All levels", None, "#96CEB4"),
    ("High School", "High School", "#FF6B6B"),
    ("College", "College", "#4ECDC4"),
    ("Pro", "Pro", "#45B7D1"),
]


def main() -> None:
    raw = pd.read_csv(DATA, low_memory=False)
    data = raw[[POWER_COL, SPEED_COL, LEVEL_COL, "athlete_uid"]].dropna().copy()
    data = data[data[SPEED_COL] > 40].copy()

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)

    for ax, (label, level, color) in zip(axes.flat, PANELS):
        subset = data if level is None else data[data[LEVEL_COL] == level]
        x = subset[[POWER_COL]]
        y = subset[SPEED_COL]
        model = LinearRegression().fit(x, y)
        fitted = model.predict(x)
        prediction = float(model.predict(pd.DataFrame({POWER_COL: [USER_POWER_W]}))[0])
        r2 = float(r2_score(y, fitted))

        grid = np.linspace(float(x.min().iloc[0]), float(x.max().iloc[0]), 200)
        line = model.predict(pd.DataFrame({POWER_COL: grid}))
        ax.scatter(x[POWER_COL], y, alpha=0.35, color=color, s=28)
        ax.plot(grid, line, color="#D62728", linewidth=2)
        ax.scatter(
            USER_POWER_W,
            prediction,
            marker="D",
            s=110,
            color="#6A0DAD",
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
            label=f"Level-line prediction: {prediction:.1f} mph",
        )
        ax.scatter(
            USER_POWER_W,
            USER_SPEED_MPH,
            marker="*",
            s=220,
            color="#FFD700",
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
            label=f"Actual max: {USER_SPEED_MPH:.1f} mph",
        )
        ax.axvline(USER_POWER_W, color="black", linestyle=":", alpha=0.55)
        ax.set_title(f"{label}\nR²={r2:.3f}, n={len(subset)}, athletes={subset['athlete_uid'].nunique()}")
        ax.set_xlabel("CMJ Peak Power (W)")
        ax.set_ylabel("Pitch Speed (mph)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

        rows.append(
            {
                "group": label,
                "rows": len(subset),
                "athletes": subset["athlete_uid"].nunique(),
                "r2_in_sample": r2,
                "slope_mph_per_w": float(model.coef_[0]),
                "intercept_mph": float(model.intercept_),
                "prediction_at_4312w_mph": prediction,
                "prediction_at_4312w_kmh": prediction * 1.609344,
                "actual_max_mph": USER_SPEED_MPH,
                "prediction_minus_actual_mph": prediction - USER_SPEED_MPH,
            }
        )

    fig.suptitle(
        "CMJ Peak Power vs Pitch Speed by Competition Level\n"
        "User: 4312 W; actual max 114 km/h; pitch speeds >40 mph",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"figure={OUTPUT}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
