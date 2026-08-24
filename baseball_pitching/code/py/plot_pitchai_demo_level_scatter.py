from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
POI = ROOT / "baseball_pitching/data/poi/poi_metrics.csv"
META = ROOT / "baseball_pitching/data/metadata.csv"
OUT = ROOT / "baseball_pitching/imgs/pitchai_demo_level_scatter.png"

FEATURES = {
    "max_rotation_hip_shoulder_separation": (45.0, "Peak hip–shoulder separation (deg)"),
    "max_shoulder_horizontal_abduction": (8.0, "Peak shoulder horizontal abduction (deg)"),
    "max_shoulder_external_rotation": (171.0, "Peak shoulder external rotation (deg)"),
    "stride_length": (0.99, "Stride length (body-height ratio)"),
    "elbow_varus_moment": (32.0, "Peak elbow varus moment (Nm)"),
    "lead_knee_extension_angular_velo_max": (402.0, "Peak lead-knee extension velocity (deg/s)"),
}


def main() -> None:
    poi = pd.read_csv(POI)
    meta = pd.read_csv(META, usecols=["session_pitch", "playing_level"])
    if meta["session_pitch"].duplicated().any():
        raise ValueError("metadata.session_pitch is not unique")
    df = poi.merge(meta, on="session_pitch", how="left", validate="one_to_one")
    if df["playing_level"].isna().any():
        raise ValueError("playing_level missing after session_pitch join")

    preferred = ["youth", "high_school", "college", "independent", "milb", "mlb"]
    observed = set(df.playing_level)
    order = [x for x in preferred if x in observed] + sorted(observed - set(preferred))
    palette = dict(zip(order, sns.color_palette("viridis", len(order))))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (feature, (demo_value, label)) in zip(axes.flat, FEATURES.items()):
        sns.scatterplot(
            data=df,
            x=feature,
            y="pitch_speed_mph",
            hue="playing_level",
            hue_order=order,
            palette=palette,
            alpha=0.68,
            s=38,
            linewidth=0,
            ax=ax,
            legend=ax is axes.flat[0],
        )
        ax.axvline(demo_value, color="red", linestyle="--", linewidth=2, label="PitchAI screenshot")
        ax.scatter([demo_value], [77.2137], marker="X", s=150, color="red", edgecolor="black", zorder=5)
        ax.set_xlabel(label)
        ax.set_ylabel("Pitch speed (mph)")
        ax.set_title(f"Screenshot value = {demo_value:g}")
        if ax is axes.flat[0]:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title="Playing level", fontsize=8, loc="best")

    fig.suptitle(
        "PitchAI screenshot values against OpenBiomechanics levels\n"
        "Red X = six-feature analogue-model prediction (77.2 mph); red line = screenshot value",
        fontsize=16,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(df.groupby("playing_level", observed=True)["pitch_speed_mph"].agg(["count", "mean", "median", "min", "max"]))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
