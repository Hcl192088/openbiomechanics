"""Relate peak shoulder external-rotation velocity to shoulder energy transfer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariates)), covariates])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlation(
    frame: pd.DataFrame, x: str, y: str, controls: tuple[str, ...] = ()
) -> tuple[int, float, float]:
    clean = frame[[x, y, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    x_values = clean[x].to_numpy(float)
    y_values = clean[y].to_numpy(float)
    if controls:
        covariates = clean[list(controls)].to_numpy(float)
        x_values = residualize(x_values, covariates)
        y_values = residualize(y_values, covariates)
    r, p = stats.pearsonr(x_values, y_values)
    return len(clean), float(r), float(p)


def cluster_model(
    frame: pd.DataFrame, target: str, controls: tuple[str, ...] = ()
) -> tuple[int, float, float, float, float]:
    columns = [target, "peak_shoulder_external_rotation_velo", "session", *controls]
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    predictors = ["peak_shoulder_external_rotation_velo", *controls]
    model = sm.OLS(clean[target], sm.add_constant(clean[predictors])).fit(
        cov_type="cluster", cov_kwds={"groups": clean["session"]}
    )
    beta = float(model.params["peak_shoulder_external_rotation_velo"])
    ci = model.conf_int().loc["peak_shoulder_external_rotation_velo"]
    return (
        len(clean),
        beta,
        float(model.pvalues["peak_shoulder_external_rotation_velo"]),
        float(ci.iloc[0]),
        float(ci.iloc[1]),
    )


def main() -> None:
    velos = pd.read_csv(
        ROOT / "data" / "full_sig" / "joint_velos.csv",
        usecols=[
            "session_pitch",
            "time",
            "shoulder_velo_z",
            "fp_poi_time",
            "MER_time",
            "BR_time",
        ],
    )
    energy = pd.read_csv(
        ROOT / "data" / "full_sig" / "energy_flow.csv",
        usecols=[
            "session_pitch",
            "time",
            "shoulder_energy_transfer_stp",
            "shoulder_energy_transfer_jfp",
            "fp_poi_time",
            "MER_time",
            "BR_time",
        ],
    )
    rows: list[dict[str, float | str]] = []
    for session_pitch, group in velos.groupby("session_pitch", sort=False):
        group = group.sort_values("time")
        events = {
            event: group[event].dropna().unique()
            for event in ["fp_poi_time", "MER_time", "BR_time"]
        }
        if any(len(values) != 1 for values in events.values()):
            raise ValueError(f"Ambiguous velocity events for {session_pitch}: {events}")
        fp, mer, br = (float(events[event][0]) for event in events)
        if not fp < mer < br:
            raise ValueError(f"Invalid FP/MER/BR order for {session_pitch}: {fp}, {mer}, {br}")
        window = group[(group["time"] >= fp) & (group["time"] <= mer)].dropna(
            subset=["shoulder_velo_z"]
        )
        if len(window) < 2:
            raise ValueError(f"Insufficient FP-MER velocity samples for {session_pitch}")
        minimum_velocity = float(window["shoulder_velo_z"].min())
        rows.append(
            {
                "session_pitch": session_pitch,
                # README convention: shoulder z velocity is internal (+)/external (-).
                "peak_shoulder_external_rotation_velo": -minimum_velocity,
                "peak_external_rotation_velo_time": float(
                    window.loc[window["shoulder_velo_z"].idxmin(), "time"]
                ),
            }
        )

    velocity_metrics = pd.DataFrame(rows)
    energy_rows: list[dict[str, float | str]] = []
    for session_pitch, group in energy.groupby("session_pitch", sort=False):
        group = group.sort_values("time")
        events = {
            event: group[event].dropna().unique()
            for event in ["fp_poi_time", "MER_time", "BR_time"]
        }
        if any(len(values) != 1 for values in events.values()):
            raise ValueError(f"Ambiguous energy events for {session_pitch}: {events}")
        fp, mer, br = (float(events[event][0]) for event in events)
        row: dict[str, float | str] = {"session_pitch": session_pitch}
        for label, end in [("fp_mer", mer), ("fp_br", br)]:
            window = group[(group["time"] >= fp) & (group["time"] <= end)].dropna(
                subset=["shoulder_energy_transfer_stp", "shoulder_energy_transfer_jfp"]
            )
            if len(window) < 2:
                raise ValueError(f"Insufficient {label} energy samples for {session_pitch}")
            time = window["time"].to_numpy(float)
            stp = float(np.trapezoid(window["shoulder_energy_transfer_stp"], time))
            jfp = float(np.trapezoid(window["shoulder_energy_transfer_jfp"], time))
            row[f"shoulder_stp_{label}"] = stp
            row[f"shoulder_jfp_{label}"] = jfp
            row[f"shoulder_total_{label}"] = stp + jfp
        energy_rows.append(row)

    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    poi = pd.read_csv(
        ROOT / "data" / "poi" / "poi_metrics.csv",
        usecols=["session_pitch", "shoulder_transfer_fp_br"],
    )
    pitch = (
        velocity_metrics.merge(pd.DataFrame(energy_rows), on="session_pitch", validate="one_to_one")
        .merge(metadata, on="session_pitch", validate="one_to_one")
        .merge(poi, on="session_pitch", validate="one_to_one")
    )
    if len(pitch) != 411:
        raise ValueError(f"Expected 411 pitches, got {len(pitch)}")
    athlete = pitch.groupby("session", as_index=False).mean(numeric_only=True)
    if len(athlete) != 100:
        raise ValueError(f"Expected 100 athletes, got {len(athlete)}")

    targets = [
        "shoulder_stp_fp_mer",
        "shoulder_total_fp_mer",
        "shoulder_stp_fp_br",
        "shoulder_transfer_fp_br",
    ]
    diagnostic_targets = ["shoulder_jfp_fp_mer", "shoulder_jfp_fp_br"]
    print(f"pitches={len(pitch)}; athletes={len(athlete)}")
    print(
        "Peak shoulder external-rotation velocity (FP-MER): "
        f"mean={athlete['peak_shoulder_external_rotation_velo'].mean():.2f} deg/s, "
        f"SD={athlete['peak_shoulder_external_rotation_velo'].std(ddof=1):.2f}, "
        f"range={athlete['peak_shoulder_external_rotation_velo'].min():.2f}.."
        f"{athlete['peak_shoulder_external_rotation_velo'].max():.2f}"
    )
    primary_partial_p_values: list[float] = []
    for target in [*targets, *diagnostic_targets]:
        raw = correlation(athlete, "peak_shoulder_external_rotation_velo", target)
        partial = correlation(
            athlete,
            "peak_shoulder_external_rotation_velo",
            target,
            controls=("session_mass_kg",),
        )
        if target in targets:
            primary_partial_p_values.append(partial[2])
        cluster_raw = cluster_model(pitch, target)
        cluster_mass = cluster_model(pitch, target, controls=("session_mass_kg",))
        print(f"\n{target}")
        print(
            f"athlete mean raw r={raw[1]:.4f}, p={raw[2]:.4f}; "
            f"partial mass r={partial[1]:.4f}, p={partial[2]:.4f}"
        )
        print(
            f"pitch cluster raw beta/100deg/s={cluster_raw[1] * 100:.4f}, "
            f"p={cluster_raw[2]:.4f}, 95%CI={cluster_raw[3] * 100:.4f}.."
            f"{cluster_raw[4] * 100:.4f}"
        )
        print(
            f"pitch cluster +mass beta/100deg/s={cluster_mass[1] * 100:.4f}, "
            f"p={cluster_mass[2]:.4f}, 95%CI={cluster_mass[3] * 100:.4f}.."
            f"{cluster_mass[4] * 100:.4f}"
        )
    print("\nBonferroni across four primary athlete-level endpoints")
    for target, p_value in zip(targets, primary_partial_p_values, strict=True):
        print(f"{target}: adjusted_p={min(p_value * len(targets), 1.0):.4f}")


if __name__ == "__main__":
    main()
