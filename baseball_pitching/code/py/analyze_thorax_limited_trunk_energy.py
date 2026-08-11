"""Compare trunk z-axis kinetic-energy loss with STP in thorax-limited periods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RAD_PER_DEG = np.pi / 180.0


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


def interval_summary(
    group: pd.DataFrame,
    sample_mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    time = group["time"].to_numpy(float)
    omega = group["torso_velo_z"].to_numpy(float)
    stp = group["shoulder_energy_transfer_stp"].to_numpy(float)
    interval_mask = sample_mask[:-1] & sample_mask[1:]
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError(f"Non-increasing time in {group['session_pitch'].iloc[0]}")

    delta_k_per_izz = 0.5 * np.diff((omega * RAD_PER_DEG) ** 2)
    stp_interval_energy = 0.5 * (stp[:-1] + stp[1:]) * dt
    selected_delta_k = delta_k_per_izz[interval_mask]

    return {
        f"{prefix}_interval_n": int(interval_mask.sum()),
        f"{prefix}_duration_s": float(dt[interval_mask].sum()),
        f"{prefix}_stp_j": float(stp_interval_energy[interval_mask].sum()),
        f"{prefix}_net_k_loss_per_izz": float(-selected_delta_k.sum()),
        f"{prefix}_positive_k_loss_per_izz": float(
            np.maximum(-selected_delta_k, 0.0).sum()
        ),
        f"{prefix}_k_gain_per_izz": float(np.maximum(selected_delta_k, 0.0).sum()),
    }


def main() -> None:
    velos = pd.read_csv(
        ROOT / "data" / "full_sig" / "joint_velos.csv",
        usecols=["session_pitch", "time", "torso_velo_z", "fp_poi_time", "BR_time"],
    )
    energy = pd.read_csv(
        ROOT / "data" / "full_sig" / "energy_flow.csv",
        usecols=[
            "session_pitch",
            "time",
            "thorax_dist_seg_pwr",
            "upper_arm_prox_seg_pwr",
            "shoulder_energy_transfer_jfp",
            "shoulder_energy_transfer_stp",
        ],
    )
    data = velos.merge(
        energy, on=["session_pitch", "time"], how="inner", validate="one_to_one"
    )
    if len(data) != len(velos) or len(data) != len(energy):
        raise ValueError(
            f"Full-signal time join mismatch: merged={len(data)}, "
            f"velos={len(velos)}, energy={len(energy)}"
        )

    data["thorax_stp"] = (
        data["thorax_dist_seg_pwr"] + data["shoulder_energy_transfer_jfp"]
    )
    data["upper_arm_stp"] = (
        data["upper_arm_prox_seg_pwr"] - data["shoulder_energy_transfer_jfp"]
    )

    rows: list[dict[str, float | str]] = []
    for session_pitch, pitch in data.groupby("session_pitch", sort=False):
        pitch = pitch.sort_values("time").dropna().copy()
        fp = pitch["fp_poi_time"].unique()
        br = pitch["BR_time"].unique()
        if len(fp) != 1 or len(br) != 1:
            raise ValueError(f"Ambiguous FP/BR for {session_pitch}")
        pitch = pitch[
            (pitch["time"] >= float(fp[0])) & (pitch["time"] <= float(br[0]))
        ].copy()
        if len(pitch) < 3:
            raise ValueError(f"Insufficient FP-BR samples for {session_pitch}")

        peak_index = int(np.argmax(np.abs(pitch["torso_velo_z"].to_numpy(float))))
        time = pitch["time"].to_numpy(float)
        thorax = pitch["thorax_stp"].to_numpy(float)
        upper_arm = pitch["upper_arm_stp"].to_numpy(float)
        opposite = thorax * upper_arm < 0
        thorax_smaller = np.abs(thorax) < np.abs(upper_arm)
        thorax_to_arm = (thorax < 0) & (upper_arm > 0)
        thorax_limited_positive = opposite & thorax_smaller & thorax_to_arm
        post_peak = np.arange(len(pitch)) >= peak_index

        row: dict[str, float | str] = {"session_pitch": session_pitch}
        row.update(
            interval_summary(pitch, thorax_limited_positive, "fp_br_thorax_limited")
        )
        row.update(
            interval_summary(
                pitch,
                thorax_limited_positive & post_peak,
                "peak_br_thorax_limited",
            )
        )
        row["torso_peak_time"] = float(time[peak_index])
        rows.append(row)

    per_pitch = pd.DataFrame(rows)
    metadata = pd.read_csv(
        ROOT / "data" / "metadata.csv",
        usecols=["session_pitch", "session", "session_mass_kg"],
    )
    inertia = pd.read_csv(
        ROOT / "data" / "poi" / "thorax_inertia_estimates.csv",
        usecols=["session", "rta_izz_kg_m2"],
    )
    per_pitch = (
        per_pitch.merge(metadata, on="session_pitch", validate="one_to_one")
        .merge(inertia, on="session", validate="many_to_one")
    )
    if len(per_pitch) != 411:
        raise ValueError(f"Expected 411 pitches, got {len(per_pitch)}")

    prefixes = ["fp_br_thorax_limited", "peak_br_thorax_limited"]
    for prefix in prefixes:
        per_pitch[f"{prefix}_net_k_loss_j"] = (
            per_pitch[f"{prefix}_net_k_loss_per_izz"] * per_pitch["rta_izz_kg_m2"]
        )
        per_pitch[f"{prefix}_positive_k_loss_j"] = (
            per_pitch[f"{prefix}_positive_k_loss_per_izz"]
            * per_pitch["rta_izz_kg_m2"]
        )
        per_pitch[f"{prefix}_k_gain_j"] = (
            per_pitch[f"{prefix}_k_gain_per_izz"] * per_pitch["rta_izz_kg_m2"]
        )

    athlete = per_pitch.groupby("session", as_index=False).mean(numeric_only=True)
    print(f"pitches={len(per_pitch)}; athletes={len(athlete)}")
    for prefix in prefixes:
        positive_duration = athlete[f"{prefix}_duration_s"] > 0
        athlete[f"{prefix}_stp_mean_power_w"] = np.where(
            positive_duration,
            athlete[f"{prefix}_stp_j"] / athlete[f"{prefix}_duration_s"],
            np.nan,
        )
        athlete[f"{prefix}_net_k_loss_rate_w"] = np.where(
            positive_duration,
            athlete[f"{prefix}_net_k_loss_j"] / athlete[f"{prefix}_duration_s"],
            np.nan,
        )
        athlete[f"{prefix}_positive_k_loss_rate_w"] = np.where(
            positive_duration,
            athlete[f"{prefix}_positive_k_loss_j"] / athlete[f"{prefix}_duration_s"],
            np.nan,
        )
        print(f"\n{prefix}")
        print(f"athletes with positive duration={int(positive_duration.sum())}")
        print(
            f"mean duration={athlete[f'{prefix}_duration_s'].mean():.5f} s; "
            f"mean STP={athlete[f'{prefix}_stp_j'].mean():.3f} J; "
            f"mean net Kz loss={athlete[f'{prefix}_net_k_loss_j'].mean():.3f} J; "
            f"mean positive Kz loss={athlete[f'{prefix}_positive_k_loss_j'].mean():.3f} J; "
            f"mean Kz gain={athlete[f'{prefix}_k_gain_j'].mean():.3f} J"
        )
        for outcome in ["stp_j", "positive_k_loss_j"]:
            duration_relationship = correlation(
                athlete,
                f"{prefix}_duration_s",
                f"{prefix}_{outcome}",
                controls=("session_mass_kg",),
            )
            print(
                f"duration vs {outcome}: partial_mass_r={duration_relationship[1]:.4f}, "
                f"p={duration_relationship[2]:.4f}"
            )
        for loss in ["net_k_loss_j", "positive_k_loss_j"]:
            predictor = f"{prefix}_{loss}"
            target = f"{prefix}_stp_j"
            raw = correlation(athlete, predictor, target)
            partial = correlation(
                athlete, predictor, target, controls=("session_mass_kg",)
            )
            partial_duration = correlation(
                athlete,
                predictor,
                target,
                controls=("session_mass_kg", f"{prefix}_duration_s"),
            )
            print(
                f"{loss}: raw_r={raw[1]:.4f}, p={raw[2]:.4f}; "
                f"partial_mass_r={partial[1]:.4f}, p={partial[2]:.4f}; "
                f"partial_mass_duration_r={partial_duration[1]:.4f}, "
                f"p={partial_duration[2]:.4f}"
            )
        for rate in ["net_k_loss_rate_w", "positive_k_loss_rate_w"]:
            predictor = f"{prefix}_{rate}"
            target = f"{prefix}_stp_mean_power_w"
            raw = correlation(athlete, predictor, target)
            partial = correlation(
                athlete, predictor, target, controls=("session_mass_kg",)
            )
            print(
                f"{rate} vs STP mean power: n={raw[0]}, "
                f"raw_r={raw[1]:.4f}, p={raw[2]:.4f}; "
                f"partial_mass_r={partial[1]:.4f}, p={partial[2]:.4f}"
            )


if __name__ == "__main__":
    main()
