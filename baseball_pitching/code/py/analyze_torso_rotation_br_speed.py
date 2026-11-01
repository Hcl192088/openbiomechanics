"""Torso rotation at BR vs pitch speed: linear, nonlinear (optimum) and within-pitcher checks.

torso_rotation_* (POI) = torso_angle_z: axial rotation toward home (+). The first frame of the
full signal is about 0 deg (side-on set position, chest toward third base for a RHP), so 90 deg is
chest square to home plate and > 90 deg is rotated past home toward first base (RHP view).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "baseball_pitching" / "data"


def fmt(model, term: str) -> str:
    lo, hi = model.conf_int().loc[term]
    return f"{model.params[term]:+.3f} [{lo:+.3f}, {hi:+.3f}] p={model.pvalues[term]:.3g}"


def main() -> None:
    poi = pd.read_csv(DATA / "poi" / "poi_metrics.csv",
                      usecols=["session_pitch", "session", "pitch_speed_mph", "torso_rotation_mer",
                               "torso_rotation_br", "torso_rotation_min", "max_torso_rotational_velo"])
    meta = pd.read_csv(DATA / "metadata.csv", usecols=["session_pitch", "session_height_m", "session_mass_kg"])
    joint = pd.read_csv(DATA / "full_sig" / "joint_angles.csv", usecols=["session_pitch", "time", "torso_angle_z"])
    start = joint.sort_values("time").groupby("session_pitch")["torso_angle_z"].first()
    data = poi.merge(meta, on="session_pitch", validate="one_to_one").dropna()
    if len(data) != 411:
        raise RuntimeError(f"Expected 411 pitches, found {len(data)}")
    x = data["torso_rotation_br"]
    print(f"first-frame torso_angle_z median={start.median():+.1f} deg (zero reference check)")
    print(f"BR torso rotation: median={x.median():.1f}, IQR=[{x.quantile(.25):.1f}, {x.quantile(.75):.1f}], "
          f"range=[{x.min():.1f}, {x.max():.1f}]; pitches < 100 deg: {(x < 100).sum()}")

    cl = {"cov_type": "cluster", "cov_kwds": {"groups": data["session"]}}
    lin = smf.ols("pitch_speed_mph ~ torso_rotation_br + session_height_m + session_mass_kg", data=data).fit(**cl)
    quad = smf.ols("pitch_speed_mph ~ torso_rotation_br + I(torso_rotation_br**2) + session_height_m + session_mass_kg",
                   data=data).fit(**cl)
    print(f"\nlinear (+height/mass): {fmt(lin, 'torso_rotation_br')}")
    print(f"quadratic term: {fmt(quad, 'I(torso_rotation_br ** 2)')}")

    pm = data.groupby("session")[["torso_rotation_br", "pitch_speed_mph"]].mean()
    pm["quintile"] = pd.qcut(pm["torso_rotation_br"], 5, labels=False) + 1
    print("\nPitcher-mean quintiles of BR torso rotation:")
    print(pm.groupby("quintile").agg(n=("pitch_speed_mph", "size"),
                                     rotation_mean=("torso_rotation_br", "mean"),
                                     speed_mean=("pitch_speed_mph", "mean")).round(2).to_string())
    print(f"Spearman pitcher-mean rho={stats.spearmanr(pm['torso_rotation_br'], pm['pitch_speed_mph']).statistic:+.3f}")

    # Within-pitcher: does the same pitcher throw faster on pitches with less rotation at BR?
    data["rot_dev"] = data["torso_rotation_br"] - data.groupby("session")["torso_rotation_br"].transform("mean")
    data["speed_dev"] = data["pitch_speed_mph"] - data.groupby("session")["pitch_speed_mph"].transform("mean")
    print(f"\nwithin-pitcher SD of BR torso rotation={data['rot_dev'].std():.1f} deg "
          f"(between-pitcher SD={pm['torso_rotation_br'].std():.1f})")
    within = smf.ols("pitch_speed_mph ~ torso_rotation_br + C(session)", data=data).fit(**cl)
    within_mer = smf.ols("pitch_speed_mph ~ torso_rotation_br + torso_rotation_mer + C(session)", data=data).fit(**cl)
    within_velo = smf.ols("pitch_speed_mph ~ torso_rotation_br + max_torso_rotational_velo + C(session)",
                          data=data).fit(**cl)
    print(f"within pitcher: {fmt(within, 'torso_rotation_br')}")
    print(f"within pitcher, MER rotation held: {fmt(within_mer, 'torso_rotation_br')}")
    print(f"within pitcher, peak torso velocity held: {fmt(within_velo, 'torso_rotation_br')}")
    slopes = []
    for session, g in data.groupby("session"):
        if len(g) >= 3 and g["torso_rotation_br"].std() > 0:
            slopes.append(stats.linregress(g["torso_rotation_br"], g["pitch_speed_mph"]).slope)
    slopes = np.array(slopes)
    print(f"per-pitcher slopes (n={len(slopes)} pitchers with >=3 pitches): "
          f"negative (less rotation = faster) {int((slopes < 0).sum())}, positive {int((slopes > 0).sum())}")

    over_rotation(data, cl)


def over_rotation(data: pd.DataFrame, cl: dict) -> None:
    """Is there a point past which more rotation at BR is slower ("rotating too far")?"""
    print("\n== Over-rotation checks ==")
    data = data.copy()
    data["torso_excursion"] = data["torso_rotation_br"] - data["torso_rotation_min"]
    data["torso_mer_to_br"] = data["torso_rotation_br"] - data["torso_rotation_mer"]
    body = "session_height_m + session_mass_kg"
    peak = smf.ols(f"pitch_speed_mph ~ torso_rotation_br + max_torso_rotational_velo + {body}", data=data).fit(**cl)
    print(f"BR rotation, peak torso velocity held (+height/mass): {fmt(peak, 'torso_rotation_br')}")
    for c in ["torso_excursion", "torso_mer_to_br"]:
        m = smf.ols(f"pitch_speed_mph ~ {c} + max_torso_rotational_velo + {body}", data=data).fit(**cl)
        w = smf.ols(f"pitch_speed_mph ~ {c} + C(session)", data=data).fit(**cl)
        print(f"{c}: median={data[c].median():.1f}; peak-held {fmt(m, c)}; within {fmt(w, c)}")

    # Hinge: slope above a cut point, with and without peak torso velocity held.
    print("Hinge models: speed slope ABOVE cut (mph/deg), 2-sided cluster p")
    for cut in [115, 120, 125, 130, 135]:
        data["_above"] = np.clip(data["torso_rotation_br"] - cut, 0, None)
        n_above = int((data["torso_rotation_br"] > cut).sum())
        n_sess = data.loc[data["torso_rotation_br"] > cut, "session"].nunique()
        h = smf.ols(f"pitch_speed_mph ~ torso_rotation_br + _above + {body}", data=data).fit(**cl)
        hp = smf.ols(f"pitch_speed_mph ~ torso_rotation_br + _above + max_torso_rotational_velo + {body}",
                     data=data).fit(**cl)
        print(f"  cut {cut}: n above={n_above} ({n_sess} pitchers) | extra slope {fmt(h, '_above')} | "
              f"peak-held {fmt(hp, '_above')}")

    # Robustness of the 130-deg hinge: drop one above-cut pitcher at a time.
    data["_above"] = np.clip(data["torso_rotation_br"] - 130, 0, None)
    loo = []
    for session in data.loc[data["_above"] > 0, "session"].unique():
        sub = data[data["session"] != session]
        m = smf.ols(f"pitch_speed_mph ~ torso_rotation_br + _above + max_torso_rotational_velo + {body}",
                    data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["session"]})
        loo.append((m.params["_above"], m.pvalues["_above"]))
    loo = np.array(loo)
    print(f"cut 130 peak-held, leave-one-pitcher-out ({len(loo)} drops): slope range "
          f"[{loo[:, 0].min():+.3f}, {loo[:, 0].max():+.3f}], p range [{loo[:, 1].min():.3g}, {loo[:, 1].max():.3g}], "
          f"p<0.05 in {int((loo[:, 1] < 0.05).sum())}")
    top = data[data["torso_rotation_br"] > 130].groupby("session").agg(
        n=("pitch_speed_mph", "size"), rotation=("torso_rotation_br", "mean"), speed=("pitch_speed_mph", "mean"),
        peak_velo=("max_torso_rotational_velo", "mean"))
    print("pitchers with any pitch > 130 deg:")
    print(top.sort_values("rotation").round(1).to_string())

    # Speed by BR rotation band, raw and adjusted for peak torso velocity / body size.
    bands = pd.cut(data["torso_rotation_br"], [0, 110, 120, 130, 180], labels=["<110", "110-120", "120-130", ">=130"])
    resid = smf.ols(f"pitch_speed_mph ~ max_torso_rotational_velo + {body}", data=data).fit().resid
    summary = pd.DataFrame({"band": bands, "speed": data["pitch_speed_mph"], "resid": resid,
                            "session": data["session"]}).groupby("band", observed=True).agg(
        n=("speed", "size"), pitchers=("session", "nunique"), speed=("speed", "mean"),
        speed_adj_resid=("resid", "mean"))
    print("BR rotation bands (resid = speed beyond what peak torso velocity and body size predict):")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
