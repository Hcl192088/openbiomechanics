"""Quantify the visual label 'hip extension direction toward home plate' (qualitative_mechanics
hip_extension good / bad) and relate candidate metrics to pitch speed.

Window: pkh_time -> fp_poi_time (FP = fp_poi_time). Lab +x = toward home; lateral = lab y converted to
throwing-arm side + (-y for RHP, +y for LHP; landmarks.csv is not mirrored).
Direction angles are atan2(arm_side, forward) in degrees: 0 = straight at home, + = arm side, - = glove side.
Candidates
  pelvis_disp_dir   : pelvis-centre (mid rear/lead hip) horizontal displacement PKH -> FP
  pelvis_vel_dir_fp : pelvis-centre horizontal velocity direction at FP
  com_vel_dir_fp    : whole-body COM horizontal velocity direction at FP
  rear_leg_dir_fp   : rear ankle -> rear hip horizontal direction at FP (line of the rear leg push)
  rear_grf_dir      : rear force-plate horizontal impulse direction PKH -> FP (sign of y checked by
                      comparing RHP / LHP means; force_plate mirroring is not documented)
Label check: 30 labelled pitches (one rater) - AUC good vs bad for each candidate (|angle| and signed).
Speed: between-athlete r on athlete means (bootstrap CI), partial r for mass/height, within-athlete r.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
LAB = HERE.parents[2] / "experiments/qualitative_mechanics/labels.csv"
OUT = HERE / "hip_extension_direction_outputs"
NB = 2000


def ang(f, a):
    return np.degrees(np.arctan2(a, f))


def main():
    OUT.mkdir(exist_ok=True)
    pts = ["rear_hip", "lead_hip", "rear_ankle_jc", "centerofmass"]
    L = pd.read_csv(DATA / "full_sig/landmarks.csv",
                    usecols=["session_pitch", "time", "pkh_time", "fp_poi_time"]
                    + [f"{p}_{a}" for p in pts for a in "xy"])
    F = pd.read_csv(DATA / "full_sig/force_plate.csv",
                    usecols=["session_pitch", "time", "rear_force_x", "rear_force_y", "pkh_time", "fp_poi_time"])
    poi = pd.read_csv(DATA / "poi/poi_metrics.csv", usecols=["session_pitch", "p_throws"])
    meta = pd.read_csv(DATA / "metadata.csv", usecols=["session", "session_pitch", "pitch_speed_mph",
                                                       "session_mass_kg", "session_height_m"])
    info = poi.merge(meta, on="session_pitch", validate="one_to_one").set_index("session_pitch")
    Fg = dict(tuple(F.groupby("session_pitch")))
    rows = []
    for sp, g in L.groupby("session_pitch"):
        pkh, fp = g[["pkh_time", "fp_poi_time"]].iloc[0]
        if sp not in info.index or np.isnan(pkh) or np.isnan(fp):
            continue
        s = -1 if info.at[sp, "p_throws"] == "R" else 1
        t = g["time"].to_numpy()
        px = ((g["rear_hip_x"] + g["lead_hip_x"]) / 2).to_numpy()
        py = ((g["rear_hip_y"] + g["lead_hip_y"]) / 2).to_numpy() * s
        r = {"session_pitch": sp, "p_throws": info.at[sp, "p_throws"]}
        r["pelvis_disp_dir"] = ang(np.interp(fp, t, px) - np.interp(pkh, t, px),
                                   np.interp(fp, t, py) - np.interp(pkh, t, py))
        for name, x, y in [("pelvis", px, py),
                           ("com", g["centerofmass_x"].to_numpy(), g["centerofmass_y"].to_numpy() * s)]:
            vx, vy = np.gradient(x, t), np.gradient(y, t)
            r[f"{name}_vel_dir_fp"] = ang(np.interp(fp, t, vx), np.interp(fp, t, vy))
        r["rear_leg_dir_fp"] = ang(np.interp(fp, t, g["rear_hip_x"]) - np.interp(fp, t, g["rear_ankle_jc_x"]),
                                   s * (np.interp(fp, t, g["rear_hip_y"]) - np.interp(fp, t, g["rear_ankle_jc_y"])))
        if sp in Fg:
            f = Fg[sp]
            w = (f["time"] >= pkh) & (f["time"] <= fp)
            r["rear_grf_x_imp"] = f.loc[w, "rear_force_x"].sum()
            r["rear_grf_y_imp_raw"] = f.loc[w, "rear_force_y"].sum()
        rows.append(r)
    df = pd.DataFrame(rows)
    # force plate lateral sign: if RHP and LHP raw-y means have opposite sign, the plate is not mirrored
    m = df.groupby("p_throws")["rear_grf_y_imp_raw"].mean()
    mirrored = np.sign(m["R"]) == np.sign(m["L"])
    side = np.where(df["p_throws"] == "R", -1, 1) if not mirrored else -1
    df["rear_grf_dir"] = ang(df["rear_grf_x_imp"], df["rear_grf_y_imp_raw"] * side)
    df = df.merge(info[["session", "pitch_speed_mph", "session_mass_kg", "session_height_m"]],
                  left_on="session_pitch", right_index=True)
    df.to_csv(OUT / "per_pitch.csv", index=False)
    cands = ["pelvis_disp_dir", "pelvis_vel_dir_fp", "com_vel_dir_fp", "rear_leg_dir_fp", "rear_grf_dir"]

    lab = pd.read_csv(LAB, dtype={"session_pitch": str})[["session_pitch", "hip_extension"]]
    df["session_pitch"] = df["session_pitch"].astype(str)
    x = lab[lab["hip_extension"].isin(["good", "bad"])].merge(df, on="session_pitch")
    res = []
    for c in cands:
        gd, bd = x.loc[x.hip_extension == "good", c].dropna(), x.loc[x.hip_extension == "bad", c].dropna()
        u = stats.mannwhitneyu(gd, bd)
        ua = stats.mannwhitneyu(gd.abs(), bd.abs())
        res.append({"metric": c, "n_good": len(gd), "n_bad": len(bd), "good_mean": gd.mean(), "bad_mean": bd.mean(),
                    "auc_signed": u.statistic / len(gd) / len(bd), "p_signed": u.pvalue,
                    "good_abs": gd.abs().mean(), "bad_abs": bd.abs().mean(),
                    "auc_abs_good_lower": 1 - ua.statistic / len(gd) / len(bd), "p_abs": ua.pvalue})
    res = pd.DataFrame(res)

    A = df.drop(columns=["session_pitch", "p_throws"]).groupby("session").mean()
    y = A["pitch_speed_mph"].to_numpy()
    B = np.column_stack([np.ones(len(A)), A["session_mass_kg"], A["session_height_m"]])

    def resid(v):
        return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]

    W = df.drop(columns=["session_pitch", "p_throws"]).groupby("session").transform(lambda s: s - s.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(A), (NB, len(A)))
    sp_rows = []
    for c in cands:
        ok = A[c].notna().to_numpy()
        v, yy = A[c].to_numpy()[ok], y[ok]
        ii = idx[:, : ok.sum()] % ok.sum()
        bs = [np.corrcoef(v[i], yy[i])[0, 1] for i in ii]
        Bo = B[ok]
        rr = lambda a: a - Bo @ np.linalg.lstsq(Bo, a, rcond=None)[0]
        wv = W[c].notna()
        sp_rows.append({"metric": c, "athletes": ok.sum(), "mean_deg": v.mean(), "between_sd": v.std(ddof=1),
                        "r_speed": np.corrcoef(v, yy)[0, 1], "lo": np.percentile(bs, 2.5),
                        "hi": np.percentile(bs, 97.5), "r_partial_body": np.corrcoef(rr(v), rr(yy))[0, 1],
                        "r_abs_speed": np.corrcoef(np.abs(v), yy)[0, 1],
                        "r_within": np.corrcoef(W.loc[wv, c], W.loc[wv, "pitch_speed_mph"])[0, 1]})
    sp_rows = pd.DataFrame(sp_rows)
    res.merge(sp_rows, on="metric").to_csv(OUT / "summary.csv", index=False)
    print("pitches", len(df), "force plate lateral mirrored:", bool(mirrored), "raw y mean R/L", m.round(0).to_dict())
    print(res.round(2).to_string(index=False))
    print(sp_rows.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
