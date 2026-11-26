"""Is the upper trunk directly above the pelvis in lab space, and does that relate to pitch speed?

Offset = thorax_prox (upper thorax, ~5 cm from the shoulder-centre midpoint) - pelvis centre
(mid rear_hip / lead_hip), from landmarks.csv (not mirrored for left-handers).
  forward = lab x (toward home plate +)
  arm_side = lab y with throwing-arm side +  (checked: at BR, throwing shoulder y < glove shoulder y
             for every RHP and > for every LHP, so arm_side = -y for R, +y for L)
Lab-space displacement, not a segment angle; compared with torso_angle_x (anterior tilt) and
torso_angle_y (lateral tilt), including partial r of offset given the matching angle and vice versa.
Events: FP = fp_poi_time, MER_time, BR_time. Curve: FP -> BR normalised to 0-100 % (101 points).
Stats: between-athlete r with speed on athlete means (bootstrap CI over athletes, partial r controlling
mass and height); within-athlete r on session-demeaned pitches. Curve: pointwise between-athlete r,
cluster mass permutation test (athlete speeds shuffled) at the |r| threshold for p < .05.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
OUT = HERE / "trunk_over_pelvis_speed_outputs"
IMG = HERE.parents[1] / "imgs"
NB, NP = 2000, 2000
PHASE = np.linspace(0, 1, 101)
AXES = {"forward": "Forward offset (toward home +)", "arm_side": "Lateral offset (throwing-arm side +)"}
COLORS = {"forward": "#2a78d6", "arm_side": "#eb6834"}


def resid(v, B):
    return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]


def clusters(mask):
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def lateral_checks(df, A, B, y, idx):
    """(1) Non-linear lateral offset: quadratic term (body-adjusted) and extreme-group contrasts on athlete
    means (most glove-side 20 % and most arm-side 20 % vs the middle 60 %, body-adjusted speed).
    (2) Sign of torso_angle_y: pitch-level r / slope against the marker thorax tilt (glove side +)."""
    rows = []
    ys = resid(y, B)  # body-adjusted athlete speed
    for ev in ["fp", "mer", "br"]:
        v = A[f"arm_side_{ev}_cm"].to_numpy()

        def quad(i):
            vc = v[i] - v[i].mean()
            X = np.column_stack([B[i], vc, vc ** 2])
            beta = np.linalg.lstsq(X, y[i], rcond=None)[0]
            return beta[-1] * 100  # mph per (10 cm)^2

        def groups(i):
            q20, q80 = np.percentile(v[i], [20, 80])
            mid = (v[i] > q20) & (v[i] < q80)
            return ys[i][v[i] <= q20].mean() - ys[i][mid].mean(), ys[i][v[i] >= q80].mean() - ys[i][mid].mean()

        X = np.column_stack([B, v - v.mean(), (v - v.mean()) ** 2])
        beta, ssr = np.linalg.lstsq(X, y, rcond=None)[:2]
        se = np.sqrt(ssr[0] / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)[-1, -1])
        g0 = groups(np.arange(len(v)))
        qb = np.array([quad(i) for i in idx])
        gb = np.array([groups(i) for i in idx])
        rows.append({"check": "arm_side_nonlinear", "event": ev, "quad_mph_per_10cm_sq": beta[-1] * 100,
                     "quad_lo": np.percentile(qb, 2.5), "quad_hi": np.percentile(qb, 97.5),
                     "quad_p": 2 * stats.t.sf(abs(beta[-1] / se), len(y) - X.shape[1]),
                     "glove20_minus_mid_mph": g0[0], "glove20_lo": np.percentile(gb[:, 0], 2.5),
                     "glove20_hi": np.percentile(gb[:, 0], 97.5),
                     "glove20_cutoff_cm": np.percentile(v, 20),
                     "arm20_minus_mid_mph": g0[1], "arm20_lo": np.percentile(gb[:, 1], 2.5),
                     "arm20_hi": np.percentile(gb[:, 1], 97.5), "arm20_cutoff_cm": np.percentile(v, 80)})
    for ev in ["fp", "mer", "br"]:
        a, m = df[f"torso_angle_y_{ev}"], df[f"marker_glove_tilt_{ev}_deg"]
        rows.append({"check": "torso_angle_y_sign", "event": ev, "r_vs_marker_glove_tilt": np.corrcoef(a, m)[0, 1],
                     "slope_deg_per_deg": np.polyfit(m, a, 1)[0], "torso_angle_y_mean": a.mean(),
                     "marker_glove_tilt_mean": m.mean(),
                     "marker_tilt_r_speed_between": np.corrcoef(A[f"marker_glove_tilt_{ev}_deg"], y)[0, 1]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "lateral_checks.csv", index=False)
    print(out.round(2).dropna(axis=1, how="all").to_string(index=False))


def main():
    OUT.mkdir(exist_ok=True)
    key = ["session_pitch", "time"]
    pts = ["thorax_prox", "thorax_dist", "rear_hip", "lead_hip", "shoulder_jc", "glove_shoulder_jc"]
    L = pd.read_csv(DATA / "full_sig/landmarks.csv",
                    usecols=key + [f"{p}_{a}" for p in pts for a in "xyz"] + ["fp_poi_time", "MER_time", "BR_time"])
    J = pd.read_csv(DATA / "full_sig/joint_angles.csv", usecols=key + ["torso_angle_y", "torso_angle_x"])
    n0 = len(L)
    L = L.merge(J, on=key, validate="one_to_one")
    if len(L) != n0:
        raise ValueError("landmarks / joint_angles merge lost rows")
    poi = pd.read_csv(DATA / "poi/poi_metrics.csv", usecols=["session_pitch", "p_throws"])
    meta = pd.read_csv(DATA / "metadata.csv", usecols=["session", "session_pitch", "pitch_speed_mph",
                                                       "session_mass_kg", "session_height_m"])
    info = poi.merge(meta, on="session_pitch", validate="one_to_one").set_index("session_pitch")
    rows, curves, skipped = [], {}, []
    for sp, g in L.groupby("session_pitch"):
        if sp not in info.index or g[["fp_poi_time", "MER_time", "BR_time"]].iloc[0].isna().any():
            skipped.append((sp, "missing event / metadata"))
            continue
        side = info.at[sp, "p_throws"]
        t = g["time"].to_numpy()
        pel = {a: (g[f"rear_hip_{a}"] + g[f"lead_hip_{a}"]).to_numpy() / 2 for a in "xy"}
        off = {"forward": g["thorax_prox_x"].to_numpy() - pel["x"],
               "arm_side": (g["thorax_prox_y"].to_numpy() - pel["y"]) * (-1 if side == "R" else 1)}
        if any(np.isnan(v).any() for v in off.values()):
            skipped.append((sp, "nan landmark"))
            continue
        fp, mer, br = g[["fp_poi_time", "MER_time", "BR_time"]].iloc[0]
        sh = (g["shoulder_jc_y"] - g["glove_shoulder_jc_y"]).to_numpy()
        if np.sign(np.interp(br, t, sh)) != (-1 if side == "R" else 1):
            raise ValueError(f"lateral sign check failed {sp}")
        r = {"session_pitch": sp, "session": info.at[sp, "session"], "mer_pct": (mer - fp) / (br - fp) * 100}
        # marker-based tilt of the thorax long axis (thorax_dist -> thorax_prox): angle out of the lab x-z
        # plane, glove side +, so forward flexion does not inflate it; used to read the sign of torso_angle_y
        # when the trunk faces home (MER, BR)
        d = np.column_stack([(g[f"thorax_prox_{a}"] - g[f"thorax_dist_{a}"]).to_numpy() for a in "xyz"])
        tilt = np.degrees(np.arcsin(d[:, 1] * (1 if side == "R" else -1) / np.linalg.norm(d, axis=1)))
        for ev, te in [("fp", fp), ("mer", mer), ("br", br)]:
            r[f"marker_glove_tilt_{ev}_deg"] = np.interp(te, t, tilt)
            for ax, v in off.items():
                r[f"{ax}_{ev}_cm"] = 100 * np.interp(te, t, v)
            for ang in ["torso_angle_y", "torso_angle_x"]:
                r[f"{ang}_{ev}"] = np.interp(te, t, g[ang].to_numpy())
        rows.append(r)
        curves[sp] = {ax: 100 * np.interp(fp + PHASE * (br - fp), t, v) for ax, v in off.items()}
    df = pd.DataFrame(rows).merge(info[["pitch_speed_mph", "session_mass_kg", "session_height_m"]],
                                  left_on="session_pitch", right_index=True)
    df.to_csv(OUT / "per_pitch.csv", index=False)
    pd.DataFrame(skipped, columns=["session_pitch", "reason"]).to_csv(OUT / "skipped.csv", index=False)

    # events
    A = df.drop(columns="session_pitch").groupby("session").mean()
    y = A["pitch_speed_mph"].to_numpy()
    B = np.column_stack([np.ones(len(A)), A["session_mass_kg"], A["session_height_m"]])
    W = df.drop(columns="session_pitch").groupby("session").transform(lambda s: s - s.mean())
    rng, res = np.random.default_rng(0), []
    idx = rng.integers(0, len(A), (NB, len(A)))
    for ax in AXES:
        for ev in ["fp", "mer", "br"]:
            c = f"{ax}_{ev}_cm"
            v = A[c].to_numpy()
            bs = [np.corrcoef(v[i], y[i])[0, 1] for i in idx]
            slope = np.linalg.lstsq(np.column_stack([B, v]), y, rcond=None)[0][-1]
            res.append({"axis": ax, "event": ev, "pitches": len(df), "athletes": len(A),
                        "mean_cm": v.mean(), "between_sd_cm": v.std(ddof=1),
                        "share_arm_or_forward": (v > 0).mean(),
                        "r_speed": np.corrcoef(v, y)[0, 1], "lo": np.percentile(bs, 2.5),
                        "hi": np.percentile(bs, 97.5),
                        "r_partial_body": np.corrcoef(resid(v, B), resid(y, B))[0, 1],
                        "mph_per_10cm_body_adj": 10 * slope,
                        "r_within": np.corrcoef(W[c], W["pitch_speed_mph"])[0, 1],
                        "r_pitch_vs_torso_angle_y": np.corrcoef(df[c], df[f"torso_angle_y_{ev}"])[0, 1],
                        "r_pitch_vs_torso_angle_x": np.corrcoef(df[c], df[f"torso_angle_x_{ev}"])[0, 1]})
            # offset vs the matching torso angle (forward -> anterior tilt x, arm_side -> lateral tilt y)
            ang = A[f"{'torso_angle_x' if ax == 'forward' else 'torso_angle_y'}_{ev}"].to_numpy()

            def partials(i):
                Bo = np.column_stack([B[i], ang[i]])
                Ba = np.column_stack([B[i], v[i]])
                return (np.corrcoef(resid(v[i], Bo), resid(y[i], Bo))[0, 1],
                        np.corrcoef(resid(ang[i], Ba), resid(y[i], Ba))[0, 1])

            po, pa = partials(np.arange(len(A)))
            pb = np.array([partials(i) for i in idx])
            res[-1].update({"matching_angle_r_speed": np.corrcoef(ang, y)[0, 1],
                            "r_offset_vs_matching_angle": np.corrcoef(v, ang)[0, 1],
                            "r_partial_body_angle": po, "lo_partial_body_angle": np.percentile(pb[:, 0], 2.5),
                            "hi_partial_body_angle": np.percentile(pb[:, 0], 97.5),
                            "angle_r_partial_body_offset": pa,
                            "lo_angle_partial": np.percentile(pb[:, 1], 2.5),
                            "hi_angle_partial": np.percentile(pb[:, 1], 97.5)})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "event_summary.csv", index=False)
    lateral_checks(df, A, B, y, idx)

    # curves
    sess = df.set_index("session_pitch")["session"]
    spd = A["pitch_speed_mph"]
    rcrit = stats.t.ppf(0.975, len(A) - 2)
    rcrit = rcrit / np.sqrt(len(A) - 2 + rcrit ** 2)
    curve_rows, clus_rows = [], []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    mer = df.groupby("session")["mer_pct"].mean()
    for k, ax in enumerate(AXES):
        M = pd.DataFrame({sp: c[ax] for sp, c in curves.items()}).T
        Ma = M.groupby(sess).mean().loc[A.index].to_numpy()
        yz = (spd.to_numpy() - spd.mean()) / spd.std()

        def rcurve(yv):
            Z = (Ma - Ma.mean(0)) / Ma.std(0)
            return (Z * ((yv - yv.mean()) / yv.std())[:, None]).mean(0)

        r = rcurve(yz)
        Mw = (M - M.groupby(sess).transform("mean")).to_numpy()
        yw = (df.set_index("session_pitch")["pitch_speed_mph"]
              - df.set_index("session_pitch").groupby("session")["pitch_speed_mph"].transform("mean")).loc[M.index]
        rw = np.array([np.corrcoef(Mw[:, i], yw)[0, 1] for i in range(101)])
        null = []
        for _ in range(NP):
            rp = rcurve(rng.permutation(yz))
            null.append(max([np.abs(rp[a:b + 1]).sum() for a, b in clusters(np.abs(rp) > rcrit)], default=0))
        null = np.array(null)
        for a, b in clusters(np.abs(r) > rcrit):
            mass = np.abs(r[a:b + 1]).sum()
            clus_rows.append({"axis": ax, "start_pct": a, "end_pct": b, "mean_r": r[a:b + 1].mean(),
                              "cluster_p": (null >= mass).mean()})
        for i in range(101):
            curve_rows.append({"axis": ax, "pct": i, "mean_cm": Ma[:, i].mean(), "sd_cm": Ma[:, i].std(ddof=1),
                               "r_between": r[i], "r_within": rw[i]})
        p = axs[k]
        p.axvspan(*np.percentile(mer, [25, 75]), color="0.9", lw=0)
        p.axvline(np.median(mer), color="0.5", lw=1, ls="--")
        p.axhline(0, color="0.6", lw=0.8)
        for s in (rcrit, -rcrit):
            p.axhline(s, color="0.75", lw=0.8, ls=":")
        p.plot(PHASE * 100, r, color=COLORS[ax], lw=2, label="between athletes")
        p.plot(PHASE * 100, rw, color=COLORS[ax], lw=1.2, ls="--", alpha=0.7, label="within athlete")
        p.set_title(AXES[ax], fontsize=10)
        p.set_xlabel("FP (fp_poi_time) -> BR, %")
        p.set_ylim(-0.5, 0.5)
        for side in ["top", "right"]:
            p.spines[side].set_visible(False)
    axs[0].set_ylabel("r with pitch speed")
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    axs[1].text(np.median(mer) + 1, 0.45, "MER median / IQR", fontsize=8, color="0.4")
    fig.suptitle("Upper thorax offset from pelvis centre (lab frame) vs pitch speed", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "trunk_over_pelvis_speed_r_curve.png", dpi=150)
    pd.DataFrame(curve_rows).to_csv(OUT / "curve.csv", index=False)
    pd.DataFrame(clus_rows).to_csv(OUT / "clusters.csv", index=False)

    print("pitches", len(df), "athletes", len(A), "skipped", len(skipped), "r crit", round(rcrit, 3))
    print(res[["axis", "event", "mean_cm", "between_sd_cm", "r_speed", "lo", "hi", "r_partial_body",
               "mph_per_10cm_body_adj", "r_within", "matching_angle_r_speed", "r_offset_vs_matching_angle",
               "r_partial_body_angle", "lo_partial_body_angle", "hi_partial_body_angle",
               "angle_r_partial_body_offset", "lo_angle_partial", "hi_angle_partial"]].round(2).to_string(index=False))
    print(pd.DataFrame(clus_rows).round(3).to_string(index=False) if clus_rows else "no clusters")


if __name__ == "__main__":
    main()
