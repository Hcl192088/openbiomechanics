"""Scapular retraction / horizontal abduction split at trunk forward-rotation onset.

Onset = last frame in [FP-300 ms, FP] with trunk velocity <= threshold (FP = fp_poi_time); the trunk
stays above the threshold from there to FP. Pitches never at/below the threshold are excluded and counted.
Sources: official torso_velo_z (main, 0 deg/s; sensitivity 50, 100) and marker thorax w_th (0).
Rise start = FP-300 ms, end = angle peak before MER. After onset the rise is also split into frames where
the segment still rotates backward in space (w_space < 0) and frames where it goes forward but lags.
Loads are net of gravity (negative = backward). Builds on analyze_scap_retraction_active_or_inertial.py.
"""
import numpy as np
import pandas as pd

import analyze_scap_retraction_active_or_inertial as base


def main():
    key, D = ["session_pitch", "time"], base.DATA
    fs = D / "full_sig"
    poi = pd.read_csv(D / "poi/poi_metrics.csv", usecols=["session_pitch", "p_throws"])
    meta = pd.read_csv(D / "metadata.csv",
                       usecols=["user", "session", "session_pitch", "session_mass_kg", "filename_new"])
    sig = pd.read_csv(fs / "joint_angles.csv",
                      usecols=key + ["shoulder_angle_x", "pkh_time", "fp_poi_time", "MER_time", "BR_time"])
    n0 = len(sig)
    for f, cols in [("forces_moments", ["shoulder_thorax_moment_z", "shoulder_thorax_force_y"]),
                    ("landmarks", ["elbow_jc_x"]), ("joint_velos", ["torso_velo_z"])]:
        sig = sig.merge(pd.read_csv(fs / f"{f}.csv", usecols=key + cols), on=key, validate="one_to_one")
    if len(sig) != n0:
        raise ValueError("merge lost rows")
    ev = sig.groupby("session_pitch")[["pkh_time", "fp_poi_time", "MER_time", "BR_time"]].first().reset_index()
    tab = poi.merge(meta, on="session_pitch", validate="one_to_one").merge(ev, on="session_pitch", validate="one_to_one")
    groups = dict(tuple(sig.groupby("session_pitch")))
    rows, rs, total = [], [], 0
    for _, r in tab.iterrows():
        loaded = base.load(r)
        if loaded is None:
            continue
        total += 1
        t, P = loaded
        g = groups[r["session_pitch"]]
        S = base.analyse(r, g, t, P)
        off = np.interp(t, g["time"], g["torso_velo_z"])
        fp = r["fp_poi_time"]
        a, f, e = (np.searchsorted(t, x) for x in (fp - 0.3, fp, r["MER_time"]))
        rs.append(base.corr(off[a:f], S["w_th"][a:f]))
        for name, w, thr in [("official0", off, 0), ("official50", off, 50), ("official100", off, 100),
                             ("marker0", S["w_th"], 0)]:
            low = np.where(w[a:f + 1] <= thr)[0]
            if len(low) == 0:
                continue
            c = a + int(low[-1])
            out = {"cut": name, "session": r["session"], "session_pitch": r["session_pitch"],
                   "onset_ms": (t[c] - fp) * 1e3}
            for nm, x, wsp, load, grav in [
                    ("arm", S["hab"], S["w_arm"], S["Mz"], S["Mz_gravity"]),
                    ("scap", S["ret"], S["w_clav"], S["Fy"], S["Fy_gravity"])]:
                pk = a + int(np.argmax(x[a:e]))
                cc = min(c, pk)
                neg = np.where(wsp[cc:pk + 1] < 0)[0]
                k = cc + int(neg[-1]) if len(neg) else cc
                net = load - grav
                out.update({f"{nm}_total": x[pk] - x[a], f"{nm}_before": x[cc] - x[a], f"{nm}_after": x[pk] - x[cc],
                            f"{nm}_after_still_back": x[k] - x[cc], f"{nm}_after_lag": x[pk] - x[k],
                            f"{nm}_space_vel_at_onset": wsp[c], f"{nm}_load_before": net[a:c + 1].mean()})
                if pk > c:
                    out[f"{nm}_load_after"] = net[c:pk + 1].mean()
            rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(base.OUT / "trunk_onset_per_pitch.csv", index=False)
    rng, summ = np.random.default_rng(0), []
    for cut, d in df.groupby("cut"):
        ath = d.drop(columns=["cut", "session_pitch"]).groupby("session").mean()
        for col in ath:
            x = ath[col].dropna().to_numpy()
            bs = x[rng.integers(0, len(x), (2000, len(x)))].mean(axis=1)
            summ.append((cut, col, len(d), len(x), x.mean(), *np.percentile(bs, [2.5, 97.5]), (x < 0).mean()))
    summ = pd.DataFrame(summ, columns=["cut", "metric", "pitches", "athletes", "mean", "lo", "hi", "share_neg"])
    summ.to_csv(base.OUT / "trunk_onset_summary.csv", index=False)
    print("total", total, "median r official vs marker trunk velocity", round(float(np.median(rs)), 3))
    print(df.groupby("cut")["onset_ms"].describe()[["count", "25%", "50%", "75%"]].round(0).to_string())
    print(summ[summ["cut"] == "official0"].drop(columns="cut").round(2).to_string(index=False))
    sel = summ[summ["metric"].isin(["arm_before", "arm_after", "scap_before", "scap_after"])]
    print(sel.pivot(index="cut", columns="metric", values="mean").round(1).to_string())


if __name__ == "__main__":
    main()
