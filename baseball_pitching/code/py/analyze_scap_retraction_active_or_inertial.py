"""Scapular retraction / arm horizontal abduction: pulled back actively or left behind by the trunk?

Markers: raw C3D, 20 Hz zero-lag Butterworth (4th order); left-handers mirrored (y -> -y, L/R swapped).
Thorax frame from CLAV/STRN/C7/T10 only (not the shoulder line, which itself moves with the scapula):
  up = mid(C7,CLAV)-mid(T10,STRN); fwd = mid(CLAV,STRN)-mid(C7,T10) orthogonalised;
  lat = fwd x up (throwing side).
hab = -atan2(humerus.fwd, humerus.lat), opening +   (humerus = elbow_mid - SHO marker)
ret = -atan2(clav.fwd, clav.lat), retraction +      (clav = SHO - CLAV; acromion proxy for the scapula)
Space decomposition about the thorax up axis (forward rotation +):
  d(hab)/dt = w_thorax - w_arm ; w_arm < 0: arm really rotates backward in space;
  0 <= w_arm < w_thorax: arm merely lags the trunk.
Kinetics: official shoulder_thorax_moment_z (horizontal adduction +) and shoulder_thorax_force_y
(anterior +), checked per handedness against point-mass inverse dynamics of arm + hand + ball.
Work: -M*d(hab)/dt > 0: net moment pulls back while opening (active);
      < 0: net moment resists while the arm is being opened (inertial/stretch).
Scapula: force_y * acromion forward velocity in the thorax frame, same logic.
Windows: opening = minimum -> peak inside [max(PKH, FP-350 ms), MER], humerus < 60 deg below horizontal.
FP = fp_poi_time; BR = BR_time. Full results go to CSV.
"""
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
OUT = HERE / "scap_retraction_active_or_inertial_outputs"
FS = 360.0
B, A = butter(4, 20 / (FS / 2))
TRUNK = ["CLAV", "STRN", "C7", "T10"]
ARM = ["SHO", "ELB", "MELB", "WRA", "WRB", "FIN"]
G = np.array([0, 0, -9.81])


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def ddt(x):
    return np.gradient(x, 1 / FS, axis=0)


def load(row):
    c = ezc3d.c3d(str(DATA / "c3d" / f"{int(row['user']):06d}" / str(row["filename_new"])))
    lab = [str(x).strip() for x in c["parameters"]["POINT"]["LABELS"]["value"]]
    rate = float(c["header"]["points"]["frame_rate"])
    if rate != FS or c["parameters"]["POINT"]["UNITS"]["value"][0] != "m":
        raise ValueError("unexpected C3D rate/unit")
    side, pts = row["p_throws"], {}
    for m in TRUNK + ARM:
        name = m if m in TRUNK else side + m
        if name not in lab:
            return None
        x = c["data"]["points"][:3, lab.index(name), :].T.astype(float)
        x[(x == 0).all(axis=1)] = np.nan
        pts[m] = x
    t = np.arange(len(pts["CLAV"])) / FS
    w = (t >= row["fp_poi_time"] - 0.37) & (t <= row["BR_time"] + 0.02)
    for m, x in pts.items():
        if np.isnan(x[w]).any():
            return None
        x = pd.DataFrame(x).interpolate(limit_direction="both").to_numpy()
        x = filtfilt(B, A, x, axis=0)
        pts[m] = x * np.array([1, -1, 1]) if side == "L" else x
    return t, pts


def analyse(r, g, t, P):
    up = unit((P["C7"] + P["CLAV"] - P["T10"] - P["STRN"]) / 2)
    f0 = (P["CLAV"] + P["STRN"] - P["C7"] - P["T10"]) / 2
    fwd = unit(f0 - dot(f0, up)[:, None] * up)
    lat = np.cross(fwd, up)
    el, wr = (P["ELB"] + P["MELB"]) / 2, (P["WRA"] + P["WRB"]) / 2
    hum, cl = el - P["SHO"], P["SHO"] - P["CLAV"]

    def angle(v):
        return np.degrees(-np.unwrap(np.arctan2(dot(v, fwd), dot(v, lat))))

    hab, ret = angle(hum), angle(cl)
    horiz = np.hypot(dot(hum, fwd), dot(hum, lat))
    below = np.degrees(np.arctan2(-dot(hum, up), horiz))
    # forward rotation of the throwing side (+): d(lat)/dt projected on fwd
    w_th = np.degrees(dot(ddt(lat), fwd))
    dh, dr = ddt(hab), ddt(ret)
    m = r["session_mass_kg"]
    seg = [(0.0271 * m, P["SHO"] + 0.577 * hum), (0.0162 * m, el + 0.457 * (wr - el)),
           (0.0061 * m + 0.145, P["FIN"])]
    F = sum(mi * (ddt(ddt(x)) - G) for mi, x in seg)
    Mv = sum(np.cross(x - P["SHO"], mi * (ddt(ddt(x)) - G)) for mi, x in seg)
    Mg = sum(np.cross(x - P["SHO"], -mi * G) for mi, x in seg)

    def sig(col):
        return np.interp(t, g["time"], g[col])

    S = {"hab": hab, "ret": ret, "dhab": dh, "dret": dr, "w_th": w_th, "w_arm": w_th - dh,
         "w_clav": w_th - dr, "below": below, "Mz": sig("shoulder_thorax_moment_z"),
         "Fy": sig("shoulder_thorax_force_y"), "hab_official": sig("shoulder_angle_x"),
         "Mz_id": dot(Mv, up), "Mz_gravity": dot(Mg, up), "Fy_id": dot(F, fwd),
         "acro_v": ddt(dot(cl, fwd)),
         "Fy_gravity": dot(sum(-mi * G for mi, _ in seg) * np.ones((len(t), 1)), fwd),
         # -(sum m r x a_shoulder).up : turning effect on the arm of the shoulder being accelerated
         "shoulder_accel_M": -dot(sum(np.cross(x - P["SHO"], mi * ddt(ddt(P["SHO"]))) for mi, x in seg), up)}
    S["elbow_err"] = el[:, 0] - sig("elbow_jc_x")
    return S


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def main():
    OUT.mkdir(exist_ok=True)
    key = ["session_pitch", "time"]
    poi = pd.read_csv(DATA / "poi/poi_metrics.csv", usecols=["session_pitch", "p_throws", "pitch_speed_mph"])
    meta = pd.read_csv(DATA / "metadata.csv",
                       usecols=["user", "session", "session_pitch", "session_mass_kg", "filename_new"])
    an = pd.read_csv(DATA / "full_sig/joint_angles.csv",
                     usecols=key + ["shoulder_angle_x", "pkh_time", "fp_poi_time", "MER_time", "BR_time"])
    km = pd.read_csv(DATA / "full_sig/forces_moments.csv",
                     usecols=key + ["shoulder_thorax_moment_z", "shoulder_thorax_force_y"])
    lm = pd.read_csv(DATA / "full_sig/landmarks.csv", usecols=key + ["elbow_jc_x"])
    sig = an.merge(km, on=key, validate="one_to_one").merge(lm, on=key, validate="one_to_one")
    if len(sig) != len(an):
        raise ValueError("merge lost rows")
    ev = sig.groupby("session_pitch")[["pkh_time", "fp_poi_time", "MER_time", "BR_time"]].first().reset_index()
    tab = poi.merge(meta, on="session_pitch", validate="one_to_one").merge(ev, on="session_pitch", validate="one_to_one")
    groups = dict(tuple(sig.groupby("session_pitch")))
    rows, curves, val, skipped = [], [], [], []
    for _, r in tab.iterrows():
        sp, fp = r["session_pitch"], r["fp_poi_time"]
        loaded = load(r)
        if loaded is None:
            skipped.append(sp)
            continue
        t, P = loaded
        S = analyse(r, groups[sp], t, P)
        pre = (t >= fp - 0.15) & (t < fp)
        post = (t >= fp) & (t <= r["MER_time"])
        v = {"session_pitch": sp, "hand": r["p_throws"], "elbow_x_err_m": np.mean(np.abs(S["elbow_err"][pre | post]))}
        for nm, k in [("pre", pre), ("post", post)]:
            v[f"r_hab_{nm}"] = corr(S["hab"][k], S["hab_official"][k])
            for q in ("Mz", "Fy"):
                v[f"r_{q}_{nm}"] = corr(S[q][k], S[q + "_id"][k])
                v[f"official_{q}_{nm}"] = np.mean(S[q][k])
                v[f"id_{q}_{nm}"] = np.mean(S[q + "_id"][k])
        val.append(v)
        res = {"session_pitch": sp, "session": r["session"], "speed": r["pitch_speed_mph"], "hand": r["p_throws"]}
        iw = np.where((t >= max(r["pkh_time"], fp - 0.35)) & (t <= r["MER_time"]) & (S["below"] < 60))[0]
        for nm, x, vel, loads in [
            ("arm", S["hab"], np.radians(S["dhab"]), {"off": S["Mz"], "id": S["Mz_id"]}),
            ("scap", S["ret"], -S["acro_v"], {"off": S["Fy"], "id": S["Fy_id"]})]:
            pk = iw[np.argmax(x[iw])]
            bef = iw[iw <= pk]
            st = bef[np.argmin(x[bef])]
            s = slice(st, pk + 1)
            res[f"{nm}_start_ms"], res[f"{nm}_peak_ms"] = (t[st] - fp) * 1e3, (t[pk] - fp) * 1e3
            res[f"{nm}_rise_deg"] = x[pk] - x[st]
            if pk - st < 4:
                continue
            res[f"{nm}_trunk_fwd_deg"] = np.trapezoid(S["w_th"][s], t[s])
            res[f"{nm}_space_fwd_deg"] = res[f"{nm}_trunk_fwd_deg"] - res[f"{nm}_rise_deg"]
            res[f"{nm}_rise_share_before_fp"] = (x[min(pk, np.searchsorted(t, fp))] - x[st]) / res[f"{nm}_rise_deg"]
            for k, ld in loads.items():
                p = -ld[s] * vel[s]
                op = vel[s] > 0
                res[f"{nm}_{k}_active_j"] = np.trapezoid(np.where(op, np.maximum(p, 0), 0), t[s])
                res[f"{nm}_{k}_resist_j"] = np.trapezoid(np.where(op, np.minimum(p, 0), 0), t[s])
                res[f"{nm}_{k}_frac_pull"] = np.mean(ld[s][op] < 0) if op.any() else np.nan
        # Phase split at the last moment the segment stops moving backward in space before its peak.
        for start_ms in (-200, -300):
            for nm, x, wsp, load_, grav in [("arm", S["hab"], S["w_arm"], S["Mz"], S["Mz_gravity"]),
                                            ("scap", S["ret"], S["w_clav"], S["Fy"], S["Fy_gravity"])]:
                a = np.searchsorted(t, max(fp + start_ms / 1e3, r["pkh_time"]))
                b = np.searchsorted(t, r["MER_time"])
                pk = a + int(np.argmax(x[a:b]))
                q = f"{nm}{-start_ms}_"
                res[q + "rise"] = x[pk] - x[a]
                neg = np.where(wsp[a:pk + 1] < 0)[0]
                c = a + int(neg[-1]) if len(neg) else a
                res[q + "split_ms"] = (t[c] - fp) * 1e3
                res[q + "rise_moving_back"] = x[c] - x[a]
                res[q + "rise_left_behind"] = x[pk] - x[c]
                A_, B_ = slice(a, c + 1), slice(c, pk + 1)
                if c - a > 3:
                    net = load_ - grav
                    res[q + "A_load_net_of_gravity"] = net[A_].mean()
                    res[q + "A_first_half_net"] = net[a:(a + c) // 2 + 1].mean()
                    res[q + "A_second_half_net"] = net[(a + c) // 2:c + 1].mean()
                if pk - c > 3:
                    res[q + "B_load"] = load_[B_].mean()
                    res[q + "B_frac_forward_load"] = np.mean(load_[B_] > 0)
                    res[q + "B_trunk_fwd"] = np.trapezoid(S["w_th"][B_], t[B_])
        k = (t >= fp - 0.06) & (t <= fp)
        res["trunk_acc_pre60"] = np.polyfit(t[k], S["w_th"][k], 1)[0]
        res["late_ret_m60_to_peak"] = S["ret"][(t >= fp - .06) & (t <= r["MER_time"])].max() - np.interp(fp - .06, t, S["ret"])
        res["late_hab_m60_to_peak"] = S["hab"][(t >= fp - .06) & (t <= r["MER_time"])].max() - np.interp(fp - .06, t, S["hab"])
        rows.append(res)
        for ms in range(-350, 101, 10):
            q = fp + ms / 1e3
            c = {"session_pitch": sp, "session": r["session"], "ms": ms}
            c.update({k: np.interp(q, t, x) for k, x in S.items() if k != "elbow_err"})
            curves.append(c)
    val, df, cu = pd.DataFrame(val), pd.DataFrame(rows), pd.DataFrame(curves)
    val.to_csv(OUT / "validation.csv", index=False)
    df.to_csv(OUT / "per_pitch.csv", index=False)
    cu.to_csv(OUT / "curves_long.csv", index=False)
    pd.Series(skipped, name="session_pitch").to_csv(OUT / "skipped.csv", index=False)
    mean = cu.groupby(["session", "ms"]).mean(numeric_only=True).groupby("ms").mean(numeric_only=True)
    mean.to_csv(OUT / "curves_athlete_mean.csv")
    print("pitches", len(df), "skipped", len(skipped), "athletes", df["session"].nunique())
    print(val.groupby("hand")[["r_hab_pre", "r_Mz_pre", "r_Fy_pre"]].median().round(3).to_string())
    cols = ["hab", "ret", "w_th", "w_arm", "w_clav", "Mz", "Mz_gravity", "shoulder_accel_M", "Fy", "Fy_gravity", "below"]
    print(mean.loc[mean.index % 50 == 0, cols].round(1).to_string())
    ath = df.groupby("session").mean(numeric_only=True)
    ath.to_csv(OUT / "per_athlete.csv")
    rng = np.random.default_rng(0)
    out = []
    for c in ath.columns:
        x = ath[c].dropna().to_numpy()
        bs = x[rng.integers(0, len(x), (2000, len(x)))].mean(axis=1)
        out.append((c, len(x), x.mean(), *np.percentile(bs, [2.5, 97.5]), (x < 0).mean()))
    summ = pd.DataFrame(out, columns=["metric", "n", "mean", "lo", "hi", "share_neg"])
    summ.to_csv(OUT / "summary.csv", index=False)
    sel = summ[summ["metric"].str.match(r"(arm|scap)(200|300)_|late_")]
    print(sel.round(2).to_string(index=False))
    for y in ("late_ret_m60_to_peak", "late_hab_m60_to_peak"):
        d = df[["session", "trunk_acc_pre60", y]].dropna()
        wi = d[["trunk_acc_pre60", y]] - d.groupby("session")[["trunk_acc_pre60", y]].transform("mean")
        print(y, "vs trunk acc: between r", round(corr(ath["trunk_acc_pre60"], ath[y]), 2), "within r", round(corr(wi["trunk_acc_pre60"], wi[y]), 2))
    for y in ("scap200_rise_moving_back", "scap200_rise_left_behind", "arm200_rise_moving_back", "arm200_rise_left_behind"):
        print(y, "vs speed r", round(corr(ath[y], ath["speed"]), 2))


if __name__ == "__main__":
    main()
