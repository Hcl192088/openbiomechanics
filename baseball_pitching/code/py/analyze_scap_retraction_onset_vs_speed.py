"""Which part of arm / scapular opening relates to pitch speed: pulled back before the trunk turns
forward, or kept back after it turns?

Input: trunk_onset_per_pitch.csv from analyze_scap_retraction_trunk_onset_split.py
(onset = last frame in [FP-300 ms, FP] with trunk velocity <= threshold; FP = fp_poi_time;
angles counted from FP-300 ms to the angle peak before MER).
Trunk onset definitions: official torso_velo_z (official0, main; official50/100 sensitivity) and
marker thorax w_th (marker0).
Components: before = start -> onset; after = onset -> peak (= after_still_back + after_lag).
Per cut and segment:
  share  = cov(component, total) / var(total) across athlete means (before + after shares = 1)
  between: athlete-mean speed ~ z(before) + z(after) + z(mass) + z(height); bootstrap over athletes
  within : session-demeaned speed ~ demeaned components (pooled), bootstrap over athletes
Full results go to CSV; stdout prints a short summary.
"""
import numpy as np
import pandas as pd

from analyze_scap_retraction_active_or_inertial import DATA, OUT

COMPS = ["before", "after", "after_still_back", "after_lag", "total"]
NB = 2000


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def z(a):
    return (a - a.mean(0)) / a.std(0)


def variability(df):
    """How much each component differs between athletes vs between pitches of one athlete.
    between_sd = SD of athlete means; within_sd = pooled SD of pitches around their athlete mean;
    icc = one-way ANOVA ICC(1) = share of pitch-level variance that is between athletes."""
    rows = []
    for cut, d in df.groupby("cut"):
        for seg in ["arm", "scap"]:
            for c in COMPS + ["onset"]:
                col = "onset_ms" if c == "onset" else f"{seg}_{c}"
                g = d.groupby("session")[col]
                m, n = g.mean(), g.size()
                k = len(m)
                ssw = ((d[col] - g.transform("mean")) ** 2).sum()
                msw = ssw / (len(d) - k)
                msb = (n * (m - d[col].mean()) ** 2).sum() / (k - 1)
                n0 = (len(d) - (n ** 2).sum() / len(d)) / (k - 1)
                vb = max((msb - msw) / n0, 0.0)
                q = np.percentile(m, [10, 25, 50, 75, 90])
                rows.append((cut, seg, c, k, m.mean(), m.std(ddof=1), np.sqrt(vb), np.sqrt(msw),
                             vb / (vb + msw), *q))
    out = pd.DataFrame(rows, columns=["cut", "segment", "component", "athletes", "mean", "between_sd_raw",
                                      "between_sd_true", "within_sd", "icc", "p10", "p25", "p50", "p75", "p90"])
    out.to_csv(OUT / "onset_split_variability.csv", index=False)
    s = out[out["cut"].isin(["official0", "marker0"]) & (out["component"] != "after_lag")]
    print(s[["cut", "segment", "component", "mean", "between_sd_true", "within_sd", "icc", "p10", "p90"]]
          .round(2).to_string(index=False))


def main():
    df = pd.read_csv(OUT / "trunk_onset_per_pitch.csv")
    meta = pd.read_csv(DATA / "metadata.csv",
                       usecols=["session_pitch", "pitch_speed_mph", "session_mass_kg", "session_height_m"])
    df = df.merge(meta, on="session_pitch", how="left", validate="many_to_one")
    if df[["pitch_speed_mph", "session_mass_kg", "session_height_m"]].isna().any().any():
        raise ValueError("missing speed / body size")
    rng, rows = np.random.default_rng(0), []
    for cut, d in df.groupby("cut"):
        for seg in ["arm", "scap"]:
            cols = [f"{seg}_{c}" for c in COMPS]
            ath = d.groupby("session")[cols + ["pitch_speed_mph", "session_mass_kg", "session_height_m",
                                                "onset_ms"]].mean()
            sess = ath.index.to_numpy()
            dm = d[cols + ["pitch_speed_mph"]] - d.groupby("session")[cols + ["pitch_speed_mph"]].transform("mean")
            dm["session"] = d["session"].to_numpy()
            by_s = {s: g for s, g in dm.groupby("session")}

            def stats(idx):
                a = ath.iloc[idx]
                tot = a[f"{seg}_total"].to_numpy()
                out = {}
                for c in ["before", "after", "after_still_back", "after_lag"]:
                    out[f"share_{c}"] = np.cov(a[f"{seg}_{c}"], tot)[0, 1] / tot.var(ddof=1)
                y = a["pitch_speed_mph"].to_numpy()
                for c in COMPS + ["onset_ms"]:
                    x = a[f"{seg}_{c}" if c != "onset_ms" else c].to_numpy()
                    out[f"r_between_{c}"] = np.corrcoef(x, y)[0, 1]
                body = a[["session_mass_kg", "session_height_m"]].to_numpy()
                b = ols(y, z(np.column_stack([a[f"{seg}_before"], a[f"{seg}_after"], body])))
                out["beta_between_before_mph_per_sd"], out["beta_between_after_mph_per_sd"] = b[0], b[1]
                b = ols(y, z(np.column_stack([a[f"{seg}_before"], a[f"{seg}_after_still_back"],
                                              a[f"{seg}_after_lag"], body])))
                out["beta_between_still_back_mph_per_sd"], out["beta_between_lag_mph_per_sd"] = b[1], b[2]
                w = pd.concat([by_s[s] for s in sess[idx]])
                yw = w["pitch_speed_mph"].to_numpy()
                for c in COMPS:
                    out[f"r_within_{c}"] = np.corrcoef(w[f"{seg}_{c}"], yw)[0, 1]
                bw = ols(yw, w[[f"{seg}_before", f"{seg}_after"]].to_numpy())
                out["slope_within_before_mph_per_10deg"] = 10 * bw[0]
                out["slope_within_after_mph_per_10deg"] = 10 * bw[1]
                return out

            est = stats(np.arange(len(ath)))
            boots = pd.DataFrame([stats(rng.integers(0, len(ath), len(ath))) for _ in range(NB)])
            for k, v in est.items():
                lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
                rows.append((cut, seg, len(d), len(ath), k, v, lo, hi))
    variability(df)
    res = pd.DataFrame(rows, columns=["cut", "segment", "pitches", "athletes", "metric", "est", "lo", "hi"])
    res.to_csv(OUT / "onset_split_vs_speed.csv", index=False)
    show = ["share_before", "r_between_before", "r_between_after", "beta_between_before_mph_per_sd",
            "beta_between_after_mph_per_sd", "r_within_before", "r_within_after"]
    s = res[res["metric"].isin(show) & res["cut"].isin(["official0", "marker0"])].copy()
    s["v"] = s.apply(lambda r: f"{r.est:.2f} ({r.lo:.2f},{r.hi:.2f})", axis=1)
    print(s.pivot(index="metric", columns=["cut", "segment"], values="v").loc[show].to_string())


if __name__ == "__main__":
    main()
