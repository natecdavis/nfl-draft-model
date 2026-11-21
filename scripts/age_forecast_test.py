"""Complaint #2a (free-agency list, Nov 2026): the comps forecast over-forecasts players 31+ from the third season on
(README step 1: ~10-20%, 34+ 0.81). The linear age shift can't bend: decline speeds up with age, and old players' comps
are younger on average.

Production forecast (veteran.comps(), season in progress features), anchors 2014-24: a random 6000 plus every anchor
aged 30+ (weights undo the oversampling). Horizons k = 2..6 (the planner's contract seasons 1-5 with skip 1).
Variants, factors fit leaving the anchor's season out:
  band   forecast x f[band, k], bands <28 / 28-30 / 31-32 / 33+ (actual / forecast in the other seasons)
  ramp   forecast x (1 - b_k x max(age - 29, 0)), b_k forecast-weighted least squares, clipped to [0.3, 1]
Pre-set: adopt a variant if MSE pooled over k = 3..6 falls with a 95% interval (bootstrap by player) below 0, 31+
actual / forecast is within 0.9-1.1 at every k = 3..6, and under-31 calibration moves by <= 0.02 at every k.
Run: .venv/bin/python scripts/age_forecast_test.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import veteran as vt  # noqa: E402

KS = range(2, 7)
BANDS = [-1, 28, 31, 33, 99]
BL = ["<28", "28-30", "31-32", "33+"]


def main():
    pd.set_option("display.width", 220)
    C = vt.comps()
    A = C.A[C.A.Y.between(2014, 2024)]
    old = A[A.age >= 30]
    rest = A[A.age < 30].sample(6000, random_state=5)
    S = pd.concat([old, rest]).copy()
    S["w"] = np.where(S.age >= 30, 1.0, (A.age < 30).sum() / 6000)
    S["band"] = pd.cut(S.age, BANDS, labels=BL, right=False)
    for k in KS:
        S[f"f{k}"] = [C.outcomes(r, k)[0].mean() if pd.notna(r[f"y{k}"]) else np.nan for r in S.to_dict("records")]
    S.to_parquet(vt.OUT / "age_forecast_test.parquet")
    print(f"anchors {len(S)} ({len(old)} aged 30+)")
    cal = []
    for k in KS:
        d = S[S[f"y{k}"].notna()]
        for b, x in d.groupby("band", observed=True):
            cal.append({"k": k, "band": b, "n": len(x), "a/f": (x.w * x[f"y{k}"]).sum() / (x.w * x[f"f{k}"]).sum()})
    print("\ncurrent actual / forecast by age band (rows) and horizon (contract season = k - 1):")
    print(pd.DataFrame(cal).pivot(index="band", columns="k", values="a/f").round(2))
    print(pd.DataFrame(cal).pivot(index="band", columns="k", values="n"))

    out, fc_all = [], {}
    for k in KS:
        d = S[S[f"y{k}"].notna()]
        a, f, w, age = (d[f"y{k}"].to_numpy(), d[f"f{k}"].to_numpy(), d.w.to_numpy(), d.age.to_numpy())
        band = d.band.astype(str).to_numpy()
        corr = {"band": f.copy(), "ramp": f.copy()}
        for y in d.Y.unique():
            tr, te = d.Y.to_numpy() != y, d.Y.to_numpy() == y
            for b in BL:
                m = tr & (band == b)
                corr["band"][te & (band == b)] *= (w[m] * a[m]).sum() / (w[m] * f[m]).sum()
            x = np.maximum(age - 29, 0)
            rr = (a[tr] / np.maximum(f[tr], 1e-6)).clip(0, 5)
            ww = w[tr] * f[tr]
            bk = np.sum(ww * x[tr] * (1 - rr)) / max(np.sum(ww * x[tr] ** 2), 1e-9)
            corr["ramp"][te] *= np.clip(1 - bk * x[te], 0.3, 1.0)
        for name, fc in corr.items():
            fc_all[(name, k)] = pd.Series(fc, index=d.index)
        fc_all[("current", k)] = pd.Series(f, index=d.index)
    # pooled k = 3..6 MSE change, bootstrap by player
    for name in ["band", "ramp"]:
        rows = []
        for k in range(3, 7):
            d = S[S[f"y{k}"].notna()]
            rows.append(pd.DataFrame({"pid": d.pfr_id, "w": d.w, "e0": (d[f"y{k}"] - fc_all[("current", k)]) ** 2,
                                      "e1": (d[f"y{k}"] - fc_all[(name, k)]) ** 2}))
        E = pd.concat(rows)
        E["dw"] = E.w * (E.e1 - E.e0)
        g = E.groupby("pid").agg(dw=("dw", "sum"), w=("w", "sum"))
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(g), len(g))
            bs.append(g.dw.to_numpy()[i].sum() / g.w.to_numpy()[i].sum())
        mse0 = (E.w * E.e0).sum() / E.w.sum()
        print(f"\n{name}: pooled k 3-6 RMSE {np.sqrt(mse0) * 100:.3f} -> "
              f"{np.sqrt((E.w * E.e1).sum() / E.w.sum()) * 100:.3f}; MSE change {g.dw.sum() / g.w.sum() * 1e4:+.4f} "
              f"({np.percentile(bs, 2.5) * 1e4:+.4f} to {np.percentile(bs, 97.5) * 1e4:+.4f}) x1e-4")
        rows = []
        for k in KS:
            d = S[S[f"y{k}"].notna()]
            for lab, m in [("31+", d.age >= 31), ("<31", d.age < 31)]:
                x = d[m]
                rows.append({"k": k, "who": lab,
                             "current": (x.w * x[f"y{k}"]).sum() / (x.w * fc_all[("current", k)][m]).sum(),
                             name: (x.w * x[f"y{k}"]).sum() / (x.w * fc_all[(name, k)][m]).sum()})
        print(pd.DataFrame(rows).pivot(index="who", columns="k").round(3))
    # all-data factors (for the tool if adopted)
    fac = {}
    for k in KS:
        d = S[S[f"y{k}"].notna()]
        fac[k] = {b: float((x.w * x[f"y{k}"]).sum() / (x.w * x[f"f{k}"]).sum()) for b, x in d.groupby("band", observed=True)}
    print("\nall-data band factors:", {k: {b: round(v, 3) for b, v in f.items()} for k, f in fac.items()})


if __name__ == "__main__":
    main()
