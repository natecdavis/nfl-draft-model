"""Good players back from an injury-shortened season: does the veteran forecast under-rate them?

Prompt: SF's optimizer cuts Fred Warner (2025 ankle, 35% of games; playing 96% of snaps in 2026). His forecast leans on
the short season. A first look (scratchpad) found good players with g1 < 0.5 who were back playing in the season in
progress realized 1.24x their forecast.

Setup as the tool uses it (planning a year ahead): anchors at Y with past-only inputs (value_ours_f) and weeks 1-11 of
season Y (p_share, p_gp); forecast the seasons after it (y2 = first contract season, y3). Comps leave the player and
the anchor season out (veteran.Comps.outcomes, exclude=True).
Groups (pre-set before fitting):
  good      v2 at or above the 80th percentile of his pricing group (anchors with value two seasons back)
  short     g1 < 0.5 (missed half of the last full season)
  back      p_gp >= 0.7 in the season in progress
Corrections, each a multiplicative factor on the forecast, fit leaving the anchor season out and forward (seasons
before Y only):
  A  returner factor: good & short & back
  B  A plus a not-back factor: good & short & not back
  C  continuous: forecast + b x (v2 - v1) x back x short (fit by OLS on the training seasons; pooled groups)
Adoption rule (same as earlier forecast changes): out-of-sample MSE on y2 better with a 95% bootstrap interval
(by player) below 0, the returner group's calibration closer to 1, and the forward run agreeing.
Run: .venv/bin/python scripts/injury_return_test.py   (forecasts cached in data/derived/injury_return_fc.parquet)
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402

fm.VALUE_COL = "value_ours_f"
CACHE = vt.OUT / "injury_return_fc.parquet"


def comps_past():
    A = vt.anchors(vt.season_table(), feat="value_ours_f")
    pp = []
    for Y in sorted(A.Y.unique()):
        Pt = vt.partial_snaps(int(Y))
        if len(Pt):
            pp.append(Pt[vt.P_FEATS].assign(Y=Y).reset_index(names="pfr_id"))
    A = A.merge(pd.concat(pp), on=["pfr_id", "Y"], how="left")
    A[vt.P_FEATS] = A[vt.P_FEATS].fillna(0.0)
    return vt.Comps(A)


def forecasts():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    C = comps_past()
    A = C.A[C.A.y2.notna()].copy()
    f2, f3 = [], []
    for i, r in enumerate(A.itertuples()):
        row = r._asdict()
        f2.append(C.outcomes(row, 2)[0].mean())
        f3.append(C.outcomes(row, 3)[0].mean() if pd.notna(r.y3) else np.nan)
        if i % 2000 == 0:
            print(f"  {i}/{len(A)}", flush=True)
    A["f2"], A["f3"] = f2, f3
    A.to_parquet(CACHE)
    return A


def flags(A):
    A = A.copy()
    A["v2r"] = A[A.v2 > 0].groupby("grp").v2.rank(pct=True)
    A["good"] = A.v2r.fillna(0) >= 0.8
    A["short"] = A.g1 < 0.5
    A["back"] = A.p_gp >= 0.7
    A["ret"] = A.good & A.short & A.back
    A["nb"] = A.good & A.short & ~A.back
    A["gap"] = (A.v2.fillna(0) - A.v1).clip(lower=0) * A.back * A.short
    return A


def ratio(d, y="y2", f="f2"):
    return d[y].sum() / d[f].sum() if d[f].sum() > 0 else 1.0


def corrected(A, train, variant, y="y2", f="f2"):
    """Corrected forecast for rows A, factors from `train`."""
    out = A[f].copy()
    if variant in ("A", "B"):
        out[A.ret] *= ratio(train[train.ret], y, f)
    if variant == "B":
        out[A.nb] *= ratio(train[train.nb], y, f)
    if variant == "C":
        d = train[train.gap > 0]
        b = float(np.dot(d.gap, d[y] - d[f]) / np.dot(d.gap, d.gap))
        out = out + b * A.gap
    return out.clip(lower=0)


def run(A, scheme, y="y2", f="f2"):
    A = A[A[y].notna()]
    res = {v: pd.Series(index=A.index, dtype=float) for v in "ABC"}
    for Y in sorted(A.Y.unique()):
        te = A.Y == Y
        tr = (A.Y != Y) if scheme == "loso" else (A.Y < Y)
        if scheme == "forward" and A[tr].ret.sum() < 30:
            for v in "ABC":
                res[v][te] = A.loc[te, f]
            continue
        for v in "ABC":
            res[v][te] = corrected(A[te], A[tr], v, y, f)
    return A, res


def report(A, res, lab, y="y2", f="f2"):
    base = (A[y] - A[f]) ** 2
    ids = A.pfr_id.unique()
    rng = np.random.default_rng(0)
    groups = A.groupby("pfr_id").indices
    print(f"\n[{lab}] {y}: {len(A)} anchors, returners {int(A.ret.sum())}, not back {int(A.nb.sum())}; "
          f"base RMSE {np.sqrt(base.mean()) * 100:.3f} % cap; returner calibration {ratio(A[A.ret], y, f):.2f}, "
          f"not back {ratio(A[A.nb], y, f):.2f}")
    for v, fc in res.items():
        dd = ((A[y] - fc) ** 2 - base).to_numpy()
        bs = []
        for _ in range(1000):
            pick = rng.choice(len(ids), len(ids))
            ix = np.concatenate([groups[ids[i]] for i in pick])
            bs.append(dd[ix].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cal = A.loc[A.ret, y].sum() / fc[A.ret].sum()
        cal_nb = A.loc[A.nb, y].sum() / fc[A.nb].sum()
        print(f"  {v}: MSE change {dd.mean() * 1e4:+.4f} x1e-4 (95% {lo * 1e4:+.4f} to {hi * 1e4:+.4f}); "
              f"returner calibration {cal:.2f}, not back {cal_nb:.2f}; returner RMSE "
              f"{np.sqrt(((A[y] - A[f])[A.ret] ** 2).mean()) * 100:.3f} -> {np.sqrt(((A[y] - fc)[A.ret] ** 2).mean()) * 100:.3f}")


def main():
    A = flags(forecasts())
    print("returner factor fit on all seasons:", round(ratio(A[A.ret]), 3), "| by group:",
          A[A.ret].groupby("grp").apply(ratio).round(2).to_dict())
    for scheme in ("loso", "forward"):
        for y, f in (("y2", "f2"), ("y3", "f3")):
            B, res = run(A, scheme, y, f)
            report(B, res, scheme, y, f)


if __name__ == "__main__":
    main()
