"""Price ranges for the contract models (#8): an 80% range and the chance a player costs more than a given APY.

Tool setting: lag 2 + weeks 1-11 (fa_market.feats_for(2)), production inputs. Residual = log(actual APY) -
log(typical prediction) (unsmeared). Forward-chained: every signing year 2016-26 predicted by a model fit on earlier
years only; a range for year Y is calibrated on the forward residuals of years before Y (>= 3 years of them), so
nothing from Y or later is used. Scored 2022-26.
Methods:
  pooled  - empirical 10th / 90th percentiles of all earlier residuals
  size    - same, within the predicted-APY tercile (cut points from the earlier predictions)
  group   - same, within position group
  gbm q   - LightGBM quantile regressions (alpha 0.1 / 0.9) on the same features, fit on earlier years
Checks: coverage of the 80% range overall and by segment (size tercile, age band, group, rookie deal before), mean
width (ratio high / low), and the probability scale: PIT (share of earlier residuals below this one) by decile -
a calibrated method puts ~10% in each.
  size x age - within predicted-APY tercile x age (<29 / 29+)   [attempt 2]
Attempt 2 (stated after attempt 1 failed; same rule): attempt 1 pooled every earlier forward residual, including
2016-18 whose models were fit on 1-3 signing years (too wide), and coverage ran high for older players. Attempt 2 pools
only years whose model had >= 4 training years (POOL_FROM = 2019) and adds the size x age method.
Attempt 1 (POOL_FROM 2016): nothing qualified - UFA size 0.84 overall, 31+ 0.92; EXT size 0.80, EDGE 0.67 (n 30).
Pre-set rule: a method qualifies if overall coverage is 0.77-0.83 and every segment with n >= 30 is 0.72-0.88; among
qualifiers the narrowest wins and goes into the tool. Run separately for free agents (UFA) and extensions (EXT).
Writes data/derived/contract_residuals.parquet (the forward residuals, for the tool).
Run: .venv/bin/python scripts/contract_ranges.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402

FEATS = fm.feats_for(2)
LO, HI = 0.10, 0.90
POOL_FROM = 2019
METHODS = ("pooled", "size", "group", "size x age", "gbm q")


def forward(D, sel):
    out = []
    for Y in range(2016, 2027):
        tr, te = D[D.Y < Y], D[(D.Y == Y)].copy()
        te = te[sel(te)]
        if te.empty:
            continue
        m = fm.fit(tr, feats=FEATS)["log_apy"]
        te["lp"] = m.predict(te[FEATS])
        out.append(te)
    P = pd.concat(out)
    P["res"] = np.log(P.apy) - P.lp
    return P


def gbm_q(D, Y, te, alpha):
    import lightgbm as lgb
    tr = D[D.Y < Y]
    m = lgb.LGBMRegressor(**{**fm.PARAMS, "objective": "quantile", "alpha": alpha}).fit(tr[FEATS], tr.log_apy)
    return m.predict(te[FEATS])


def segs(P):
    out = {"all": np.ones(len(P), bool)}
    for k, (a, b) in {"age <26": (0, 26), "age 26-28": (26, 29), "age 29-30": (29, 31), "age 31+": (31, 99)}.items():
        out[k] = P.age.between(a, b, inclusive="left").to_numpy()
    for t in range(3):
        out[f"size tercile {t + 1}"] = (P.size_t == t).to_numpy()
    out["rookie deal before"] = (P.prev_rookie == 1).to_numpy()
    out.update({f"grp {g}": (P.grp == g).to_numpy() for g in fm.GROUPS})
    return out


def evaluate(D, P, label):
    rows = []
    for Y in range(2022, 2027):
        te = P[P.Y == Y].copy()
        pool = P[(P.Y < Y) & (P.Y >= POOL_FROM)]
        cuts = np.quantile(pool.lp, [1 / 3, 2 / 3])
        te["size_t"] = np.searchsorted(cuts, te.lp)
        pool = pool.assign(size_t=np.searchsorted(cuts, pool.lp))
        te["sa"] = te.size_t * 2 + (te.age >= 29)
        pool = pool.assign(sa=pool.size_t * 2 + (pool.age >= 29))
        # pooled
        q = np.quantile(pool.res, [LO, HI])
        te["pooled_lo"], te["pooled_hi"] = q[0], q[1]
        te["pooled_pit"] = np.searchsorted(np.sort(pool.res), te.res) / len(pool)
        for key, col in (("size", "size_t"), ("group", "grp"), ("size x age", "sa")):
            lo, hi, pit = [], [], []
            for v, r in zip(te[col], te.res):
                rr = np.sort(pool[pool[col] == v].res.to_numpy())
                if len(rr) < 30:
                    rr = np.sort(pool.res.to_numpy())
                lo.append(np.quantile(rr, LO)); hi.append(np.quantile(rr, HI))
                pit.append(np.searchsorted(rr, r) / len(rr))
            te[f"{key}_lo"], te[f"{key}_hi"], te[f"{key}_pit"] = lo, hi, pit
        te["gbm q_lo"] = gbm_q(D, Y, te, LO) - te.lp
        te["gbm q_hi"] = gbm_q(D, Y, te, HI) - te.lp
        rows.append(te)
    T = pd.concat(rows)
    S = segs(T)
    res, verdict = {}, {}
    for m in METHODS:
        inside = ((T.res >= T[f"{m}_lo"]) & (T.res <= T[f"{m}_hi"])).to_numpy()
        cov = {k: (int(s.sum()), inside[s].mean()) for k, s in S.items() if s.sum() > 0}
        width = np.exp((T[f"{m}_hi"] - T[f"{m}_lo"]).clip(lower=0)).median()
        ok = 0.77 <= cov["all"][1] <= 0.83 and all(0.72 <= c <= 0.88 for k, (n, c) in cov.items() if n >= 30)
        res[m] = {k: f"{c:.2f}" for k, (n, c) in cov.items()}
        res[m]["median high/low"] = f"{width:.2f}"
        verdict[m] = (ok, width)
    n = {k: int(s.sum()) for k, s in S.items()}
    out = pd.DataFrame(res)
    out.insert(0, "n", pd.Series(n))
    print(f"\n{label}: 80% range coverage, forward 2022-26 (n {len(T)})")
    print(out.to_string())
    print("  PIT by decile (share of signings; calibrated = 0.10 each):")
    for m in METHODS[:-1]:
        h = np.histogram(T[f"{m}_pit"], bins=np.linspace(0, 1, 11))[0] / len(T)
        print(f"    {m:10s}", " ".join(f"{x:.2f}" for x in h))
    q = {m: w for m, (ok, w) in verdict.items() if ok}
    best = min(q, key=q.get) if q else None
    print(f"  qualify: {[m for m in q] or 'none'} -> {'use ' + best if best else 'no method adopted'}")
    return best


def main():
    pd.set_option("display.width", 220)
    out = []
    for label, D, sel in [("UFA", fm.training(2), lambda d: d.ufa == 1),
                          ("EXT", fm.training(2, kinds=("UFA", "Extension")), lambda d: d.ufa == 0)]:
        P = forward(D, sel)
        best = evaluate(D, P, label)
        out.append(P[["pfr_id", "Y", "grp", "age", "lp", "res"]].assign(model=label, method=best))
    R = pd.concat(out)
    R.to_parquet(fm.OUT / "contract_residuals.parquet")
    print(f"\nwrote {fm.OUT / 'contract_residuals.parquet'}: {len(R)} forward residuals")


if __name__ == "__main__":
    main()
