"""Complaint #3 (free-agency list, Nov 2026): our contract prices lag a rising market (2025-26 signings ran ~10% above
forward predictions, mostly younger players; recency weights didn't help since trees don't extrapolate).

Forward-chained as used in November before signing year Y: fit the lag-2 production model (fa_market.feats_for(2)) on UFA
signings before Y, predict Y. typ = exp(predicted log APY). Scale variants (prediction = typ x scale):
  fixed      production SMEAR[2] (1.153, leave-year-out over all years)
  cum        sum actual / sum typ over all earlier forward years
  last1      ... over year Y-1 only        last2   ... over Y-1 and Y-2
  last2_seg  last2 separately for players coming off a rookie deal and veterans
  lin        log(ratio by year) on year over earlier forward years, extrapolated to Y
Forward predictions 2017-2026 (training from 2015); scored on 2020-26.
Pre-set: adopt a variant if pooled MSE of APY (share of cap) falls vs fixed with a 95% bootstrap interval below 0, the
spread of year-level actual / predicted (sd of log) shrinks, and the median |actual / predicted - 1| is no more than
0.5 points worse. Run: .venv/bin/python scripts/contract_trend.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402

FWD, EVAL = range(2017, 2027), range(2020, 2027)


def main():
    pd.set_option("display.width", 220)
    D = fm.training(2)
    feats = fm.feats_for(2)
    P = []
    for Y in FWD:
        m = fm.fit(D[D.Y < Y], feats=feats)
        te = D[D.Y == Y].copy()
        te["typ"] = np.exp(m["log_apy"].predict(te[feats]))
        P.append(te)
    P = pd.concat(P, ignore_index=True)
    P["seg"] = np.where(P.prev_rookie == 1, "rookie", "vet")
    rat = lambda d: d.apy.sum() / d.typ.sum()  # noqa: E731
    yr = P.groupby("Y").apply(rat, include_groups=False)
    print("forward actual / typical by year (typ = exp(log pred)):", yr.round(3).to_dict())
    print("  off rookie deal:", P[P.seg == "rookie"].groupby("Y").apply(rat, include_groups=False).round(3).to_dict())
    print("  veterans:       ", P[P.seg == "vet"].groupby("Y").apply(rat, include_groups=False).round(3).to_dict())
    S = {}
    for Y in EVAL:
        prev = P[P.Y < Y]
        m = P.Y == Y
        S.setdefault("fixed", pd.Series(np.nan, index=P.index)).loc[m] = fm.SMEAR[2]
        S.setdefault("cum", pd.Series(np.nan, index=P.index)).loc[m] = rat(prev)
        S.setdefault("last1", pd.Series(np.nan, index=P.index)).loc[m] = rat(P[P.Y == Y - 1])
        S.setdefault("last2", pd.Series(np.nan, index=P.index)).loc[m] = rat(P[P.Y.between(Y - 2, Y - 1)])
        s2 = S.setdefault("last2_seg", pd.Series(np.nan, index=P.index))
        for g in ["rookie", "vet"]:
            s2.loc[m & (P.seg == g)] = rat(P[P.Y.between(Y - 2, Y - 1) & (P.seg == g)])
        ly = np.log(prev.groupby("Y").apply(rat, include_groups=False))
        b = np.polyfit(ly.index.to_numpy(float), ly.to_numpy(), 1)
        S.setdefault("lin", pd.Series(np.nan, index=P.index)).loc[m] = np.exp(np.polyval(b, Y))
    E = P[P.Y.isin(EVAL)].copy()
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(E), len(E)) for _ in range(2000)]
    e0 = (E.apy - E.typ * S["fixed"][E.index]) ** 2
    rows = []
    for name, s in S.items():
        pred = E.typ * s[E.index]
        e = (E.apy - pred) ** 2
        d = (e - e0).to_numpy()
        bs = [d[i].mean() for i in idx]
        yr_r = (E.apy.groupby(E.Y).sum() / pred.groupby(E.Y).sum())
        r_seg = {g: E.apy[E.seg == g].sum() / pred[E.seg == g].sum() for g in ["rookie", "vet"]}
        rows.append({"variant": name, "MSE x1e4": e.mean() * 1e4,
                     "change vs fixed (95%)": f"{d.mean() * 1e4:+.4f} ({np.percentile(bs, 2.5) * 1e4:+.4f} to "
                                              f"{np.percentile(bs, 97.5) * 1e4:+.4f})",
                     "median miss": f"{np.median(np.abs(E.apy / pred - 1)):.1%}",
                     "year a/p sd(log)": np.log(yr_r).std(), "all a/p": E.apy.sum() / pred.sum(),
                     "rookie a/p": r_seg["rookie"], "vet a/p": r_seg["vet"],
                     "2025": yr_r.get(2025), "2026": yr_r.get(2026), "scale 2027": None})
    R = pd.DataFrame(rows)
    # what each variant would use for 2027 (forward predictions through 2026)
    Y = 2027
    sc = {"fixed": fm.SMEAR[2], "cum": rat(P), "last1": rat(P[P.Y == Y - 1]), "last2": rat(P[P.Y.between(Y - 2, Y - 1)]),
          "last2_seg": "rookie %.3f / vet %.3f" % tuple(rat(P[P.Y.between(Y - 2, Y - 1) & (P.seg == g)]) for g in ["rookie", "vet"])}
    ly = np.log(P.groupby("Y").apply(rat, include_groups=False))
    sc["lin"] = float(np.exp(np.polyval(np.polyfit(ly.index.to_numpy(float), ly.to_numpy(), 1), Y)))
    R["scale 2027"] = R.variant.map(lambda v: sc[v] if isinstance(sc[v], str) else round(sc[v], 3))
    print(f"\nForward 2020-26 UFA signings (n {len(E)}):")
    print(R.round(3).to_string(index=False))
    P.to_parquet(fm.OUT / "contract_trend.parquet")


if __name__ == "__main__":
    main()
