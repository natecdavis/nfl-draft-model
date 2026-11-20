"""Do counting stats improve the free-agent contract model? (the market pays for stats beyond grades:
scripts/stat_premium.py found WR yards +1.86% of cap per SD, sacks, QB TDs, passes defended)

Variants (src/fa_market.py): base (season value from snaps x PFF-based quality, age, position, previous contract);
stats1 = base + last season's counting stats per 17 games (passing yds / TDs, rushing yds / TDs, receptions, receiving
yds / TDs, sacks, QB hits, tackles, INTs, passes defended); stats2 = + the season before too.
Checks, UFA signings 2015-2026:
  A leave-one-signing-year-out, lag 1 and lag 2 (R2 log APY, median miss, within 25%)
  B forward-chained (fit on earlier years only, predict 2020-26; lag 2, as used for 2027; scored 2022-26)
  C calibration by group and the biggest changes in 2027 predictions
Pre-set rule: adopt a stats variant if it beats base on R2 and median miss in B and in A at both lags.
Run: .venv/bin/python scripts/contract_stats.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402

VARIANTS = {"base": fm.FEATS_BASE, "stats1": fm.FEATS_STATS1, "stats2": fm.FEATS_STATS2}


def score(P, col):
    e = np.log(P.apy) - np.log(P[col])
    r2 = 1 - (e ** 2).sum() / ((np.log(P.apy) - np.log(P.apy).mean()) ** 2).sum()
    rel = np.abs(P.apy / (P[col] * (P.apy.sum() / P[col].sum())) - 1)  # miss after scaling to the mean
    return r2, np.median(rel), (rel < 0.25).mean()


def main():
    pd.set_option("display.width", 200)
    rows = []
    keep = {}
    for lag in (1, 2):
        D = fm.training(lag)
        for name, feats in VARIANTS.items():
            P = []
            for Y in sorted(D.Y.unique()):
                tr, te = D[D.Y != Y], D[D.Y == Y].copy()
                te["pred"] = np.exp(fm.fit(tr, feats=feats)["log_apy"].predict(te[feats]))
                P.append(te)
            P = pd.concat(P)
            keep[(lag, name)] = P
            r2, med, w25 = score(P, "pred")
            rows.append({"check": f"A leave-year-out lag {lag}", "variant": name, "R2": round(r2, 3),
                         "median miss": f"{med:.1%}", "within 25%": f"{w25:.0%}"})
    D = fm.training(2)
    for name, feats in VARIANTS.items():
        P = []
        for Y in range(2020, 2027):
            tr, te = D[D.Y < Y], D[D.Y == Y].copy()
            te["pred"] = np.exp(fm.fit(tr, feats=feats)["log_apy"].predict(te[feats]))
            P.append(te)
        P = pd.concat(P)
        P = P[P.Y >= 2022]
        keep[("fwd", name)] = P
        r2, med, w25 = score(P, "pred")
        rows.append({"check": "B forward 2022-26 lag 2", "variant": name, "R2": round(r2, 3),
                     "median miss": f"{med:.1%}", "within 25%": f"{w25:.0%}"})
    print(pd.DataFrame(rows).to_string(index=False))
    # C: by group, lag 2 leave-year-out: median miss base vs stats1
    print("\nC. median miss by group (lag 2 leave-year-out): base / stats1 / stats2")
    out = {}
    for g in fm.GROUPS:
        out[g] = []
        for name in VARIANTS:
            P = keep[(2, name)]
            d = P[P.grp == g]
            if len(d) > 20:
                out[g].append(f"{score(d, 'pred')[1]:.0%}")
        out[g] = " / ".join(out[g]) + f"  (n {len(keep[(2, 'base')][keep[(2, 'base')].grp == g])})"
    for g, v in out.items():
        print(f"  {g:5s} {v}")
    pd.concat([keep[(2, n)].assign(variant=n) for n in VARIANTS]).to_parquet(fm.OUT / "contract_stats.parquet")


if __name__ == "__main__":
    main()
