"""Should the free-agent contract model weight recent signings more? (Spotrac check: older veterans signed for 0.89x
our out-of-sample prediction in 2024-26)

Forward-chained, as the model is used: for each signing year Y in 2020-2026, fit on UFA signings before Y only (lag 2
features, as for 2027), predict Y. Variants: equal weights; weights 0.5 ** ((Y - 1 - year) / h), half-life h = 2, 4, 6.
The mean scale factor (exp(log prediction) -> mean) is re-estimated the same forward way: actual / predicted over the
forward predictions of the years before Y (the first years use the full-sample 1.185 as a start).
Pre-set rule: adopt a half-life if, over 2022-26, (a) median |error| and R2 (log APY) are no worse than equal weights
and (b) the age 31+ bias (actual / predicted) moves toward 1; otherwise keep equal weights.
Run: .venv/bin/python scripts/contract_recency.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402

YEARS = range(2020, 2027)
VARIANTS = {"equal": None, "h2": 2, "h4": 4, "h6": 6}


def main():
    pd.set_option("display.width", 200)
    D = fm.training(2)
    out = []
    for name, h in VARIANTS.items():
        P = []
        for Y in YEARS:
            tr, te = D[D.Y < Y], D[D.Y == Y].copy()
            m = fm.fit(tr, half_life=h, ref_year=Y - 1)
            te["typ"] = np.exp(m["log_apy"].predict(te[fm.FEATS]))
            P.append(te)
        P = pd.concat(P)
        # forward mean-scaling: factor for year Y from earlier forward years (start 1.185)
        P["smear"] = 1.185
        for Y in YEARS:
            prev = P[P.Y < Y]
            if len(prev) > 200:
                P.loc[P.Y == Y, "smear"] = prev.apy.sum() / prev.typ.sum()
        P["pred"] = P.typ * P.smear
        P["variant"] = name
        out.append(P)
    A = pd.concat(out)
    A = A[A.Y >= 2022]
    rng = np.random.default_rng(0)

    def ratio(d):
        x = d.apy.sum() / d.pred.sum()
        b = [(lambda s: s.apy.sum() / s.pred.sum())(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(500)]
        return f"{x:.2f} ({np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f})"
    rows = []
    for name, d in A.groupby("variant", sort=False):
        e = np.log(d.apy) - np.log(d.typ)
        r2 = 1 - ((np.log(d.apy) - np.log(d.typ)) ** 2).sum() / ((np.log(d.apy) - np.log(d.apy).mean()) ** 2).sum()
        rows.append({"variant": name, "n": len(d), "R2 log": round(r2, 3),
                     "median |error|": f"{np.median(np.abs(d.apy / d.pred - 1)):.1%}",
                     "all": ratio(d), "age 31+": ratio(d[d.age >= 31]), "28-30": ratio(d[(d.age >= 28) & (d.age < 31)]),
                     "<=27": ratio(d[d.age < 28]),
                     "off rookie deal, starter": ratio(d[(d.prev_rookie == 1) & (d.v1 >= 0.02)])})
    print("Forward-chained predictions of 2022-2026 UFA signings (actual / predicted, 95% CI):")
    print(pd.DataFrame(rows).to_string(index=False))
    for name, d in A.groupby("variant", sort=False):
        print(f"  {name}: by year all", d.groupby("Y").apply(lambda x: round(x.apy.sum() / x.pred.sum(), 2),
                                                          include_groups=False).to_dict(),
              "| 31+", d[d.age >= 31].groupby("Y").apply(lambda x: round(x.apy.sum() / x.pred.sum(), 2),
                                                         include_groups=False).to_dict())
    A.to_parquet(fm.OUT / "contract_recency.parquet")


if __name__ == "__main__":
    main()
