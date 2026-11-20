"""Check the free-agent contract model (src/fa_market.py): leave-one-signing-year-out, UFA signings 2015-2026.

For each year Y, fit on the other years, predict Y. Lag 1 = data through the season just ended (normal March use);
lag 2 = one season more of gap (planning a year ahead; what the app uses for March 2027 with data through 2025).
Baselines: previous contract APY only; value only (v1, v2, age, group) - both OLS on log APY, same folds.
Run: .venv/bin/python scripts/contract_predict.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402


def main():
    pd.set_option("display.width", 200)
    kinds = ("UFA", "Extension", "Franchise", "Transition") if "--all" in sys.argv else ("UFA",)
    for lag in (1, 2):
        D = fm.training(lag, kinds=kinds)
        P = []
        for Y in sorted(D.Y.unique()):
            tr, te = D[D.Y != Y], D[(D.Y == Y) & (D.ufa == 1)].copy()  # always scored on UFA signings
            te = te.join(fm.predict(fm.fit(tr, feats=fm.feats_for(lag)), te, fm.SMEAR[lag] if "--smeared" in sys.argv else 1.0))
            te["b_prev"] = np.exp(smf.ols("log_apy ~ np.log(prev_apy) + prev_rookie", tr).fit().predict(te))
            te["b_val"] = np.exp(smf.ols("log_apy ~ C(grp) + v1 + v2 + g1 + age + I(age**2)", tr).fit().predict(te))
            P.append(te)
        P = pd.concat(P)
        print(f"\n=== lag {lag}: {len(P)} UFA signings 2015-2026, median APY {P.apy.median() * 100:.2f}% of cap ===")
        for c in ["apy_pred", "b_prev", "b_val"]:
            r2 = 1 - ((P.log_apy - np.log(P[c])) ** 2).sum() / ((P.log_apy - P.log_apy.mean()) ** 2).sum()
            print(f"  {c:9s} R2 (log APY) {r2:.3f}  MAE {np.abs(P.apy - P[c]).mean() * 100:.2f}% of cap  "
                  f"median |error| {np.median(np.abs(P.apy / P[c] - 1)):.0%}  within 25%: {(np.abs(P.apy / P[c] - 1) < 0.25).mean():.0%}")
        P["dec"] = pd.qcut(P.apy_pred, 5, labels=False)
        print("  calibration by predicted quintile (% of cap):")
        print(P.groupby("dec").agg(n=("apy", "size"), pred=("apy_pred", "mean"), actual=("apy", "mean"))
              .mul([1, 100, 100]).round(2).T.to_string())
        print(f"  years: exact {(P.yrs_pred == P.yrs).mean():.0%}, within 1 {(np.abs(P.yrs_pred - P.yrs) <= 1).mean():.0%}; "
              f"guaranteed share MAE {np.abs(P.gshare_pred - P.gshare).mean():.2f} (mean {P.gshare.mean():.2f})")
        big = P[P.apy_pred >= 0.04]
        print(f"  predicted >= 4% of cap (n {len(big)}): median |error| {np.median(np.abs(big.apy / big.apy_pred - 1)):.0%}, "
              f"actual / predicted {big.apy.sum() / big.apy_pred.sum():.2f}")
        print("  by group, actual / predicted:", P.groupby("grp").apply(lambda d: round(d.apy.sum() / d.apy_pred.sum(), 2),
                                                                      include_groups=False).to_dict())
        sub = P[(P.prev_rookie == 1) & (P.v1 >= 0.05)]
        print(f"  off rookie deals, value >= 5% (n {len(sub)}): actual / predicted {sub.apy.sum() / sub.apy_pred.sum():.2f}")
        P.to_parquet(fm.OUT / f"contract_predict_lag{lag}{'_all' if '--all' in sys.argv else ''}.parquet")


if __name__ == "__main__":
    main()
