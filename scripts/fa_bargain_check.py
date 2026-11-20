"""Do the free agents the ranking calls bargains deliver more value per dollar? (Free agency page, ranked list)

Past UFA contracts 2015-23 finished by 2025 (scripts/vet_value.py: forecast value with the team under the cut rule,
realized value with the team, actual cost = APY while there + dead money proxy). Bargain score = forecast value per
season / predicted market APY (scripts/contract_predict.py lag 1, leave-one-year-out, scaled to the mean). Pre-set
check: realized value per $1 of actual cost by bargain-score tercile rises; and realized points of the top tercile
beat the bottom. Also the market side: actual APY / predicted APY by tercile (do teams pay up for the bargains?).
Run: .venv/bin/python scripts/fa_bargain_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402

F = pd.read_parquet(vt.OUT / "vet_signings_ufa.parquet")
P = pd.read_parquet(vt.OUT / "contract_predict_lag1.parquet")[["pfr_id", "Y", "apy_pred"]]
P["apy_pred"] *= fm.SMEAR[1]  # contract_predict.py saves unscaled predictions unless run with --smeared (run it plain first)
F = F.merge(P, on=["pfr_id", "Y"], how="inner")
C = vt.comps()
fc = []
for r in F.itertuples():
    V, S = C.paths(r._asdict(), int(r.n))
    val, cost, on = vt.team_seasons(V, int(r.n), r.apy, r.gtd)
    fc.append(val.sum(axis=1).mean() / r.n)
F["fc_yr"] = fc
F["real_w"] = sum(np.where(F.n >= k, F[f"w{k}"].fillna(0), 0) for k in range(1, 6))
F["cost"] = F.apy * F.kept + F.dead
F["score"] = F.fc_yr / F.apy_pred
F["t"] = pd.qcut(F.score.rank(method="first"), 3, labels=["low", "mid", "high"])
rng = np.random.default_rng(0)
print(f"{len(F)} finished UFA contracts with an out-of-sample predicted price; mean actual/predicted APY "
      f"{F.apy.mean() / F.apy_pred.mean():.2f}")
T = F.groupby("t", observed=True).apply(lambda d: pd.Series({
    "n": len(d), "pred APY": d.apy_pred.mean() * 100, "actual APY": d.apy.mean() * 100,
    "actual/pred APY": d.apy.sum() / d.apy_pred.sum(), "forecast value/yr": d.fc_yr.mean() * 100,
    "realized value per $1 cost": d.real_w.sum() / d.cost.sum(), "realized pts": d.real_pts.mean(),
    "predicted pts": d.pred_pts.mean()}), include_groups=False)
print(T.round(3).to_string())
hi, lo = F[F.t == "high"], F[F.t == "low"]
d = []
for _ in range(2000):
    a, b = hi.sample(len(hi), replace=True, random_state=rng.integers(1e9)), lo.sample(len(lo), replace=True, random_state=rng.integers(1e9))
    d.append(a.real_w.sum() / a.cost.sum() - b.real_w.sum() / b.cost.sum())
print(f"value per $1, high - low tercile: {hi.real_w.sum() / hi.cost.sum() - lo.real_w.sum() / lo.cost.sum():+.3f} "
      f"(95% {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
G = F[F.cost > 0]
print(f"corr(score, realized value / cost) per contract (cost > 0, ratio capped at 5): "
      f"{np.corrcoef(G.score, (G.real_w / G.cost).clip(upper=5))[0, 1]:.2f}")
for lab, d in [("2015-18 signings", F[F.Y <= 2018]), ("2019-23 signings", F[F.Y >= 2019])]:
    t = d.groupby("t", observed=True).apply(lambda x: x.real_w.sum() / x.cost.sum(), include_groups=False)
    print(f"  {lab}: value per $1 by tercile low / mid / high {t.round(2).tolist()}")
