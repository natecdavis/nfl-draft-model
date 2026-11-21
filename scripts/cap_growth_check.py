"""The flat contract model charges APY as the same share of the cap every season, but contract dollars are fixed while
the cap grows (~6% a year 2015-23, 7% projected in the tool). Does charging season t at apy / growth**t keep free-agent
verdicts calibrated? (Nov 21 2026, after the structure editor exposed it.)

Contracts and as-used forecasts from scripts/fa_realized_asused.py (UFA signed 2016-23, forecast a year ahead, skip 1).
Realized points two ways: "share" (as before: APY and dead money at the signing year's cap share every season) and
"dollars" (each season's pay and dead money divided by that season's real cap). Predicted with growth 1.00 (current) and
1.07 (the tool's projection), with the cut-rule theta refit to retention for each growth.
Pre-set: adopt growth 1.07 if, against dollar-realized points, its gap (realized - predicted) has a 95% interval
covering 0 and |gap| is no larger than growth 1.00's. Run: .venv/bin/python scripts/cap_growth_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src", "scripts"]
import veteran as vt  # noqa: E402

CAP = {2013: 122.3, 2014: 132.5, 2015: 142.9, 2016: 155.8, 2017: 165.0, 2018: 175.0, 2019: 185.0, 2020: 190.7,
       2021: 181.5, 2022: 207.4, 2023: 224.8, 2024: 250.0, 2025: 275.0, 2026: 299.2}  # median apy / apy_cap_pct


def realized(r, dollars):
    n = int(r.n)
    w = np.array([[r[f"w{k}"] for k in range(1, n + 1)]])
    sh = np.array([[r[f"s{k}"] if r[f"on{k}"] == 1 else 0.0 for k in range(1, n + 1)]])
    o = np.array([[r[f"on{k}"] == 1 and k <= r.kept for k in range(1, n + 1)]])
    c = np.where(o, r.apy, 0.0)
    if r.kept < n:
        c[0, int(r.kept)] = r.dead
    if dollars:
        c = c * np.array([CAP[r.Y] / CAP[r.Y + t] for t in range(n)])
    return vt.points(np.where(o, w, 0.0), sh, c, o, r.grp == "QB").sum()


def main():
    pd.set_option("display.width", 220)
    C = vt.comps()
    F = pd.read_parquet(vt.OUT / "fa_realized_asused.parquet")
    F["real_share"] = [realized(r, False) for _, r in F.iterrows()]
    F["real_dollars"] = [realized(r, True) for _, r in F.iterrows()]
    Aidx = C.A.set_index(["pfr_id", "Y"])
    paths = {}
    for i, r in F.iterrows():
        a = Aidx.loc[(r.pfr_id, r.Y - 1)]
        row = {"pfr_id": r.pfr_id, "Y": r.Y - 1, "grp": a.grp, "age": a.age, "v1": a.v1, "v2": a.v2, "v2f": a.v2f,
               "g1": a.g1, "p_share": a.p_share, "p_gp": a.p_gp}
        paths[i] = C.paths(row, int(r.n), exclude=True, skip=1)
    on_real = {k: F.loc[F.n >= k, f"on{k}"].mean() for k in range(2, 6)}
    print("still on the team, actual:", {k: round(v, 2) for k, v in on_real.items()})
    res = {}
    for g in (1.0, 1.07):
        best = None
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            on_m = {k: [] for k in range(2, 6)}
            for i, r in F.iterrows():
                _, _, on = vt.team_seasons(paths[i][0], int(r.n), r.apy, r.gtd, th, growth=g)
                for k in range(2, int(r.n) + 1):
                    on_m[k].append(on[:, k - 1].mean())
            err = sum((np.mean(on_m[k]) - on_real[k]) ** 2 * (F.n >= k).sum() for k in on_m if on_m[k])
            if best is None or err < best[1]:
                best = (th, err, {k: round(np.mean(v), 2) for k, v in on_m.items() if v})
        th = best[0]
        print(f"growth {g}: theta {th} (retention {best[2]})")
        pred = []
        for i, r in F.iterrows():
            V, S = paths[i]
            val, cost, on = vt.team_seasons(V, int(r.n), r.apy, r.gtd, th, growth=g)
            pred.append(vt.points(val, S, cost, on, r.grp == "QB").sum(axis=1).mean())
        res[g] = np.array(pred)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(F), len(F)) for _ in range(2000)]
    F["size"] = pd.cut(F.apy, [0, 0.02, 0.05, 1], labels=["1-2%", "2-5%", "5%+"])
    F["long"] = np.where(F.n >= 3, "3+ yrs", "1-2 yrs")
    rows = []
    for real in ["real_share", "real_dollars"]:
        for g, P in res.items():
            gap = F[real].to_numpy() - P
            bs = [gap[i].mean() for i in idx]
            row = {"realized": real, "growth": g, "pred": P.mean(), "real": F[real].mean(),
                   "gap (95%)": f"{gap.mean():+.3f} ({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f})"}
            for col in ["size", "long"]:
                for lab, m in F.groupby(col, observed=True).groups.items():
                    row[f"gap {lab}"] = gap[F.index.get_indexer(m)].mean()
            rows.append(row)
    print(f"\n{len(F)} contracts; points over the contract")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
