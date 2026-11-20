"""Does a situation- and season-dependent cap price change draft pick values? (follow-up to src/veteran.py)

Current (pick_value.pick_effect): each season's points = rate x value - 0.10 x (rookie cost - vet min), then titles via
the strength curve at that season's strength. So a cap dollar in season t costs 0.10 points x titles-per-point in t.
Shadow-price version: unused cap rolls over, and the next-best use of a cap dollar is market free agency (~0.12 points
per 1% of cap, scripts/vet_value.py UFA signings). So 1% of cap in season t costs 0.12 x max over s >= t (within the
window) of titles-per-point in s. Value side unchanged. Prints titles added (% of one Super Bowl) old vs new.

Window (argument, default 7): 5 = the model's seasons 1-5. 7 adds seasons 6-7 (and season 5 for picks after round 1,
which the model counts as 0) from what drafted players really gave the drafting team then (scripts/second_contract.py,
classes 2011-20: value while still with the team, minus the pay of his contract with them; 0 once gone), as an
expected amount by pick (kernel in log(pick + 10), QB and non-QB separately) added to every outcome path. The team's
baseline strength path and the title curve run for all 7 seasons (pick_value.baseline_paths extended).
Run: .venv/bin/python scripts/cap_shadow_picks.py [5|7]
"""
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

sys.path[:0] = ["src"]
import pick_value as pv  # noqa: E402
import veteran as vt  # noqa: E402
from outlook import OUT, RAW, star_thresholds  # noqa: E402
from value import VET_MIN, rookie_cost_curve  # noqa: E402

YIELD = 0.12
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 7
PICKS = [1, 5, 10, 20, 32, 48, 64, 100, 160, 224]
GRPS = ["QB", "WR", "ED", "T", "CB", "LB", "RB", "S"]


def baseline_paths(sit, rng, n=pv.N, T=7):
    """pick_value.baseline_paths for T seasons (same random draws for the first 5)."""
    m = sit["start"] * pv.decay(np.arange(T))
    for season, delta in sit["shift"].items():
        m[season - 1:] += delta
    u = np.zeros((n, T)); u[:, 0] = rng.normal(0, np.sqrt(1 - pv.PERM) * pv.START_SD, n)
    innov = np.sqrt(1 - pv.RHO ** 2) * np.sqrt(1 - pv.PERM) * 5.3
    for t in range(1, T):
        u[:, t] = pv.RHO * u[:, t - 1] + rng.normal(0, innov, n)
    return m + u


def late_seasons(rates):
    """Expected (value points, cap charged % of cap) per drafted player in seasons 5-7 for the drafting team, by pick."""
    S = pd.read_parquet(OUT / "second_contract_seasons.parquet")
    S = S[S.pick.notna()]
    S["qb"] = S.grp == "QB"
    rate = np.where(S.qb, rates["qb"], rates["nonqb"])
    S["vpts"] = np.where(S.stay, rate * np.maximum(S.value - VET_MIN, 0) * 100, 0.0)
    S["capc"] = np.where(S.stay, np.maximum(S.cost_apy.fillna(0) - VET_MIN, 0) * 100, 0.0)
    out = {}
    for qb in (True, False):
        for k in (5, 6, 7):
            d = S[(S.qb == qb) & (S.k == k)]
            lp = np.log(d.pick.to_numpy() + 10)
            for p in range(1, 263):
                w = np.exp(-0.5 * ((lp - np.log(p + 10)) / (0.35 if qb else 0.2)) ** 2)
                out[(qb, k, p)] = (np.average(d.vpts, weights=w), np.average(d.capc, weights=w))
    return out


def main():
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    cost = rookie_cost_curve(c)
    bar = star_thresholds(c).xs(2026, level="season").to_dict()
    rates = json.load(open(OUT / "bridge.json"))["preseason"]
    late = late_seasons(rates) if WINDOW == 7 else {}
    curve = vt._curve()
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)
    out = pd.read_parquet(OUT / "pick_outcomes.parquet")
    w = out[out.grp.isin(GRPS) & (out.grp != "QB")].grp.value_counts(normalize=True)
    rows, Ms = [], {}
    for sname, sit in pv.SITUATIONS.items():
        base = baseline_paths(sit, np.random.default_rng(11), T=WINDOW)
        m = (sb(base + 0.5) - sb(base - 0.5)).mean(axis=0)  # titles per point, by season
        M = np.maximum.accumulate(m[::-1])[::-1]  # best season from t on (rollover)
        Ms[sname] = (m, M)
        for grp in GRPS:
            for pick in PICKS:
                shares, price = pv.sample_outcomes(None, None, grp, pick, np.random.default_rng(pick))
                pv.COST_RATE = 0.10
                p5_old = pv.pick_effect(shares, price, pick, grp, sit, cost, bar, rates)
                if p5_old is None:
                    continue
                pv.COST_RATE = 0.0
                p5_val = pv.pick_effect(shares, price, pick, grp, sit, cost, bar, rates)
                n = len(p5_old)
                p_val, capc = np.zeros((n, WINDOW)), np.zeros((n, WINDOW))
                p_val[:, :5] = p5_val
                capc[:, :5] = (p5_val - p5_old) / 0.10  # % of cap charged, per season
                for k in range(5, WINDOW + 1) if late else ():
                    if k == 5 and pick <= 32:
                        continue  # round 1 season 5 = the option year, already in pick_effect
                    vp, cc = late[(grp == "QB", k, pick)]
                    p_val[:, k - 1] += vp
                    capc[:, k - 1] += cc
                p_old = p_val - 0.10 * capc
                old = (sb(base + p_old) - sb(base)).sum(axis=1).mean()
                val = (sb(base + p_val) - sb(base)).sum(axis=1).mean()
                new = val - (YIELD * capc.mean(axis=0) * M).sum()
                rows.append({"situation": sname, "grp": grp, "pick": pick, "old": old * 100, "new": new * 100})
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / f"cap_shadow_picks_w{WINDOW}.parquet")
    pd.set_option("display.width", 220)
    nq = (R[R.grp != "QB"].assign(w=lambda d: d.grp.map(w))
          .groupby(["situation", "pick"], sort=False)
          .apply(lambda d: pd.Series({k: np.average(d[k], weights=d.w) for k in ["old", "new"]}), include_groups=False))
    print(f"WINDOW {WINDOW} seasons. Typical non-QB pick, titles added (% of one SB): old charge vs shadow price")
    print(nq.unstack("pick").round(2).to_string())
    print("\nQB pick:")
    print(R[R.grp == "QB"].pivot_table(index="situation", columns="pick", values=["old", "new"]).round(2).to_string())
    print("\ntitles per point by season, and the rollover max:")
    for sname, (m, M) in Ms.items():
        print(f"  {sname:28s}", np.round(m * 100, 2), np.round(M * 100, 2))
    print("\ntrade-down examples, typical non-QB picks (old | new):")
    for sname in pv.SITUATIONS:
        t = nq.loc[sname]
        print(f"  {sname:28s} #5 vs #20+#48: {t.loc[5, 'old']:.2f} vs {t.loc[20, 'old'] + t.loc[48, 'old']:.2f} | "
              f"{t.loc[5, 'new']:.2f} vs {t.loc[20, 'new'] + t.loc[48, 'new']:.2f};  "
              f"#10 vs #32+#64: {t.loc[10, 'old']:.2f} vs {t.loc[32, 'old'] + t.loc[64, 'old']:.2f} | "
              f"{t.loc[10, 'new']:.2f} vs {t.loc[32, 'new'] + t.loc[64, 'new']:.2f}")


if __name__ == "__main__":
    main()
