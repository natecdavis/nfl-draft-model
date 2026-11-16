"""Rescore past choice tests on a second-contract pay yardstick with the counting-stat premium removed.

Follow-up to scripts/stat_premium.py: teams pay for counting stats beyond graded quality, and those dollars buy ~0
team strength. The "2nd-contract pay" yardstick (pick_outcomes.best_pay = best post-rookie APY, % of cap) carries
that premium. Two adjusted yardsticks, per drafted player's best second contract:
  pay_adj   = APY - stat part (all stat terms of the pay model at signing)
  pay_noise = APY - noise part (only stats that don't predict next season's quality: sacks, INT, passes defended, TDs)
Contracts outside the decomposition (no prior season of 30%+ snaps or no quality) keep their APY. Floor 0.
Each saved choice-test frame is rescored by (draft year, pick the player went at); paired differences per team-draft
with shared scoring (a player taken on n teams' paths counts 1/n), as in plan_choice.report. Raw pay is recomputed
from the same contracts and checked against the stored column.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stat_premium as sp  # noqa: E402

OUT, RAW = sp.OUT, sp.RAW
PAIRS = {
    "plan_choice": [("B", "A"), ("C", "B"), ("D", "B"), ("E", "A")],
    "paypos_choice": [("P", "B"), ("Q", "B")],
    "market_choice": [("M", "B")],
    "market_choice_s0.5": [("M", "B")],
    "repl_pooled_choice": [("M", "B")],
    "tilt_choice": [("T", "B"), ("T/2", "B")],
    "posproj_choice": [("L1", "L0"), ("cur", "L0")],
    "stack_choice": [("W", "B"), ("WO", "B")],
    "lookahead_recheck": [("C", "B"), ("D", "B")],
    "qb_sit_choice": [("fix", "base")],
}


def adjusted_pay():
    C, pos, q, share, st = sp.load()
    c = sp.contracts(pos, q, share, st)
    _, parts = sp.pay_models(c)
    k = parts.set_index(["gsis_id", "year_signed"])
    con = pd.read_parquet(RAW / "contracts.parquet")
    con = con[con.apy_cap_pct.notna() & con.draft_overall.notna() & (con.year_signed > con.draft_year)].copy()
    con["stat"] = [k.stat_part.get((g, y), 0.0) if pd.notna(g) else 0.0 for g, y in zip(con.gsis_id, con.year_signed)]
    con["noise"] = [k.noise_part.get((g, y), 0.0) if pd.notna(g) else 0.0 for g, y in zip(con.gsis_id, con.year_signed)]
    con["covered"] = [(g, y) in k.index for g, y in zip(con.gsis_id, con.year_signed)]
    best = con.sort_values("apy_cap_pct").groupby(["draft_year", "draft_overall"]).last().reset_index()
    best = best[best.draft_year <= 2021]  # same classes as pick_outcomes.best_pay (2022 picks score 0 there too)
    best["pay"] = best.apy_cap_pct * 100
    best["pay_adj"] = (best.pay - best.stat).clip(lower=0)
    best["pay_noise"] = (best.pay - best.noise).clip(lower=0)
    print(f"best second contracts: {len(best)}, covered by the decomposition {best.covered.mean():.0%} "
          f"(APY-weighted {np.average(best.covered, weights=best.pay):.0%})")
    pos_ = pd.read_parquet(RAW / "contracts.parquet")[["gsis_id", "position"]].drop_duplicates("gsis_id")
    best = best.drop(columns="position", errors="ignore").merge(pos_, on="gsis_id", how="left")
    g = best.groupby(best.position.map(lambda x: {"OLB": "ED", "ED": "ED"}.get(x, x)))
    print("mean adjustment (pay - pay_adj) by position, % of cap, best contracts >= 3% of cap:")
    b3 = best[best.pay >= 3]
    print((b3.groupby("position").apply(lambda d: pd.Series({"n": len(d), "stat": d.stat.mean(), "noise": d.noise.mean()})))
          .round(2).to_string())
    return {(int(y), int(p)): (r1, r2, r3) for y, p, r1, r2, r3 in
            zip(best.draft_year, best.draft_overall, best.pay, best.pay_adj, best.pay_noise)}


def attach(P, M):
    get = lambda y, a, i: 0.0 if pd.isna(a) else M.get((int(y), int(a)), (0.0, 0.0, 0.0))[i]
    P["pay_raw"] = [get(y, a, 0) for y, a in zip(P.year, P.drafted_at)]
    P["pay_adj"] = [get(y, a, 1) for y, a in zip(P.year, P.drafted_at)]
    P["pay_noise"] = [get(y, a, 2) for y, a in zip(P.year, P.drafted_at)]
    return P


def boot(x, rng):
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    M = adjusted_pay()
    rnd = pd.read_parquet(OUT / "units_choice.parquet")
    rmap = dict(zip(zip(rnd.year, rnd.pick), rnd["round"]))
    rows = []
    for f, pairs in PAIRS.items():
        P = attach(pd.read_parquet(OUT / f"{f}.parquet"), M)
        chk = np.corrcoef(P.pay.fillna(0), P.pay_raw)[0, 1]
        P["round"] = [rmap.get((y, k)) for y, k in zip(P.year, P.pick)]
        P["n_paths"] = P.groupby(["version", "year", "player"]).team.transform("nunique").clip(lower=1)
        for a, b in pairs:
            for rl, sel in [("all", lambda r: r > 0), ("R1", lambda r: r == 1), ("R2-4", lambda r: (r >= 2) & (r <= 4))]:
                rng = np.random.default_rng(0)
                out = dict(test=f, pair=f"{a} - {b}", rounds=rl, raw_check=round(chk, 3))
                for m in ["t_single", "pay", "pay_raw", "pay_adj", "pay_noise"]:
                    if m not in P:
                        out[m] = ""
                        continue
                    x = {}
                    for v in (a, b):
                        Q = P[(P.version == v) & sel(P["round"].fillna(0))]
                        x[v] = (Q[m].fillna(0) / Q.n_paths).groupby([Q.year, Q.team]).sum()
                    idx = x[a].index.union(x[b].index)
                    d = x[a].reindex(idx, fill_value=0) - x[b].reindex(idx, fill_value=0)
                    mu, lo, hi = boot(d, rng)
                    out[m] = f"{mu:+.2f} ({lo:+.2f},{hi:+.2f})"
                rows.append(out)
    # units_choice: single vs family units, and model (single) vs team
    U = rnd.copy()
    for who in ["team", "family", "single"]:
        U = U.rename(columns={f"pay_{who}": f"pay_{who}"})
        tmp = attach(pd.DataFrame({"year": U.year, "drafted_at": U[f"{who}_at"]}), M)
        for m in ["pay_raw", "pay_adj", "pay_noise"]:
            U[f"{m}_{who}"] = tmp[m].values
    for a, b in [("single", "family"), ("single", "team")]:
        for rl, sel in [("all", U["round"] > 0), ("R1", U["round"] == 1), ("R2-4", U["round"].between(2, 4))]:
            rng = np.random.default_rng(0)
            out = dict(test="units_choice", pair=f"{a} - {b}", rounds=rl,
                       raw_check=round(np.corrcoef(U.pay_team.fillna(0), U.pay_raw_team)[0, 1], 3))
            for m, (ca, cb) in {"t_single": (f"t_single_{a}", f"t_single_{b}"), "pay": (f"pay_{a}", f"pay_{b}"),
                                "pay_raw": (f"pay_raw_{a}", f"pay_raw_{b}"), "pay_adj": (f"pay_adj_{a}", f"pay_adj_{b}"),
                                "pay_noise": (f"pay_noise_{a}", f"pay_noise_{b}")}.items():
                Q = U[sel]
                d = (Q[ca].fillna(0) - Q[cb].fillna(0)).groupby([Q.year, Q.team]).sum()
                mu, lo, hi = boot(d, rng)
                out[m] = f"{mu:+.2f} ({lo:+.2f},{hi:+.2f})"
            rows.append(out)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 260, "display.max_colwidth", 30)
    print("\nPer team-draft, version a minus b (shared scoring): titles (single units), stored pay, recomputed pay, "
          "stat-adjusted pay, noise-only-adjusted pay")
    print(T.to_string(index=False))
    T.to_csv(OUT / "pay_rescore.csv", index=False)


if __name__ == "__main__":
    main()
