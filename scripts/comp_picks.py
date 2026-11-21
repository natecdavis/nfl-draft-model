"""Compensatory picks: can the award formula be reproduced from contract data? (letting free agents walk)

Actual comp picks (data/derived/comp_picks_actual.parquet): picks after #32 within rounds 3-7, drafts 2015-2026, traced
back through the trade log to the team that earned them (119 of 393 were traded). Since 2021 the league also gives
round-3 picks for developing minority head coaches / GMs; they look the same and are left in (noise in round 3).
Free-agent moves (data/derived/cfa_moves.parquet): unrestricted free agents whose previous contract expired (released
players don't count) signing with a new team, league years 2014-2025, valued by the new contract's APY share of cap.

Formula (the league's isn't public; the shape is): each lost CFA is valued into round 3-7 by APY (cut-offs fit here)
or doesn't qualify; each CFA gained (most valuable first) cancels the most valuable lost CFA at or below his value, or
the least valuable lost CFA if all are worth more; remaining lost CFAs earn picks of their round, at most 4 per team.
Value = APY x (0.5 + 0.5 x snap share with the new team that season) (ALPHA; APY alone matched 56% vs 60%).
Fit: cut-offs by coordinate search maximizing matched picks (team, year, round); checked leaving each FA year out.
Writes data/derived/comp_pick_cuts.json (cuts, alpha), read by offseason.comp_estimate.
Run: .venv/bin/python scripts/comp_picks.py [--rebuild]   (rebuilds the pick and move tables)
Pre-set: build into the tool's totals if >= 70% of actual picks are matched by team and round out of sample;
otherwise show as an estimate only.
"""
import itertools
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path[:0] = ["src", "scripts"]
from outlook import OUT  # noqa: E402

GRID = np.round(np.arange(0.002, 0.10, 0.001), 4)


def value_round(apy, cuts):
    """cuts = (t3, t4, t5, t6, t7) descending APY shares; round of a CFA or None if below t7."""
    for rnd, t in zip((3, 4, 5, 6, 7), cuts):
        if apy >= t:
            return rnd
    return None


def predict(lost, gained, cuts, cap4=True):
    L = sorted([r for r in (value_round(a, cuts) for a in lost) if r is not None])  # 3 = most valuable
    G = sorted([r for r in (value_round(a, cuts) for a in gained) if r is not None])
    if len(G) >= len(L):
        return []
    L = list(L)
    for g in G:
        below = [x for x in L if x >= g]
        if below:
            L.remove(min(below))  # most valuable lost CFA at or below him
        else:
            L.remove(max(L))      # all worth more: the least valuable
    L.sort()
    return L[:4] if cap4 else L


def team_years(M):
    out = {}
    for (y, t), d in M.groupby(["Y", "from"]):
        out.setdefault((y, t), [[], []])[0] = list(d.apy)
    for (y, t), d in M.groupby(["Y", "to"]):
        out.setdefault((y, t), [[], []])[1] = list(d.apy)
    return out


def score(TY, A, cuts, years):
    """matched picks (team, year, round) and actual total over the given FA years."""
    match = tot = pred_n = 0
    cnt_ok = 0
    teams = 0
    for (y, t), (lost, gained) in TY.items():
        if y not in years:
            continue
        p = Counter(predict(lost, gained, cuts))
        a = Counter(A.get((y + 1, t), []))
        match += sum((p & a).values())
        tot += sum(a.values())
        pred_n += sum(p.values())
        cnt_ok += sum(p.values()) == sum(a.values())
        teams += 1
    return match, tot, pred_n, cnt_ok, teams


def fit(TY, A, years, start=(0.06, 0.035, 0.02, 0.012, 0.006)):
    cuts = list(start)
    best = score(TY, A, cuts, years)[0]
    for _ in range(6):
        improved = False
        for i in range(5):
            for v in GRID:
                trial = cuts.copy()
                trial[i] = v
                if any(trial[j] < trial[j + 1] for j in range(4)):
                    continue
                s = score(TY, A, trial, years)[0]
                if s > best:
                    best, cuts, improved = s, trial, True
        if not improved:
            break
    return tuple(cuts), best


ALPHA = 0.5  # value = APY x (ALPHA + (1 - ALPHA) x snap share with the new team); 1.0 = APY only (56% vs 60%)


def build_actual():
    """Comp picks 2015-2026 (after #32 within rounds 3-7), traced to the team that earned them via the trade log."""
    import roster_need as rn
    d = pd.read_parquet(OUT.parent / "raw" / "draft_picks.parquet")
    d = d[d.season.between(2015, 2026)].copy()
    d["inround"] = d.pick - d.groupby(["season", "round"]).pick.transform("min") + 1
    C = d[(d["round"] >= 3) & (d.inround > 32)][["season", "round", "pick", "team"]].copy()
    C["team"] = C.team.map(rn.team)
    t = pd.read_parquet(OUT.parent / "raw" / "trades.parquet")
    t = t[t.pick_number.notna()].copy()
    t["gave"] = t.gave.map(rn.team)
    t = t.sort_values(["trade_date", "trade_id"])
    C["orig"] = [(lambda tt: tt.iloc[0].gave if len(tt) else tm)(t[(t.pick_season == y) & (t.pick_number == pk)])
                 for y, pk, tm in zip(C.season, C.pick, C.team)]
    C.to_parquet(OUT / "comp_picks_actual.parquet")
    return C


def build_moves():
    """UFA signings with a new team whose previous contract expired, league years 2014-2025, with APY share of cap and
    the player's snap share (x games) that season."""
    import roster_need as rn
    import veteran as vt
    c = pd.read_parquet(OUT.parent / "raw" / "contracts.parquet")
    cap = (c.apy / c.apy_cap_pct).replace([np.inf], np.nan).groupby(c.year_signed).median()
    c = c.assign(n=c.contract_history.map(lambda x: len(x) if x is not None else 0)).sort_values("n").drop_duplicates("gsis_id", keep="last")
    nick = dict(zip(*pd.read_parquet(OUT.parent / "raw" / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS", "Oilers": "TEN"})
    T = lambda x: rn.team(nick.get(x, x)) if x else None  # noqa: E731
    rows = []
    for g, nm, ch in zip(c.gsis_id, c.player, c.contract_history):
        h = [x for x in (ch if ch is not None else []) if x.get("contract_type") != "Practice" and x.get("year_signed") is not None]
        for i, x in enumerate(h):
            if x.get("contract_type") != "UFA" or i == 0:
                continue
            y = int(x["year_signed"])
            if y < 2014 or y > 2025 or y not in cap.index or x.get("apy") is None:
                continue
            prev = h[i - 1]
            if prev.get("status") != "Expired":
                continue
            a, b = T(prev.get("team")), T(x.get("team"))
            if not a or not b or a == b or "/" in str(prev.get("team")):
                continue
            rows.append((nm, g, y, a, b, x["apy"] / cap[y], int(x.get("yrs") or 1)))
    M = pd.DataFrame(rows, columns=["player", "gsis", "Y", "from", "to", "apy", "yrs"])
    pl = pd.read_parquet(OUT.parent / "raw" / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    V = vt.season_table().set_index(["pfr_id", "season"])
    sh = (V.share * V.gfrac).to_dict()
    M["share"] = [sh.get((g2p.get(g), y), 0.0) for g, y in zip(M.gsis, M.Y)]
    M.to_parquet(OUT / "cfa_moves.parquet")
    return M


def main():
    import json
    rebuild = "--rebuild" in sys.argv
    C = build_actual() if rebuild or not (OUT / "comp_picks_actual.parquet").exists() else pd.read_parquet(OUT / "comp_picks_actual.parquet")
    M = build_moves() if rebuild or not (OUT / "cfa_moves.parquet").exists() else pd.read_parquet(OUT / "cfa_moves.parquet")
    if "share" not in M:
        M = build_moves()
    A = C.groupby(["season", "orig"])["round"].apply(list).to_dict()
    print(f"comp picks {len(C)} ({C.groupby('season').size().to_dict()}); free-agent moves {len(M)}")
    M = M.assign(apy=M.apy * (ALPHA + (1 - ALPHA) * M.share))
    TY = team_years(M)
    years = sorted(M.Y.unique())
    cuts, best = fit(TY, A, years)
    m, tot, pn, ok, n = score(TY, A, cuts, years)
    print(f"in-sample cut-offs (APY % of cap) rounds 3-7: {[round(c * 100, 2) for c in cuts]}")
    print(f"  matched {m} of {tot} actual picks by team + round ({m / tot:.0%}); predicted {pn}; team-years with the "
          f"right count {ok / n:.0%}")
    rows = []
    for y in years:
        cy, _ = fit(TY, A, [x for x in years if x != y], start=cuts)
        m, tot, pn, ok, n = score(TY, A, cy, [y])
        rows.append({"FA year": y, "draft": y + 1, "actual": tot, "predicted": pn, "matched": m, "right count": ok, "teams": n})
    R = pd.DataFrame(rows)
    print("\nleave-one-year-out (cut-offs fit without that year):")
    print(R.to_string(index=False))
    print(f"  total matched {R.matched.sum()} of {R.actual.sum()} ({R.matched.sum() / R.actual.sum():.0%}); "
          f"predicted {R.predicted.sum()}; right count for {R['right count'].sum() / R.teams.sum():.0%} of team-years")
    # by round, out of sample is close to in-sample; show in-sample confusion by round
    P = Counter(); Aa = Counter(); Mm = Counter()
    for (y, t), (lost, gained) in TY.items():
        p = Counter(predict(lost, gained, cuts)); a = Counter(A.get((y + 1, t), []))
        P.update(p); Aa.update(a); Mm.update(p & a)
    print("\nby round (in-sample): actual / predicted / matched",
          {r: (Aa[r], P[r], Mm[r]) for r in (3, 4, 5, 6, 7)})
    # within one round, and comp value per team-year (rough round weights), leaving each year out
    val = {3: 4, 4: 3, 5: 2, 6: 1.5, 7: 1}
    w1 = tot = 0
    pv, av = [], []
    for y in years:
        cy, _ = fit(TY, A, [x for x in years if x != y], start=cuts)
        for (yy, t), (lost, gained) in TY.items():
            if yy != y:
                continue
            p, a = sorted(predict(lost, gained, cy)), sorted(A.get((y + 1, t), []))
            tot += len(a)
            pp = list(p)
            for r in a:
                near = [q for q in pp if abs(q - r) <= 1]
                if near:
                    pp.remove(min(near, key=lambda q: abs(q - r)))
                    w1 += 1
            pv.append(sum(val[r] for r in p))
            av.append(sum(val[r] for r in a))
    print(f"\nleave-one-year-out: {w1} of {tot} actual picks within one round ({w1 / tot:.0%}); team-year comp value corr "
          f"{np.corrcoef(pv, av)[0, 1]:.2f}, mean predicted {np.mean(pv):.2f} vs actual {np.mean(av):.2f}")
    json.dump({"cuts": [float(c) for c in cuts], "alpha": ALPHA}, open(OUT / "comp_pick_cuts.json", "w"))


if __name__ == "__main__":
    main()
