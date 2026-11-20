"""Late-round choice test: take an off-board player with a top-30 visit instead of the model's round 5-7 pick?

Lead: off-board undrafted players brought in for a top-30 private visit by a team other than the one that ended up
with them beat the public pre-draft model (2018-22 x2.4; fresh 2016-17 + 2023 x1.8; visits_udfa_test.py,
visits_udfa_replication.py). The model's candidates are the consensus board plus drafted players, so it never
considers them. Here: at each round 5-7 pick of 2016-2022 (outcomes complete), replace the model's
recommendation with an off-board player who was still available, and compare what each actually delivered
(review.RealizedV2: titles for the team on the clock, at that pick's cost; the review's own real_model column for
the model's pick, same yardstick).

Pool, by draft class: players not on the consensus board (prospect universe, prospect_model.load), still
available at the pick (undrafted, or drafted later). "Visited" = a reported top-30 private visit with some team
other than the one that drafted or signed him (that link can be reported after the draft). Position group: the
draft record if drafted, else his first undrafted OverTheCap contract, else the universe's coarse group.
Rules, specified before looking (each picks one player per pick; a player is counted once per draft, at the first
pick a rule takes him, as in the review's scorecard):
  V  visited: the visited player with the highest public pre-draft expectation (full_pre, fit leave-one-class-out
     on 2014-2022 classes, as in the replication)
  P  placebo: the same, among off-board players WITHOUT any reported visit (does the public model alone find them?)
  Vr visited, at random among those available (no public-model selection; mean of 20 draws)
Reported: realized new minus model's pick, paired, with bootstrap 95% CI; by round; 2016-17 (never in the lead's
discovery sample) vs 2018-22.
Run: python scripts/visits_late_choice.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import multisport_test as mt
import prospect_model as pm
import review as rv
from public_board import _norm
from trade import Context
from visits_test import visits
from visits_udfa_test import NICK

YEARS = tuple(range(2016, 2023))
COARSE = {"OL": "G", "DB": "CB", "DL": "IDL", "EDGE": "ED", "DE": "ED", "DT": "IDL", "OT": "T", "OLB": "ED"}
OTC = {"LT": "T", "RT": "T", "LG": "G", "RG": "G", "FB": "RB", "K": "SPEC", "P": "SPEC", "LS": "SPEC"}


def public(uni):
    f = pm.OUT / "visits_late_pub.parquet"
    if f.exists():
        return pd.read_parquet(f)
    saved = dict(pm.PARAMS); pm.PARAMS.update(mt.REG)
    out = []
    try:
        for c in YEARS:
            tr = uni[(uni.cls != c) & uni.cls.between(2014, 2022)]
            te = uni[uni.cls == c][["player_id"]].copy()
            te["exp_pub"] = 0.0
            for t in pm.SEASONS:
                m, feats = pm.fit(tr, "full_pre", t)
                te["exp_pub"] += m.predict(uni.loc[uni.cls == c, feats])
            out.append(te)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    F = pd.concat(out)
    F.to_parquet(f)
    return F


def pool(uni):
    u = uni[uni.cls.isin(YEARS) & uni.hasan_rank.isna()].copy()
    u = u.merge(public(uni), on="player_id", how="left")
    u["k"] = u.player_name.map(_norm)
    u = u[~u.duplicated(["k", "cls"], keep=False)]
    dp = pd.read_parquet("data/raw/draft_picks.parquet")
    dpos = {(y, p): rv.DRAFT_POS.get(pos, "SPEC") for y, p, pos in zip(dp.season, dp.pick, dp.position)}
    dteam = {(y, p): t for y, p, t in zip(dp.season, dp.pick, dp.team)}  # PFR codes, as in the visits data
    c = pd.read_parquet("data/raw/contracts.parquet")
    c = c[c.draft_round.isna() & (c.draft_year == c.year_signed) & c.year_signed.isin(YEARS)].copy()
    c["k"] = c.player.map(_norm)
    c = c.sort_values("guaranteed", ascending=False).drop_duplicates(["k", "year_signed"])
    cpos = {(y, k): OTC.get(p, p) for y, k, p in zip(c.year_signed, c.k, c.position)}
    cteam = {(y, k): NICK.get(str(t).split()[-1]) for y, k, t in zip(c.year_signed, c.k, c.team)}
    u["drafted_at"] = u.overall_pick.where(u.drafted.astype(bool))
    u["grp"] = [dpos.get((y, int(a))) if pd.notna(a) else cpos.get((y, k), COARSE.get(g, g))
                for y, a, k, g in zip(u.cls, u.drafted_at, u.k, u.position_group.astype(str))]
    u["ended_with"] = [dteam.get((y, int(a))) if pd.notna(a) else cteam.get((y, k)) for y, a, k in zip(u.cls, u.drafted_at, u.k)]
    vk = visits(uni)
    vk["player_id"] = vk.player_id.astype(str)
    u["player_id"] = u.player_id.astype(str)
    pri = vk[vk.PRI > 0].groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    met = vk.groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    u["visited"] = [len(pri.get((p, y), set()) - {e}) > 0 for p, y, e in zip(u.player_id, u.cls, u.ended_with)]
    u["any_meeting"] = [len(met.get((p, y), set())) > 0 for p, y in zip(u.player_id, u.cls)]
    u = u[u.grp.isin(rv.GROUPS) & u.exp_pub.notna()]
    return u


def boot(x, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return f"n={len(x)}"
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.3f} (95% {np.percentile(b, 2.5):+.3f} to {np.percentile(b, 97.5):+.3f}) n={len(x)}"


def main():
    uni = pm.load()
    P = pool(uni)
    R = pd.read_parquet(pm.OUT / "review_picks.parquet")
    R = R[R.year.isin(YEARS) & (R["round"] >= 5) & R.real_model.notna()].copy()
    S = pd.read_parquet(pm.OUT / "review_states.parquet")
    sits = {(r.year, r.team): rv._sit(r) for r in S.itertuples()}
    pneed = {(r.year, r.team): r.p_need for r in S.itertuples()}
    RV = rv.RealizedV2(Context())
    print(f"off-board pool 2016-22: {len(P)} players; visited (top-30, not the team that got him) {int(P.visited.sum())} "
          f"({P.groupby('cls').visited.sum().astype(int).to_dict()}); round 5-7 picks scored {len(R)}")

    cache = {}

    def real(r, c):
        key = (r.year, r.team, r.pick, c.player_id)
        if key not in cache:
            cache[key] = RV.titles_rec(sits[(r.year, r.team)], pneed[(r.year, r.team)], c.grp, int(r.year),
                                       c.drafted_at, int(r.pick), c.player_name)
        return cache[key]

    rng = np.random.default_rng(1)
    rows = []
    for r in R.itertuples():
        a = P[(P.cls == r.year) & (P.drafted_at.isna() | (P.drafted_at > r.pick))]
        v, nv = a[a.visited], a[~a.any_meeting]
        out = {"year": r.year, "pick": r.pick, "round": r.round, "team": r.team,
               "old": r.real_model, "n_visited": len(v)}
        for lab, g in [("V", v), ("P", nv)]:
            if len(g):
                c = g.loc[g.exp_pub.idxmax()]
                out[f"{lab}_player"], out[lab] = c.player_name, real(r, c)
        if len(v):
            draws = [real(r, v.iloc[i]) for i in rng.integers(0, len(v), min(20, len(v)))]
            out["Vr"] = float(np.mean(draws))
        rows.append(out)
    D = pd.DataFrame(rows)
    D.to_parquet(pm.OUT / "visits_late_choice.parquet")
    print(f"picks with a visited player available: {(D.n_visited > 0).mean():.0%} (median {D.n_visited.median():.0f} available)")
    print("\nrealized titles (% of one), new minus the model's pick; each rule's player counted once per draft (first pick)")
    for lab, desc in [("V", "V  visited, best public model"), ("P", "P  placebo: unvisited, best public model"),
                      ("Vr", "Vr visited, random")]:
        d = D.dropna(subset=[lab])
        if lab != "Vr":
            d = d.sort_values("pick").drop_duplicates(["year", f"{lab}_player"])
        d = d.assign(gain=d[lab] - d.old)
        print(f"  {desc:42s} all {boot(d.gain)}")
        for rd in (5, 6, 7):
            print(f"      round {rd}: {boot(d.gain[d['round'] == rd])}")
        print(f"      2016-17: {boot(d.gain[d.year <= 2017])} | 2018-22: {boot(d.gain[d.year >= 2018])}")
        print(f"      mean realized: rule {d[lab].mean():.3f} vs model's pick {d.old.mean():.3f}")
    v = D.dropna(subset=["V"]).sort_values("pick").drop_duplicates(["year", "V_player"])
    top = v.sort_values("V", ascending=False).head(6)
    print("\nlargest V outcomes: " + ", ".join(f"{n} ({y} #{p}) {x:.2f}" for n, y, p, x in zip(top.V_player, top.year, top.pick, top.V)))


if __name__ == "__main__":
    main()
