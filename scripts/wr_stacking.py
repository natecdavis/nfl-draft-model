"""WR stacking: is a second / third good young WR on one team worth as much as the model credits?

The model values every pick on its own: a WR's value is his snap share x the pay-curve price of his quality,
and the only stacking adjustment is within one draft (2nd rookie at a group: years 1-2 playing time x0.6,
3rd x0.3; draftday.NEED_EARLY). It never asks what a new WR does to the team's other WRs. Targets are
shared, so a new WR could lower his teammates' value (cannibalization), which the model doesn't charge.

Four tests, every position group run the same way so WR can be compared with the rest:
 A. Incumbents: players who played for the team last season and didn't leave before the draft (signed
    elsewhere / traded away; roster_need.departures_and_adds). Their summed value this season (0 if they
    didn't play for the team) on whether the team drafted a player at their group in rounds 1-2 (and
    rounds 1-3 over the last 2 drafts), controlling for their summed value / share last season, count,
    group x season. Placebo: a round 1-2 pick at the group in NEXT year's draft (can't cause this season;
    picks up teams drafting because incumbents disappointed).
 B. Rookies: a round 1-3 rookie's own value in years 1-2 vs the incumbents' value at his group last
    season, controlling for group x log pick.
 C. Team points: offensive rating on WR values sorted (WR1, WR2, WR3, rest) plus the other units' sums,
    leak-free prior-season quality and same-season quality. Diminishing returns = WR2/WR3 buy less per 1%.
 D. Market: betting-line absence test (carry), WR absence split by how good the WRs still playing are.

Values: cfdarko veteran pay curve price_g(share, q) / 1.10, % of cap, same as scripts/replacement_units.py.
Usage: python scripts/wr_stacking.py [rebuild]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/derived"
PS_FILE = OUT / "wr_stacking_ps.parquet"
GROUPS = ["RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
pd.set_option("display.width", 220)


def player_seasons():
    if PS_FILE.exists() and "rebuild" not in sys.argv:  # cached pricing of every player-season
        return pd.read_parquet(PS_FILE)
    import replacement_units as ru
    C = ru.capture()
    ps = ru.player_seasons(C)
    for c in ["base", "v_now", "v_prior"]:
        ps[c] = ps[c] * 100 / ru.UNITS
    ps = ps.drop(columns=["gsis"]).rename(columns={"pfr_player_id": "pfr"})
    ps.to_parquet(PS_FILE)
    return ps


def draft_table(ps):
    dp = pd.read_parquet(ROOT / "data/raw/draft_picks.parquet")
    dp = dp[dp.season.between(2011, 2026)].copy()
    from roster_need import team
    dp["team"] = dp.team.map(team).replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
    # group: pricing group from player-seasons where known, else draft position
    g_of = ps.groupby("pfr").g.agg(lambda x: x.mode().iloc[0])
    DP = {"WR": "WR", "RB": "RB", "FB": "RB", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
          "DT": "DL", "NT": "DL", "DL": "DL", "DE": "EDGE", "OLB": "EDGE", "LB": "LB", "ILB": "LB",
          "CB": "DB", "S": "DB", "DB": "DB", "SAF": "DB", "QB": "QB"}
    dp["g"] = [g_of.get(p) if p in g_of.index else DP.get(pos) for p, pos in zip(dp.pfr_player_id, dp.position)]
    return dp


def test_a(ps, dp):
    from roster_need import departures_and_adds
    v = ps.groupby(["pfr", "season", "team", "g"])[["v_now", "share"]].sum().reset_index()
    rows = []
    for y in range(2014, 2026):
        signed, away, _ = departures_and_adds(y)
        prev = v[v.season == y - 1]
        left = np.array([(p, tm) in away or (p in signed and tm not in signed[p])
                         for p, tm in zip(prev.pfr, prev.team)], bool)
        inc = prev[~left].copy()
        rookies_now = set(dp[dp.season == y].pfr_player_id)
        inc = inc[~inc.pfr.isin(rookies_now)]
        now = v[v.season == y].set_index(["pfr", "team"]).v_now
        inc["v_t"] = [float(now.get((p, tm), 0.0)) for p, tm in zip(inc.pfr, inc.team)]
        inc["season"] = y
        rows.append(inc)
    I = pd.concat(rows)
    I = I[I.g.isin(GROUPS)]
    T = (I.groupby(["season", "team", "g"])
         .agg(v_t=("v_t", "sum"), v_prev=("v_now", "sum"), sh_prev=("share", "sum"), n=("pfr", "size"),
              v_top=("v_now", "max")).reset_index())
    d = dp[dp.g.isin(GROUPS)]

    def flag(rounds, years_back, lead=0):
        k = d[d["round"].isin(rounds)]
        out = set()
        for s, tm, g in zip(k.season, k.team, k.g):
            for b in range(years_back):
                out.add((s + b - lead, tm, g))
        return np.array([(s, tm, g) in out for s, tm, g in zip(T.season, T.team, T.g)], float)

    T["r12_now"] = flag([1, 2], 1)
    T["r1_now"] = flag([1], 1)
    T["r13_last2"] = flag([1, 2, 3], 2)  # this draft or last draft
    T["r12_next"] = flag([1, 2], 1, lead=1)  # next year's draft (placebo)
    T["cl"] = T.team
    res = []
    for g in GROUPS:
        D = T[T.g == g]
        for x in ["r12_now", "r1_now", "r12_next"]:
            f = f"v_t ~ {x} + v_prev + sh_prev + n + v_top + C(season)"
            m = smf.ols(f, D).fit(cov_type="cluster", cov_kwds={"groups": D.cl})
            lo, hi = m.conf_int().loc[x]
            res.append({"group": g, "treatment": x, "n_treated": int(D[x].sum()), "effect": m.params[x],
                        "lo": lo, "hi": hi, "p": m.pvalues[x], "mean v_t": D.v_t.mean()})
    R = pd.DataFrame(res)
    print("\nA. Incumbents' summed value this season (% of cap) when the team drafts at their group")
    print("   (controls: incumbents' value, share, count, best player last season; season FE; cluster team)")
    print(R.round(3).to_string(index=False))
    # dose: WR by incumbents' value tercile, and with the rookie's own value for scale
    W = T[T.g == "WR"].copy()
    W["terc"] = pd.qcut(W.v_prev, 3, labels=["low", "mid", "high"])
    out = []
    for t, D in W.groupby("terc", observed=True):
        m = smf.ols("v_t ~ r12_now + v_prev + sh_prev + n + v_top + C(season)", D).fit(
            cov_type="cluster", cov_kwds={"groups": D.cl})
        lo, hi = m.conf_int().loc["r12_now"]
        out.append({"incumbent WR value": t, "n": len(D), "treated": int(D.r12_now.sum()),
                    "effect": m.params["r12_now"], "lo": lo, "hi": hi})
    print("\n   WR, by incumbents' value last season (tercile):")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    # year 2 (rookie's second season): incumbents present in y-1 and y
    T.to_parquet(OUT / "wr_stacking_a.parquet")
    return T


def test_b(ps, dp, T):
    v = ps.groupby(["pfr", "season", "team"]).v_now.sum()
    k = dp[dp["round"].between(1, 3) & dp.g.isin(GROUPS) & dp.season.between(2014, 2024)].copy()
    k["v1"] = [float(v.get((p, y, tm), 0.0)) for p, y, tm in zip(k.pfr_player_id, k.season, k.team)]
    k["v2"] = [float(v.get((p, y + 1, tm), 0.0)) for p, y, tm in zip(k.pfr_player_id, k.season, k.team)]
    k["v12"] = (k.v1 + k.v2) / 2
    inc = T.set_index(["season", "team", "g"])
    k["inc_v"] = [float(inc.v_prev.get((y, tm, g), 0.0)) for y, tm, g in zip(k.season, k.team, k.g)]
    k["inc_top"] = [float(inc.v_top.get((y, tm, g), 0.0)) for y, tm, g in zip(k.season, k.team, k.g)]
    # other same-group picks rounds 1-3 this draft or last draft (young teammates)
    kk = dp[dp["round"].between(1, 3)]
    cnt = kk.groupby(["season", "team", "g"]).size()
    k["young"] = [cnt.get((y, tm, g), 0) - 1 + cnt.get((y - 1, tm, g), 0) for y, tm, g in zip(k.season, k.team, k.g)]
    k["lp"] = np.log(k.pick + 10)
    k["cl"] = k.season.astype(str) + k.team
    res = []
    for g in GROUPS:
        D = k[k.g == g]
        m = smf.ols("v12 ~ inc_v + young + lp + C(season)", D).fit(cov_type="cluster", cov_kwds={"groups": D.cl})
        for x in ["inc_v", "young"]:
            lo, hi = m.conf_int().loc[x]
            res.append({"group": g, "n": len(D), "x": x, "effect": m.params[x], "lo": lo, "hi": hi,
                        "p": m.pvalues[x], "mean v12": D.v12.mean(), "sd x": D[x].std()})
    print("\nB. Round 1-3 rookie's own value, years 1-2 avg (% of cap), vs incumbents' value at his group last "
          "season (per 1% of cap) and other round 1-3 picks at his group this/last draft (per player)")
    print(pd.DataFrame(res).round(3).to_string(index=False))


def test_c(ps):
    from synergy_test import ratings
    R = ratings()
    out = {}
    for which in ["prior", "now"]:
        col = f"v_{which}"
        rows = []
        for (y, tm), d in ps.groupby(["season", "team"]):
            r = {"season": y, "team": tm}
            for g in ["QB", "RB", "TE", "OL"]:
                r[g] = d[d.g == g][col].sum()
            w = np.sort(d[d.g == "WR"][col].to_numpy())[::-1]
            w = np.r_[w, np.zeros(4)]
            r.update(WR1=w[0], WR2=w[1], WR3=w[2], WRrest=w[3:].sum())
            o = np.sort(d[d.g == "OL"][col].to_numpy())[::-1]
            o = np.r_[o, np.zeros(6)]
            r.update(OLtop2=o[:2].sum(), OL345=o[2:5].sum(), OLrest=o[5:].sum())
            rows.append(r)
        W = pd.DataFrame(rows).merge(R, on=["season", "team"])
        cols = [c for c in W.columns if c not in ("season", "team")]
        W[cols] = W[cols] - W.groupby("season")[cols].transform("mean")
        m = smf.ols("off ~ QB + RB + TE + OL + WR1 + WR2 + WR3 + WRrest", W).fit(
            cov_type="cluster", cov_kwds={"groups": W.team})
        m2 = smf.ols("off ~ QB + RB + TE + OLtop2 + OL345 + OLrest + WR1 + WR2 + WR3 + WRrest", W).fit(
            cov_type="cluster", cov_kwds={"groups": W.team})
        ci = m.conf_int()
        t = pd.DataFrame({"b": m.params, "lo": ci[0], "hi": ci[1]}).drop("Intercept")
        t["sd of x"] = W[t.index].std().values
        out[which] = t
        h = m.t_test("WR1 - WR3 = 0")
        h2 = m.t_test("WR1 - WR2 = 0")
        print(f"\nC ({which} quality): WR1-WR3 = {float(h.effect[0]):.3f} (p {float(h.pvalue):.3f}); "
              f"WR1-WR2 = {float(h2.effect[0]):.3f} (p {float(h2.pvalue):.3f}); n={len(W)}")
        ci2 = m2.conf_int()
        print("   OL split for comparison: " + ", ".join(
            f"{k} {m2.params[k]:.2f} ({ci2.loc[k, 0]:.2f} to {ci2.loc[k, 1]:.2f})" for k in ["OLtop2", "OL345", "OLrest"]))
    print("\nC. Offensive rating (pts/game, season-demeaned) per 1% of cap of each slot's value, team-seasons "
          "2013-25; WRs sorted by value within team")
    print(pd.concat(out, axis=1).round(3).to_string())


def test_d():
    import absence_test as at
    A = pd.read_parquet(OUT / "absence_starters_carry.parquet")
    # per team-game: present (not absent) WR starters and their prices
    A["present_price"] = np.where(~A.absent, A.price_f, 0.0)
    rows = []
    for grp in ["WR", "OL", "CB"]:
        sub = A[A.grp == grp]
        pres = sub.groupby(["game_id", "team"]).present_price.sum().rename("pres")
        npres = sub.assign(p=~sub.absent).groupby(["game_id", "team"]).p.sum().rename("npres")
        A2 = A.join(pres, on=["game_id", "team"]).join(npres, on=["game_id", "team"])
        A2[["pres", "npres"]] = A2[["pres", "npres"]].fillna(0)
        isab = A2.absent & (A2.grp == grp)
        med = A2.loc[isab, "pres"].median()
        A2[f"abs_{grp}_deep"] = (isab & (A2.pres > med)).astype(float)
        A2[f"abs_{grp}_thin"] = (isab & (A2.pres <= med)).astype(float)
        others = [f"abs_{g}" for g in at.GROUPS if g != grp]
        cols = [f"abs_{grp}_deep", f"abs_{grp}_thin"] + others
        s = pd.read_parquet(at.RAW / "schedules.parquet")
        s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
        for c in ["home_team", "away_team"]:
            s[c] = s[c].replace(at.RELOC)
        s = s[s.game_id.isin(set(A2.game_id))]
        M, names, g = at.design(s, A2, cols)
        for y, lab in [("spread_line", "market"), ("result", "margin")]:
            f = at.fit(g[y].to_numpy(float), M, names, g.home_team.to_numpy())
            for c in cols[:2]:
                rows.append({"group": grp, "absent starter": c.split("_")[-1], "yardstick": lab,
                             "pts/game": f.loc[c, "b"], "lo": f.loc[c, "lo"], "hi": f.loc[c, "hi"],
                             "n absent": int(A2[c].sum()), "median price still playing": med})
    print("\nD. Absent starter's worth (pts/game), split by the summed pre-season price (% of cap) of the "
          "group's starters still playing (above / below median)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def test_d_cont():
    """D with the absent starter's own price controlled: absent x (own price, teammates' price), centred."""
    import absence_test as at
    A = pd.read_parquet(OUT / "absence_starters_carry.parquet")
    A["present_price"] = np.where(~A.absent, A.price_f, 0.0)
    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    rows = []
    for grp in ["WR", "OL", "CB", "DL", "TE"]:
        pres = A[A.grp == grp].groupby(["game_id", "team"]).present_price.sum().rename("pres")
        A2 = A.join(pres, on=["game_id", "team"])
        A2["pres"] = A2.pres.fillna(0)
        isab = A2.absent & (A2.grp == grp)
        mp, mo = A2.loc[isab, "pres"].mean(), A2.loc[isab, "price_f"].mean()
        A2["a"] = isab.astype(float)
        A2["a_own"] = np.where(isab, A2.price_f - mo, 0)
        A2["a_pres"] = np.where(isab, A2.pres - mp, 0)
        cols = ["a", "a_own", "a_pres"] + [f"abs_{g}" for g in at.GROUPS if g != grp]
        M, names, g = at.design(s, A2, cols)
        for y, lab in [("spread_line", "market"), ("result", "margin")]:
            f = at.fit(g[y].to_numpy(float), M, names, g.home_team.to_numpy())
            r = {"group": grp, "yardstick": lab}
            for c, nm in [("a", "absent (avg)"), ("a_own", "per 1% own price"), ("a_pres", "per 1% teammates")]:
                r[nm] = f"{-f.loc[c, 'b']:.3f} ({-f.loc[c, 'hi']:.3f} to {-f.loc[c, 'lo']:.3f})"
            r["sd teammates %"] = round(A2.loc[isab, "pres"].std(), 1)
            rows.append(r)
    print("\nD2. Points lost per absent starter (positive = costs points), with his own price and the price of the "
          "group's starters still playing (both centred)")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ps = player_seasons()
    dp = draft_table(ps)
    print(f"player-seasons {len(ps)}, groups {ps.g.value_counts().to_dict()}")
    T = test_a(ps, dp)
    test_b(ps, dp, T)
    test_c(ps)
    test_d()
    test_d_cont()


if __name__ == "__main__":
    main()
