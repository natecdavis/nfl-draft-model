"""EDGE x EDGE pairing: does a good edge rusher make the other edge better, and is a second good edge worth more
(or less) when the first is good?

Specified before looking (Nov 22 2026). PFF season files 2006-25 (defense_summary: total_pressures,
snap_counts_pass_rush, grades_pass_rush_defense), position ED; DI = interior.
A. SPILLOVER (player level). ED player-seasons with >= 150 pass-rush snaps. Outcome: pressure rate (pressures per
   pass-rush snap). Own prior: his pressure rate over the 2 previous seasons (>= 150 snaps there, shrunk toward the
   ED mean with k = 300 snaps). Partner: the team's other ED with the most pass-rush snaps that season (>= 150), his
   prior measured the same way (earlier seasons, so not this season's shared opponents/scheme). Controls: own prior,
   best DI's prior, age, season effects, team's coverage grade the season before. Clustered by team-season.
   PRIMARY: pressure rate per +1 sd of partner prior. SECONDARY: pass-rush grade outcome; player fixed effects
   (same player, partner changes); sacks per snap (sacks can be 'stolen' or 'cleaned up'); partner top-10% flag.
B. PAIRING VALUE (team level). Team-seasons 2006-25: defensive points rating (synergy_test.ratings, higher =
   fewer allowed) and team pressure rate on ED1 / ED2 (top two ED by pass-rush snaps, pass-rush grade z-scored
   within season), best DI, coverage, run defense, LB unit grades. PRIMARY: ED1 x ED2 interaction on the defensive
   rating (positive = complements: a second good edge is worth more next to a good first; negative =
   substitutes). SECONDARY: same on team pressure rate; ED x DI interactions; team fixed effects.
The model values each player additively, so only B's interaction (or A's spillover) would change a pick's value.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from synergy_test import PFF, TEAM, YEARS, ratings, units

K = 300


def load():
    rows = []
    for y in YEARS:
        d = pd.read_csv(PFF / str(y) / "defense_summary.csv").drop_duplicates("player_id")
        d = d[d.position.isin(["ED", "DI"])].assign(season=y, team=lambda x: x.team_name.replace(TEAM))
        rows.append(d[["player_id", "player", "position", "team", "season", "total_pressures", "sacks",
                       "snap_counts_pass_rush", "grades_pass_rush_defense"]])
    d = pd.concat(rows, ignore_index=True).rename(columns={"snap_counts_pass_rush": "prs",
                                                            "grades_pass_rush_defense": "prg"})
    d = d[d.prs > 0].copy()
    d["rate"] = d.total_pressures / d.prs
    d["srate"] = d.sacks / d.prs
    # prior: previous 2 seasons, pooled, any team, shrunk to the position mean
    by = d.groupby(["player_id", "season"])[["total_pressures", "prs"]].sum()
    mu = d.groupby("position").apply(lambda x: x.total_pressures.sum() / x.prs.sum())
    pri, pri_n = [], []
    for pid, yr, pos in zip(d.player_id, d.season, d.position):
        p = by.loc[pid] if pid in by.index.get_level_values(0) else None
        if p is None:
            pri.append(np.nan); pri_n.append(0); continue
        p = p[(p.index >= yr - 2) & (p.index < yr)]
        n = p.prs.sum()
        pri.append((p.total_pressures.sum() + K * mu[pos]) / (n + K) if n >= 150 else np.nan); pri_n.append(n)
    d["prior"], d["prior_n"] = pri, pri_n
    age = pd.read_parquet(ROOT / "data/derived/darko_pff.parquet")[["player_id", "season", "age"]]
    return d.merge(age.drop_duplicates(["player_id", "season"]), on=["player_id", "season"], how="left")


def part_a(d, U):
    ed = d[(d.position == "ED") & (d.prs >= 150)].copy()
    di = d[(d.position == "DI") & (d.prs >= 150) & d.prior.notna()]
    best_di = di.groupby(["team", "season"]).prior.max().rename("di_prior")
    rows = []
    for (tm, yr), g in ed.groupby(["team", "season"]):
        g = g.sort_values("prs", ascending=False)
        for i, r in g.iterrows():
            o = g.drop(i)
            if len(o):
                p = o.iloc[0]
                rows.append((i, p.prior, p.player_id))
    P = pd.DataFrame(rows, columns=["i", "partner_prior", "partner_id"]).set_index("i")
    ed = ed.join(P).join(best_di, on=["team", "season"])
    cov = U.set_index(["season", "team"])["cov"]
    ed["cov_prev"] = [cov.get((y - 1, t), np.nan) for y, t in zip(ed.season, ed.team)]
    ed["age"] = ed.age.fillna(ed.age.median())
    q = ed.dropna(subset=["prior", "partner_prior", "di_prior", "cov_prev"]).copy()
    sdp = q.partner_prior.std()
    q["partner_z"] = (q.partner_prior - q.partner_prior.mean()) / sdp
    q["di_z"] = (q.di_prior - q.di_prior.mean()) / q.di_prior.std()
    q["top"] = (q.partner_prior >= ed.prior.quantile(0.9)).astype(float)
    q["cl"] = q.team + q.season.astype(str)
    print(f"A. ED player-seasons >= 150 pass-rush snaps: {len(ed)}; with own + partner + DI prior and coverage: {len(q)}"
          f"\n   mean pressure rate {q.rate.mean():.3f} (sd {q.rate.std():.3f}); partner prior sd {sdp:.3f}")
    ctl = "prior + di_z + age + I(age**2) + cov_prev + C(season)"

    def run(lab, y, term, data, f=ctl, fe=False):
        if fe:
            f = f + " + C(player_id)"
        m = smf.ols(f"{y} ~ {term} + {f}", data).fit(cov_type="cluster", cov_kwds={"groups": data.cl})
        ci = m.conf_int().loc[term]
        sd = data[y].std()
        print(f"   {lab:42s} n={len(data):5d} | {term} {m.params[term]:+.4f} ({ci[0]:+.4f} to {ci[1]:+.4f}, "
              f"p {m.pvalues[term]:.2f}) = {m.params[term] / sd:+.3f} sd of outcome")
        return m

    print("   PRIMARY")
    run("pressure rate on partner prior (per sd)", "rate", "partner_z", q)
    print("   SECONDARY")
    run("pass-rush grade", "prg", "partner_z", q)
    run("sacks per snap", "srate", "partner_z", q)
    run("partner top-10% prior", "rate", "top", q)
    multi = q[q.groupby("player_id").partner_id.transform("nunique") > 1]
    run("player fixed effects (partner changed)", "rate", "partner_z", multi, fe=True)
    run("DI prior (per sd), same model", "rate", "di_z", q)
    return q


def part_b(d, U, R):
    ed = d[d.position == "ED"].sort_values("prs", ascending=False)
    top2 = ed.groupby(["team", "season"]).head(2).copy()
    top2["k"] = top2.groupby(["team", "season"]).cumcount() + 1
    W = top2.pivot_table(index=["season", "team"], columns="k", values="prg").rename(columns={1: "ed1", 2: "ed2"})
    di = d[d.position == "DI"].sort_values("prs", ascending=False).groupby(["team", "season"]).head(1)
    W = W.join(di.set_index(["season", "team"]).prg.rename("di1"))
    tp = d.groupby(["season", "team"])[["total_pressures", "prs"]].sum()
    W["team_rate"] = tp.total_pressures / tp.prs
    D = W.reset_index().merge(U, on=["season", "team"]).merge(R, on=["season", "team"]).dropna(
        subset=["ed1", "ed2", "di1", "cov", "rdef", "lb"])
    for c in ["ed1", "ed2", "di1", "cov", "rdef", "lb"]:
        D[c] = D.groupby("season")[c].transform(lambda x: (x - x.mean()) / x.std())
    # order the pair by grade (stronger = a), so the interaction is symmetric in who is called ED1
    D["a"], D["b"] = D[["ed1", "ed2"]].max(axis=1), D[["ed1", "ed2"]].min(axis=1)
    print(f"\nB. team-seasons: {len(D)}; defensive rating sd {D.dfn.std():.2f} pts/game")
    base = "a + b + di1 + cov + rdef + lb"

    def run(lab, y, f, terms, fe=False):
        if fe:
            f += " + C(team)"
        m = smf.ols(f"{y} ~ {f}", D).fit(cov_type="cluster", cov_kwds={"groups": D.team})
        ci = m.conf_int()
        print(f"   {lab:34s} " + "  ".join(f"{t} {m.params[t]:+.3f} [{ci.loc[t, 0]:+.3f},{ci.loc[t, 1]:+.3f}]"
                                         for t in terms))
        return m

    print("   PRIMARY (defensive rating, pts/game per sd)")
    run("ED pair interaction", "dfn", base + " + a:b", ["a", "b", "a:b"])
    print("   SECONDARY")
    run("team pressure rate", "team_rate", base + " + a:b", ["a", "b", "a:b"])
    run("ED x DI interactions", "dfn", base + " + a:b + a:di1 + b:di1", ["a:b", "a:di1", "b:di1"])
    run("team fixed effects", "dfn", base + " + a:b", ["a", "b", "a:b"], fe=True)
    tr, te = D[D.season <= 2017], D[D.season >= 2018]
    e = {nm: np.sqrt(((te.dfn - smf.ols(f"dfn ~ {f}", tr).fit().predict(te)) ** 2).mean())
         for nm, f in [("additive", base), ("with a:b", base + " + a:b")]}
    print(f"   out of sample 2018-25 RMSE: additive {e['additive']:.3f}, with a:b {e['with a:b']:.3f}")
    return D


def main():
    d = load()
    U = units()
    R = ratings()
    q = part_a(d, U)
    D = part_b(d, U, R)
    q.to_parquet(ROOT / "data/derived/edge_pair_a.parquet")
    D.to_parquet(ROOT / "data/derived/edge_pair_b.parquet")


if __name__ == "__main__":
    main()
