"""Out-of-sample replication on pre-board classes (2006-2015): coaching contact and EDGE height.

The original tests used drafted players 2016-2022 (coaching_test.py, contact_test.py, size_bias_test.py).
Classes 2006-2015 never entered those tests, and their careers are complete. With no consensus board before
2014, only the "vs draft slot" versions can be replicated.

Expected value from draft slot: the project's slot model (pm.fit, version "slot": position group + log pick,
Tweedie GBM, monotone in pick) fit leave-one-class-out within 2006-2015, so each class is predicted from
the other nine (same era, own class excluded). Label: pm.LABEL_FILE (default darko), seasons 1-4, never in
the NFL = 0. Drafted players only (undrafted careers are labeled from 2016).

Specified before looking (originals in brackets):
  A. Coaching connection (src/coaching.py definition: someone on the drafting team's draft-year staff coached
     at the player's final college in the previous 5 seasons). Visits begin in 2015, so this is coaching only.
     Picks by team-seasons whose Wikipedia staff wasn't parsed are dropped.
     A1 primary: players whose school had a link to some NFL staff, drafted by a connected team vs not,
         covariate log(1 + # connected teams)          [coaching x1.06 n.s.; pooled contact x1.10 p 0.06]
     A2 picks 1-100                                   [pooled contact x1.14 p 0.03]
     A3 recent link (last 2 seasons), rounds 1-3       [x1.25, 28 players]
     each also with class fixed effects, and a random-team placebo (connection status of a random other
     team with staff data, 500 draws): the real effect should sit in the placebo's upper tail.
  B. EDGE height (z within position group, using the original test's universe mean / sd)
     B1 primary: value per sd vs slot, EDGE, class fixed effects   [x1.13-1.18 vs slot; x1.26 vs consensus]
     B2 controlling weight z; B3 split 2006-10 / 2011-15 (rookie wage scale); other groups for context.
"""
import sys

import nflreadpy as nr
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import prospect_model as pm
from multisport_test import compare
from size_bias_test import load as size_load, value_cont

CLASSES = range(2006, 2016)
OUT = pm.OUT


def expected_slot():
    L = pd.read_parquet(pm.LABEL_FILE)
    d = L[L.draft_year.between(min(CLASSES), max(CLASSES)) & L.drafted.astype(bool)].copy()
    d = d.dropna(subset=["overall_pick", "group"]).drop_duplicates("pfr_id")
    d["cls"] = d.draft_year.astype(int)
    d["grp"] = d.group
    d["pick"] = d.overall_pick.astype(int)
    d["log_pick"] = np.log(d.pick + 10)
    d["grp_code"] = d.grp.astype("category").cat.codes
    for t in pm.SEASONS:
        d[f"y{t}"] = d[f"value_y{t}"].fillna(0.0).clip(lower=0)
    d["actual"] = d[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    d["exp_slot"] = 0.0
    for c in CLASSES:
        te = d.cls == c
        for t in pm.SEASONS:
            m, feats = pm.fit(d[~te], "slot", t)
            d.loc[te, "exp_slot"] += m.predict(d.loc[te, feats])
    r = d.actual.sum() / d.exp_slot.sum()
    print(f"drafted 2006-15 with labels: {len(d)}; actual / slot expectation overall x{r:.2f}; by class "
          + " ".join(f"{c}:{g.actual.sum() / g.exp_slot.sum():.2f}" for c, g in d.groupby("cls")))
    return d


def connections_all(window=5):
    staffs = pd.concat([pd.read_parquet(OUT / "nfl_staffs_2006.parquet"), pd.read_parquet(OUT / "nfl_staffs.parquet")])
    careers = pd.concat([pd.read_parquet(OUT / "coach_careers.parquet"), pd.read_parquet(OUT / "coach_careers_2006.parquet")])
    car, college = co.college_keys(careers.drop_duplicates())
    car = car.dropna(subset=["key"])
    st = staffs.dropna(subset=["page"]).drop_duplicates(["fr", "year", "page"])
    m = st.merge(car[["page", "key", "year"]].rename(columns={"year": "cyear"}), on="page")
    m = m[(m.cyear >= m.year - window) & (m.cyear <= m.year - 1)]
    cn = m.groupby(["fr", "year", "key", "page", "name", "role"]).cyear.agg(["min", "max"]).reset_index() \
        .rename(columns={"min": "first", "max": "last"})
    have = staffs[["fr", "year"]].drop_duplicates()
    return cn, college, have


def coaching_part(d):
    cn, college, have = connections_all()
    p = nr.load_draft_picks().to_pandas()
    p = p[p.season.between(min(CLASSES), max(CLASSES))].copy()
    p["fr"] = p.team.map(co.franchise)
    p["key"] = p.college.map(college)
    have_set = set(zip(have.fr, have.year))
    p = p[[(f, y) in have_set for f, y in zip(p.fr, p.season)]]
    team = cn.groupby(["fr", "year", "key"]).agg(last=("last", "max")).reset_index()
    p = p.merge(team, left_on=["fr", "season", "key"], right_on=["fr", "year", "key"], how="left").drop(columns="year")
    p["conn"] = p["last"].notna()
    p["conn_recent"] = p.conn & (p["last"] >= p.season - 2)
    linked = cn.groupby(["year", "key"]).fr.agg(set).rename("teams").reset_index()
    p = p.merge(linked, left_on=["season", "key"], right_on=["year", "key"], how="left").drop(columns="year")
    p["teams"] = p.teams.apply(lambda s: s if isinstance(s, set) else set())
    teams_with_staff = have.groupby("year").fr.agg(list).to_dict()
    p["n_teams_conn"] = p.teams.map(len)
    v = d[["pfr_id", "cls", "grp", "pick", "actual", "exp_slot"]].merge(
        p[["pfr_player_id", "fr", "round", "conn", "conn_recent", "teams", "n_teams_conn"]],
        left_on="pfr_id", right_on="pfr_player_id", how="inner")
    v["log_nconn"] = np.log1p(v.n_teams_conn)
    fe = pd.get_dummies(v.cls, prefix="c", drop_first=True, dtype=float)
    v = pd.concat([v, fe], axis=1)
    fec = tuple(fe.columns)
    s = v[v.n_teams_conn > 0]
    print(f"\nA. COACHING CONNECTION, picks 2006-15 with staff data and outcomes: {len(v)}; connected {v.conn.sum()} "
          f"({v.conn.mean():.1%}); school linked to some staff {len(s)}; teams with staff data per year "
          f"{ {y: len(t) for y, t in sorted(teams_with_staff.items()) if y < 2016} }")
    for lab, g, flag in [("A1 primary: linked schools", s, "conn"), ("A2 picks 1-100", s[s.pick <= 100], "conn"),
                         ("A3 recent link, rounds 1-3", s[s["round"] <= 3], "conn_recent")]:
        print(f"  {lab}")
        compare(g, flag, "exp_slot", covs=("log_nconn",), label="as originally specified")
        compare(g, flag, "exp_slot", covs=("log_nconn",) + fec, label="+ class fixed effects")
        compare(g[g.cls <= 2010], flag, "exp_slot", covs=("log_nconn",), label="2006-10 only")
        compare(g[g.cls >= 2011], flag, "exp_slot", covs=("log_nconn",), label="2011-15 only")
        print(f"    placebo: {placebo(g, flag, teams_with_staff)}")
    for y, g in s.groupby("cls"):
        r = lambda x: x.actual.sum() / x.exp_slot.sum()
        print(f"    {y}: connected {g.conn.sum():3d}  value x{r(g[g.conn]) / r(g[~g.conn]):.2f}")
    v.drop(columns=list(fec) + ["teams"]).to_parquet(OUT / "old_class_coaching.parquet")


def placebo(g, flag, teams_with_staff, n=500):
    """Ratio for the real flag vs the same ratio when each pick is credited with a random other team's
    connection (for `conn_recent`, the placebo uses any connection of that team; a conservative stand-in)."""
    r = lambda f: (g.actual[f].sum() / g.exp_slot[f].sum()) / (g.actual[~f].sum() / g.exp_slot[~f].sum())
    real = r(g[flag].values)
    rng = np.random.default_rng(0)
    others = [[t for t in teams_with_staff[y] if t != fr] for y, fr in zip(g.cls, g.fr)]
    sims = []
    for _ in range(n):
        pick = [o[rng.integers(len(o))] for o in others]
        f = np.array([t in s for t, s in zip(pick, g.teams)])
        if f.sum() >= 5 and (~f).sum() >= 5:
            sims.append(r(f))
    sims = np.array(sims)
    return (f"real x{real:.2f}; random-team flag median x{np.median(sims):.2f} (90% {np.percentile(sims, 5):.2f}-"
            f"{np.percentile(sims, 95):.2f}); share of placebos >= real {np.mean(sims >= real):.2f}")


def height_part(d):
    _, u = size_load()  # original universe (2012-25): its per-group height mean / sd define the z scale
    ref = u.groupby("grp")[["ht", "wt"]].agg(["mean", "std"])
    c = pd.read_parquet(pm.CF / "data/processed/combine_metrics.parquet")[["player_id", "combine_ht", "combine_wt"]]
    c["player_id"] = c.player_id.astype(str)
    h = d.assign(player_id=d.player_id.astype(str)).merge(
        c.drop_duplicates("player_id").rename(columns={"combine_ht": "ht", "combine_wt": "wt"}), on="player_id", how="left")
    for x in ("ht", "wt"):
        h[f"{x}_z"] = (h[x] - h.grp.map(ref[(x, "mean")])) / h.grp.map(ref[(x, "std")])
    print(f"\nB. HEIGHT vs SLOT, drafted 2006-15 with combine height: {h.ht.notna().sum()} of {len(h)}")
    e = h[h.grp == "EDGE"]
    value_cont(e, "exp_slot", "ht_z", "B1 primary: EDGE height")
    value_wt(e, "B2 EDGE height, controlling weight")
    value_cont(e[e.cls <= 2010], "exp_slot", "ht_z", "B3 EDGE height, 2006-10")
    value_cont(e[e.cls >= 2011], "exp_slot", "ht_z", "B3 EDGE height, 2011-15")
    value_cont(e[e.pick <= 100], "exp_slot", "ht_z", "   EDGE height, picks 1-100")
    print("  context: other groups")
    value_cont(h, "exp_slot", "ht_z", "all groups")
    for g in ["QB", "DB", "WR", "OL", "DL", "LB", "TE", "RB"]:
        value_cont(h[h.grp == g], "exp_slot", "ht_z", f"  {g}")
    h.to_parquet(OUT / "old_class_height.parquet")


def value_wt(s, label):
    import statsmodels.api as sm
    s = s.dropna(subset=["exp_slot", "ht_z", "wt_z"])
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([s[["ht_z", "wt_z"]].astype(float), fe], axis=1))
    m = sm.GLM(s.actual, X, family=sm.families.Poisson(), offset=np.log(s.exp_slot.clip(lower=1e-4))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc["ht_z"])
    print(f"    {label:40s} n={len(s):4d}  per 1 sd: value x{np.exp(m.params['ht_z']):.2f} (95% {lo:.2f}-{hi:.2f}, "
          f"p {m.pvalues['ht_z']:.2f}); weight x{np.exp(m.params['wt_z']):.2f}")


def main():
    d = expected_slot()
    coaching_part(d)
    height_part(d)


if __name__ == "__main__":
    main()
