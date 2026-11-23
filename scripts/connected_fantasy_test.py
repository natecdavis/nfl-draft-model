"""Does the coaching-connection edge (drafting staff coached the player in college) show up in fantasy points, beyond
what the fantasy market already prices?

Skill-position rookies (QB/RB/WR/TE) drafted 2012-2025 (2025-26 staffs from scripts/staffs_2025.py). Connection as in
scripts/contact_value.py (draft-year staff coached at the player's final college in the 5 prior seasons). Fair group =
players whose school had a link to some NFL staff (controls for big-program effects), with log # connected teams.

Specified before looking:
  A. Market test (redraft / season props proxy): rookie-year half-PPR points vs preseason FantasyPros ADP
     (2023 Fantasy Draft/Fantasy-Football-Models-master/adp_reports.csv). Expected = Poisson fit of points on
     position x log(ADP) among all rookies with ADP, leave-one-season-out.
  B. Draft-capital test (dynasty rookie drafts, which price off draft capital): half-PPR points in year 1 and
     years 1-3 vs slot, expected = Poisson fit on position x log(pick), leave-one-class-out.
  Report connected vs not (value ratio, Poisson flag coef, beat-expectation odds) via multisport_test.compare.
"""
import re
import sys

import nflreadpy as nr
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
from multisport_test import compare

CLASSES = range(2012, 2026)
SKILL = ["QB", "RB", "WR", "TE"]
ADP = "/Users/nathanieldavis/2023 Fantasy Draft/Fantasy-Football-Models-master/adp_reports.csv"
norm = lambda s: re.sub(r"(jr|sr|ii|iii|iv)$", "", re.sub(r"[^a-z]", "", str(s).lower()))


def connections_ext(window=5):
    """old_class_replication.connections_all plus the 2025-26 staffs and careers."""
    OUT = co.OUT
    staffs = pd.concat([pd.read_parquet(OUT / f) for f in
                        ("nfl_staffs_2006.parquet", "nfl_staffs.parquet", "nfl_staffs_2025.parquet")])
    careers = pd.concat([pd.read_parquet(OUT / f) for f in
                         ("coach_careers.parquet", "coach_careers_2006.parquet", "coach_careers_2025.parquet")])
    car, college = co.college_keys(careers.drop_duplicates())
    car = car.dropna(subset=["key"])
    st = staffs.dropna(subset=["page"]).drop_duplicates(["fr", "year", "page"])
    m = st.merge(car[["page", "key", "year"]].rename(columns={"year": "cyear"}), on="page")
    m = m[(m.cyear >= m.year - window) & (m.cyear <= m.year - 1)]
    cn = m.groupby(["fr", "year", "key", "page", "name", "role"]).cyear.agg(["min", "max"]).reset_index() \
        .rename(columns={"min": "first", "max": "last"})
    return cn, college, staffs[["fr", "year"]].drop_duplicates()


def rookies():
    cn, college, have = connections_ext()
    p = nr.load_draft_picks().to_pandas()
    p = p[p.season.between(min(CLASSES), max(CLASSES)) & p.position.isin(SKILL)].copy()
    p["fr"] = p.team.map(co.franchise)
    p["key"] = p.college.map(college)
    have_set = set(zip(have.fr, have.year))
    p = p[[(f, y) in have_set for f, y in zip(p.fr, p.season)]]
    team = cn.groupby(["fr", "year", "key"]).agg(last=("last", "max")).reset_index()
    p = p.merge(team, left_on=["fr", "season", "key"], right_on=["fr", "year", "key"], how="left").drop(columns="year")
    p["conn"] = p["last"].notna()
    nt = cn.groupby(["year", "key"]).fr.nunique().rename("n_teams_conn").reset_index()
    p = p.merge(nt, left_on=["season", "key"], right_on=["year", "key"], how="left").drop(columns="year")
    p["n_teams_conn"] = p.n_teams_conn.fillna(0)
    p["log_nconn"] = np.log1p(p.n_teams_conn)
    p["cls"] = p.season.astype(int)
    p["pos"] = p.position
    p["nm"] = p.pfr_player_name.map(norm)
    return p[["cls", "pick", "round", "pos", "pfr_player_name", "nm", "gsis_id", "fr", "college", "conn", "n_teams_conn",
              "log_nconn"]]


def points(p):
    s = nr.load_player_stats(list(range(2012, 2026)), summary_level="reg").to_pandas()
    s["half"] = (s.fantasy_points + s.fantasy_points_ppr) / 2
    s = s.groupby(["player_id", "season"]).half.sum().clip(lower=0)
    for t in (1, 2, 3):
        idx = list(zip(p.gsis_id, p.cls + t - 1))
        p[f"pts{t}"] = [s.get(i, 0.0) for i in idx]
        p.loc[p.cls + t - 1 > 2025, f"pts{t}"] = np.nan
    p["pts13"] = p[["pts1", "pts2", "pts3"]].sum(axis=1, min_count=3)
    return p


def loo_expect(d, y, x, by):
    e = pd.Series(np.nan, index=d.index)
    X = pd.get_dummies(d.pos, dtype=float)
    for c in SKILL:
        X[f"{c}_x"] = X[c] * d[x]
    for g in d[by].unique():
        te = d[by] == g
        tr = ~te & d[y].notna()
        m = sm.GLM(d.loc[tr, y], X[tr], family=sm.families.Poisson()).fit()
        e[te] = m.predict(X[te])
    return e


def run(d, y, e, title, fair=True):
    s = d.dropna(subset=[y]).copy()
    s["actual"] = s[y]
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    s = pd.concat([s, fe], axis=1)
    covs = ("log_nconn", *fe.columns)
    print(f"  {title}")
    compare(s, "conn", e, covs=covs, label="all")
    if fair:
        compare(s[s.n_teams_conn > 0], "conn", e, covs=covs, label="fair group (school linked)")
    compare(s[s["round"] <= 2], "conn", e, covs=covs, label="rounds 1-2")
    for pos in SKILL:
        compare(s[s.pos == pos], "conn", e, covs=("log_nconn",), label=f"  {pos} (no class FE)", min_n=4)


def main():
    p = points(rookies())
    print(f"Skill rookies {CLASSES.start}-{CLASSES.stop - 1} with staff data: {len(p)}; connected {p.conn.sum()} "
          f"({p.conn.mean():.1%}); rounds 1-2 connected {(p.conn & (p['round'] <= 2)).sum()}")

    a = pd.read_csv(ADP)
    a["nm"] = a.Name.map(norm)
    a = a.rename(columns={"year": "cls"}).groupby(["cls", "nm"]).fp_adp.min().reset_index()
    m = p.merge(a, on=["cls", "nm"], how="inner")
    m["ladp"] = np.log(m.fp_adp)
    m["e_adp"] = loo_expect(m, "pts1", "ladp", "cls")
    print(f"\nA. MARKET: rookie-year half-PPR vs preseason ADP. Rookies with ADP: {len(m)}; connected {m.conn.sum()}")
    print(m.groupby("cls").agg(n=("nm", "size"), conn=("conn", "sum")).T.to_string())
    run(m, "pts1", "e_adp", "points vs ADP expectation")

    p["lpick"] = np.log(p.pick)
    p["e1"] = loo_expect(p, "pts1", "lpick", "cls")
    p["e13"] = loo_expect(p, "pts13", "lpick", "cls")
    print("\nB. DRAFT CAPITAL: half-PPR vs slot (position x log pick)")
    run(p, "pts1", "e1", "year 1")
    run(p, "pts13", "e13", "years 1-3 (classes 2012-23)")

    show = m[m.conn].assign(ratio=lambda x: x.pts1 / x.e_adp).sort_values(["cls", "fp_adp"])
    print("\nConnected rookies with ADP:")
    print(show[["cls", "pfr_player_name", "pos", "fr", "pick", "fp_adp", "e_adp", "pts1", "ratio"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
