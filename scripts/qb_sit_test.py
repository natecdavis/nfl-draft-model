"""Rookie QBs: does sitting behind an incumbent help, hurt, or not matter, and is the model's rule right?

The model's rule (pick_value.pick_effect): a drafted QB's season-by-season snap shares come from comparable
historical QBs at his talent level; seasons before the team's QB need begins (qb_from) are set to zero ("sits
behind the starter"); later seasons keep the comparables' shares. Two claims to test:
  1. A rookie behind a solid incumbent delivers ~nothing in those seasons.
  2. Sitting doesn't change what he becomes: his later seasons match comparables (most of whom started early).

Rookie QBs drafted 2006-2022 (labels seasons 1-5, pm.LABEL_FILE). Playing time: share of his team's dropbacks
(attempts + sacks) by season, nflverse season stats. Incumbent on draft day (pre-draft proxy, all years): the
team's main QB of the previous season had 300+ dropbacks, EPA per dropback at or above that season's league
median (QBs with 200+), and is on the team the next season (some snaps; mild post-draft information). For
2014-22 the model's own draft-day QB status (review_states: qb_from >= 2 = incumbent) is a robustness check.

Tests (specified before running):
  A. By season 1-4: actual value / slot expectation (slot model per season, fit leave-one-class-out over all
     drafted players 2006-22), rookies with vs without a solid incumbent, rounds 1-3. The model predicts ~0 in
     the sitting seasons and ~1.0 after.
  B. Effect of sitting year 1 (share < 25%) on later outcomes: value seasons 2-4, value season 5, starter
     (50%+ share) in season 3 or 4, EPA per dropback seasons 2-4 (players with 200+ dropbacks). OLS with
     log pick, age, era (2011 CBA); then 2SLS using the incumbent as the instrument for sitting.
Rounds 1-3 primary (later-round QBs mostly never play); all rounds for context.

Run:  python scripts/qb_sit_test.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from review import team as nt  # noqa: E402

CLASSES = range(2006, 2023)
SIT_SHARE, START_SHARE = 0.25, 0.5


def season_stats():
    s = pd.concat([pd.read_parquet(RAW / "player_stats_1999_2011.parquet"), pd.read_parquet(RAW / "player_stats.parquet")])
    s = s[s.season_type == "REG"].copy()
    s["team"] = s.recent_team.map(nt)
    s["db"] = s.attempts.fillna(0) + s.sacks_suffered.fillna(0)
    s["epa"] = s.passing_epa.fillna(0)
    q = s[s.db > 0]
    team_db = q.groupby(["season", "team"]).db.sum()
    q = q.assign(share=lambda x: (x.db / x.set_index(["season", "team"]).index.map(team_db)).clip(upper=1.0))
    med = q[q.db >= 200].assign(e=lambda x: x.epa / x.db).groupby("season").e.median()
    return q, med


def incumbents(q, med):
    """(draft year, team) -> solid incumbent on draft day (pre-draft proxy)."""
    out = {}
    for (y, tm), g in q.groupby(["season", "team"]):
        main = g.sort_values("db").iloc[-1]
        solid = main.db >= 300 and main.epa / main.db >= med.get(y, 0)
        stays = ((q.season == y + 1) & (q.team == tm) & (q.player_id == main.player_id)).any()
        out[(y + 1, tm)] = bool(solid and stays)
    return out


def rookies():
    L = pd.read_parquet(pm.LABEL_FILE)
    d = L[L.draft_year.between(min(CLASSES), max(CLASSES)) & L.drafted.astype(bool)].copy()
    d = d.dropna(subset=["overall_pick", "group"]).drop_duplicates("pfr_id")
    d["cls"], d["grp"], d["pick"] = d.draft_year.astype(int), d.group, d.overall_pick.astype(int)
    d["log_pick"] = np.log(d.pick + 10)
    d["grp_code"] = d.grp.astype("category").cat.codes
    for t in range(1, 6):
        d[f"y{t}"] = d[f"value_y{t}"].fillna(0.0).clip(lower=0) * 100
    # per-season slot expectation, leave one class out (all positions, so the QB curve is the slot model's)
    for t in range(1, 5):
        d[f"e{t}"] = np.nan
        for c in CLASSES:
            te = d.cls == c
            tr = d[~te].assign(**{f"y{t}": lambda x: x[f"y{t}"] / 100})
            m, feats = pm.fit(tr, "slot", t)
            d.loc[te, f"e{t}"] = m.predict(d.loc[te, feats]) * 100
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    info = dp.set_index("pfr_player_id")[["team", "gsis_id", "age"]]
    d = d.join(info, on="pfr_id")
    d["team"] = d.team.map(nt)
    return d[d.grp == "QB"].copy()


def main():
    q, med = season_stats()
    inc = incumbents(q, med)
    d = rookies()
    d["incumbent"] = [inc.get((c, tm), False) for c, tm in zip(d.cls, d.team)]
    sh = q.set_index(["player_id", "season"]).share
    ep = q.set_index(["player_id", "season"])[["epa", "db"]]
    for t in range(1, 6):
        d[f"s{t}"] = [float(sh.get((g, c + t - 1), 0.0)) if isinstance(g, str) else np.nan for g, c in zip(d.gsis_id, d.cls)]
    e = []
    for g, c in zip(d.gsis_id, d.cls):
        rows = [ep.loc[(g, c + k)] for k in (1, 2, 3) if isinstance(g, str) and (g, c + k) in ep.index]
        db = sum(float(r.db) for r in rows)
        e.append(sum(float(r.epa) for r in rows) / db if db >= 200 else np.nan)
    d["epa24"] = e
    d["sat1"] = (d.s1 < SIT_SHARE).astype(float)
    d["start34"] = ((d.s3 >= START_SHARE) | (d.s4 >= START_SHARE)).astype(float)
    d["v24"], d["e24"] = d.y2 + d.y3 + d.y4, d.e2 + d.e3 + d.e4
    d["era"] = (d.cls >= 2011).astype(float)
    d["age"] = d.age.fillna(d.age.median())
    S = pd.read_parquet(OUT / "review_states.parquet")
    qf = {(y, t): f for y, t, f in zip(S.year, S.team, S.qb_from)}
    d["model_inc"] = [np.nan if c < 2014 else float(pd.notna(qf.get((c, tm))) and qf.get((c, tm)) >= 2)
                      for c, tm in zip(d.cls, d.team)]
    print(f"Drafted QBs 2006-22: {len(d)} (rounds 1-3: {(d.pick <= 100).sum()}); solid incumbent: {d.incumbent.mean():.0%}; "
          f"proxy vs model's qb_from>=2 (2014-22) agree {(d.incumbent == d.model_inc)[d.model_inc.notna()].mean():.0%}")

    for lab, m in [("ROUNDS 1-3 (picks <= 100)", d.pick <= 100), ("ROUND 1", d.pick <= 32), ("ALL ROUNDS", d.pick > 0)]:
        x = d[m]
        print(f"\nA. {lab}: actual / slot expectation by season, and mean share of team dropbacks")
        rows = []
        for gname, g in [("solid incumbent", x[x.incumbent]), ("no solid incumbent", x[~x.incumbent])]:
            r = {"group": gname, "n": len(g)}
            for t in range(1, 5):
                r[f"value y{t}"] = g[f"y{t}"].sum() / g[f"e{t}"].sum()
            for t in range(1, 5):
                r[f"share y{t}"] = g[f"s{t}"].mean()
            r["sat y1"] = g.sat1.mean()
            rows.append(r)
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    x = d[d.pick <= 100].copy()
    print("\nB. Effect of sitting year 1 (share < 25%), rounds 1-3. Controls: log pick, age, era.")
    ctrl = ["log_pick", "age", "era"]
    fs = sm.OLS(x.sat1, sm.add_constant(x[["incumbent"] + ctrl].astype(float))).fit(cov_type="HC1")
    print(f"   first stage: solid incumbent -> sat year 1 {fs.params['incumbent']:+.2f} (t {fs.tvalues['incumbent']:.1f}, "
          f"F {fs.tvalues['incumbent'] ** 2:.0f}); sat rate {x.sat1.mean():.0%}")
    for y, lab in [("v24", "value seasons 2-4 (% cap)"), ("y5", "value season 5"), ("start34", "starter in season 3 or 4"),
                   ("epa24", "EPA / dropback seasons 2-4 (200+ db)")]:
        z = x.dropna(subset=[y])
        X = sm.add_constant(z[["sat1"] + ctrl].astype(float))
        ols = sm.OLS(z[y], X).fit(cov_type="HC1")
        # 2SLS by hand: fitted sat from the first stage, robust se via the second stage on residual-corrected X
        f1 = sm.OLS(z.sat1, sm.add_constant(z[["incumbent"] + ctrl].astype(float))).fit()
        Xh = X.copy(); Xh["sat1"] = f1.fittedvalues
        iv = sm.OLS(z[y], Xh).fit()
        resid = z[y] - X @ iv.params
        A = Xh.to_numpy(float)
        XtX = np.linalg.inv(A.T @ A)
        V = XtX @ (A.T * resid.to_numpy() ** 2) @ A @ XtX
        se = float(np.sqrt(V[list(X.columns).index("sat1")] [list(X.columns).index("sat1")]))
        rf = sm.OLS(z[y], sm.add_constant(z[["incumbent"] + ctrl].astype(float))).fit(cov_type="HC1")
        print(f"   {lab:38s} n={len(z):3d} mean {z[y].mean():6.2f} | OLS sat {ols.params['sat1']:+.2f} (se {ols.bse['sat1']:.2f}) | "
              f"reduced form incumbent {rf.params['incumbent']:+.2f} (se {rf.bse['incumbent']:.2f}) | IV sat {iv.params['sat1']:+.2f} (se {se:.2f})")
    print("\n   Seasons 2-4 vs slot expectation (seasons 2-4), rounds 1-3, by year-1 role:")
    for gname, g in [("sat year 1", x[x.sat1 == 1]), ("played year 1", x[x.sat1 == 0])]:
        print(f"     {gname:14s} n={len(g):3d}  actual/expected seasons 2-4 x{g.v24.sum() / g.e24.sum():.2f}  "
              f"season 5 value mean {g.y5.mean():.2f}  starter y3-4 {g.start34.mean():.0%}")
    # C. The model's own rule, 2014-22: seasons before qb_from are zeroed. What did those rookies deliver?
    print("\nC. Model's draft-day QB status (2014-22, all rounds / rounds 1-3): seasons before the need begins")
    z = d[d.cls >= 2014].copy()
    z["qb_from"] = [qf.get((c, tm)) for c, tm in zip(z.cls, z.team)]
    z["status"] = [dict(zip(zip(S.year, S.team), S.status)).get((c, tm)) for c, tm in zip(z.cls, z.team)]
    for lab, zz in [("all rounds", z), ("rounds 1-3", z[z.pick <= 100])]:
        rows = []
        for t in range(1, 5):
            for grp, m in [("model: sits (t < qb_from)", zz.qb_from.notna() & (zz.qb_from > t)),
                           ("model: plays (t >= qb_from)", zz.qb_from.notna() & (zz.qb_from <= t)),
                           ("model: no QB need (never plays)", zz.qb_from.isna())]:
                g = zz[m & (zz.cls + t - 1 <= 2025)]
                if len(g):
                    rows.append({"season": t, "group": grp, "n": len(g), "mean share": g[f"s{t}"].mean(),
                                 "share >= 25%": (g[f"s{t}"] >= SIT_SHARE).mean(),
                                 "value / slot exp.": g[f"y{t}"].sum() / g[f"e{t}"].sum()})
        print(f"  -- {lab}")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\n  Rounds 1-3 rookies the model has sitting in season 1, by draft-day status:")
    g = z[(z.pick <= 100) & z.qb_from.notna() & (z.qb_from >= 2)]
    print(g.groupby("status").agg(n=("s1", "size"), qb_from=("qb_from", "mean"), share_y1=("s1", "mean"),
                                  share_y2=("s2", "mean"), value_y1_sum=("y1", "sum")).round(2).to_string())
    print(g.sort_values("s1", ascending=False).head(10)[["player_name", "cls", "team", "pick", "status", "qb_from", "s1", "s2"]]
          .round(2).to_string(index=False))
    d.to_parquet(OUT / "qb_sit.parquet")


if __name__ == "__main__":
    main()
