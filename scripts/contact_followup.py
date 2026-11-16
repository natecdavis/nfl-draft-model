"""Follow-ups to scripts/contact_value.py (drafted 2006-22, slot model fit leave-one-class-out, saved in
data/derived/contact_value.parquet). Specified before looking:

A. Which NFL jobs do college-linked coaches hold? Coach-seasons with a college link in the 5 seasons before the
   draft (college programs only), by role; also "fresh from college" (coached in college the season before).
   Connected picks per coach-season by role; draft value per coach-season = picks x $ per connected pick.
B. Can connected coaches help avoid players?
   B1 near passes: for each drafted player at a linked school, connected teams' picks of other players in the W picks
      just before his (W = 10, 32), when he was closest to his market price. Value vs slot per near pass,
      controlling # linked teams and class; placebo: the same count for random teams (same number as linked), 500 draws.
   B2 accuracy: connected vs unconnected picks (fair group): bust rate (actual < 25% of expected), star rate (> 2x),
      spread of log((actual + 0.01) / (expected + 0.01)).
   B3 picking among teammates: school-class groups with 2+ drafted players including a connected pick: the connected
      pick vs his schoolmates taken by other teams, value vs slot with school-class fixed effects.
C. Does the edge need the coach on staff? Link types per pick (drafting team, coach coached at the player's school
   in the 5 seasons before the draft):
      current  - coach on the draft-year staff (the main measure)
      former   - coach on the team's staff in one of the 2 prior seasons, gone by the draft (still reachable by phone)
      future   - coach joins the team's staff the season after the draft, not on it before (placebo: can't have helped)
   Each (only that type) vs unlinked picks, fair group, class fixed effects.
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
from contact_value import CAP_2025, role_kind
from multisport_test import compare
from outlook import OUT

PER_PICK = 1.88  # $M, contact_value part 1


def load():
    v = pd.read_parquet(OUT / "contact_value.parquet")
    staffs = pd.concat([pd.read_parquet(OUT / "nfl_staffs_2006.parquet"), pd.read_parquet(OUT / "nfl_staffs.parquet")])
    careers = pd.concat([pd.read_parquet(OUT / "coach_careers.parquet"), pd.read_parquet(OUT / "coach_careers_2006.parquet")])
    car, college = co.college_keys(careers.drop_duplicates())
    dp = nr.load_draft_picks().to_pandas()
    dp = dp[dp.season.between(2006, 2022)].copy()
    dp["fr"] = dp.team.map(co.franchise)
    dp["key"] = dp.college.map(college)
    ckeys = set(dp.key.dropna())
    car = car[car.key.isin(ckeys)][["page", "key", "year"]].drop_duplicates()
    st = staffs.dropna(subset=["page"]).drop_duplicates(["fr", "year", "page"])
    return v, st, car, dp


def detail_role(role):
    r = str(role).lower().strip()
    if r.startswith("head coach"):
        return "1 head coach"
    if "coordinator" in r and not r.startswith("assistant"):
        return "2 coordinator"
    if re.search(r"quality control|intern|fellow|analyst", r):
        return "6 quality control / intern"
    if r.startswith("assistant") or "assistant" in r:
        return "5 assistant (position / ST / general)"
    if role_kind(role) == "position":
        return "3 position coach"
    return "4 other / senior / unclear"


def part_a(v, st, car):
    years = range(2006, 2023)
    links = st[st.year.isin(years)].merge(car.rename(columns={"year": "cy"}), on="page")
    links = links[(links.cy >= links.year - 5) & (links.cy <= links.year - 1)]
    lk = links.groupby(["fr", "year", "page", "role"]).cy.max().rename("last").reset_index()
    lk["fresh"] = lk["last"] == lk.year - 1
    lk["kind"] = lk.role.map(detail_role)
    allst = st[st.year.isin(years)].assign(kind=lambda x: x.role.map(detail_role))
    # connected picks credited to each linked coach (split equally when several coaches link the same pick)
    pk = v[v.conn][["pfr_id", "fr", "cls", "key", "exp_slot"]]
    cl = pk.merge(links[["fr", "year", "page", "key"]].drop_duplicates(), left_on=["fr", "cls", "key"],
                  right_on=["fr", "year", "key"])
    cl["share"] = 1 / cl.groupby("pfr_id").page.transform("nunique")
    cred = cl.groupby(["fr", "year", "page"]).share.sum()
    lk["picks"] = [cred.get((f, y, p), 0.0) for f, y, p in zip(lk.fr, lk.year, lk.page)]
    print("A. NFL JOBS OF COLLEGE-LINKED COACHES (coach-seasons 2006-22, college link in the 5 prior seasons)")
    t = pd.DataFrame({"all staff": allst.kind.value_counts(normalize=True),
                      "linked": lk.kind.value_counts(normalize=True),
                      "fresh from college": lk[lk.fresh].kind.value_counts(normalize=True)}).sort_index()
    t["linked n"] = lk.kind.value_counts()
    t["fresh n"] = lk[lk.fresh].kind.value_counts()
    g = lk.groupby("kind").picks.mean()
    t["connected picks / season"] = g
    t["$M / season"] = g * PER_PICK
    print(t.round(3).to_string())
    print(f"   linked coach-seasons {len(lk)} ({len(lk) / len(allst):.0%} of staff); fresh from college {lk.fresh.sum()}; "
          f"overall {lk.picks.mean():.3f} connected picks per linked coach-season (${lk.picks.mean() * PER_PICK:.2f}M); "
          f"fresh {lk[lk.fresh].picks.mean():.3f} (${lk[lk.fresh].picks.mean() * PER_PICK:.2f}M)")
    fr = lk[lk.fresh]
    print("   most common fresh-from-college roles: " + ", ".join(f"{r} {n}" for r, n in fr.role.value_counts().head(12).items()))
    return links


def part_b(v, dp, links):
    print("\nB. AVOIDING PLAYERS")
    teams = links.groupby(["year", "key"]).fr.agg(lambda s: set(s)).to_dict()
    have = set(zip(links.fr, links.year)) | set(zip(v.fr, v.cls))
    order = {y: g.sort_values("pick")[["pick", "fr", "pfr_player_id"]].values for y, g in dp.groupby("season")}
    s = v[v.n_teams_conn > 0].copy()
    allfr = sorted(dp.fr.unique())
    rng = np.random.default_rng(0)
    for W in (10, 32):
        def count(tset, y, pick, pid):
            o = order[y]
            w = o[(o[:, 0] < pick) & (o[:, 0] >= pick - W)]
            return sum(1 for _, f, p in w if f in tset)
        s[f"near{W}"] = [count(teams.get((y, k), set()) - {f}, y, p, pid) for y, k, f, p, pid in
                         zip(s.cls, s.key, s.fr, s.pick, s.pfr_id)]
        fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
        X = lambda col: sm.add_constant(pd.concat([s[[col, "log_nconn"]].astype(float).reset_index(drop=True),
                                                   fe.reset_index(drop=True)], axis=1))
        m = sm.GLM(s.actual.values, X(f"near{W}"), family=sm.families.Poisson(),
                   offset=np.log(s.exp_slot.clip(lower=1e-4)).values).fit(cov_type="HC1")
        sims = []
        for _ in range(300):
            s["pl"] = [count(set(rng.choice([a for a in allfr if a != f], size=len(teams.get((y, k), set()) - {f}), replace=False)),
                             y, p, None) for y, k, f, p in zip(s.cls, s.key, s.fr, s.pick)]
            mp = sm.GLM(s.actual.values, X("pl"), family=sm.families.Poisson(),
                        offset=np.log(s.exp_slot.clip(lower=1e-4)).values).fit()
            sims.append(np.exp(mp.params["pl"]))
        sims = np.array(sims)
        b = np.exp(m.params[f"near{W}"])
        print(f"   B1 near passes, {W} picks before his: mean {s[f'near{W}'].mean():.2f} per player ({(s[f'near{W}'] > 0).mean():.0%} have 1+); "
              f"value x{b:.2f} per pass (95% {np.exp(m.conf_int().iloc[1, 0]):.2f}-{np.exp(m.conf_int().iloc[1, 1]):.2f}, "
              f"p {m.pvalues.iloc[1]:.2f}); random-team placebo median x{np.median(sims):.2f} (90% {np.percentile(sims, 5):.2f}-"
              f"{np.percentile(sims, 95):.2f}), share <= real {np.mean(sims <= b):.2f}")
        for lo, hi in [(0, 0), (1, 1), (2, 99)]:
            g = s[s[f"near{W}"].between(lo, hi)]
            print(f"      {lo}{'+' if hi == 99 else ''} near passes: n {len(g)}, actual / expected {g.actual.sum() / g.exp_slot.sum():.2f}")

    # B2 accuracy
    f = v[v.n_teams_conn > 0].copy()
    f["ratio"] = np.log((f.actual + 0.01) / (f.exp_slot + 0.01))
    f["bust"] = f.actual < 0.25 * f.exp_slot
    f["star"] = f.actual > 2 * f.exp_slot
    print("   B2 accuracy (fair group):")
    for lab, g in [("connected", f[f.conn]), ("unconnected", f[~f.conn])]:
        print(f"      {lab:12s} n {len(g):4d}  bust {g.bust.mean():.0%}  star {g.star.mean():.0%}  "
              f"median log ratio {g.ratio.median():+.2f}  sd {g.ratio.std():.2f}")
    for col in ("bust", "star"):
        X = sm.add_constant(pd.concat([f[["conn"]].astype(float), np.log(f.exp_slot).rename("lexp"),
                                       pd.get_dummies(f.cls, prefix="c", drop_first=True, dtype=float)], axis=1))
        lg = sm.Logit(f[col].astype(int), X).fit(disp=0, cov_type="HC1")
        print(f"      {col}: connected odds x{np.exp(lg.params.conn):.2f} (p {lg.pvalues.conn:.2f})")
    for r, g in f.groupby(f["round"].clip(upper=3)):
        a, b = g[g.conn], g[~g.conn]
        print(f"      round {r}{'+' if r == 3 else ''}: bust {a.bust.mean():.0%} vs {b.bust.mean():.0%}, "
              f"star {a.star.mean():.0%} vs {b.star.mean():.0%} (connected n {len(a)})")

    # B3 among teammates
    f["grp_sc"] = f.cls.astype(str) + "_" + f.key.astype(str)
    gs = f.groupby("grp_sc").agg(n=("pfr_id", "size"), c=("conn", "sum"))
    keep = gs[(gs.n >= 2) & (gs.c >= 1) & (gs.c < gs.n)].index
    t = f[f.grp_sc.isin(keep)].copy()
    fe = pd.get_dummies(t.grp_sc, prefix="g", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([t[["conn"]].astype(float).reset_index(drop=True), fe.reset_index(drop=True)], axis=1))
    m = sm.GLM(t.actual.values, X, family=sm.families.Poisson(), offset=np.log(t.exp_slot.clip(lower=1e-4)).values).fit(cov_type="HC1")
    lg = sm.Logit((t.actual > t.exp_slot).astype(int).values,
                  sm.add_constant(pd.concat([t[["conn"]].astype(float).reset_index(drop=True),
                                             np.log(t.exp_slot.clip(lower=1e-4)).rename("lexp").reset_index(drop=True)], axis=1))).fit(disp=0, cov_type="HC1")
    print(f"   B3 among schoolmates ({len(keep)} school-classes, {int(t.conn.sum())} connected vs {int((~t.conn).sum())} teammates "
          f"drafted elsewhere): value x{np.exp(m.params.conn):.2f} (95% {np.exp(m.conf_int().loc['conn', 0]):.2f}-"
          f"{np.exp(m.conf_int().loc['conn', 1]):.2f}, p {m.pvalues.conn:.2f}); beat slot {(t[t.conn].actual > t[t.conn].exp_slot).mean():.0%} "
          f"vs {(t[~t.conn].actual > t[~t.conn].exp_slot).mean():.0%} (odds x{np.exp(lg.params.conn):.2f}, p {lg.pvalues.conn:.2f})")


def part_c(v, st, car):
    print("\nC. DOES THE COACH NEED TO BE ON STAFF?")
    on = st.groupby(["fr", "year"]).page.agg(set).to_dict()
    ck = car.groupby(["page", "key"]).year.agg(set).to_dict()
    def linked(pages, key, y):
        return any(any(y - 5 <= cy <= y - 1 for cy in ck.get((p, key), ())) for p in pages)
    rows = []
    for pid, f, y, k in zip(v.pfr_id, v.fr, v.cls, v.key):
        cur = on.get((f, y), set())
        prior = (on.get((f, y - 1), set()) | on.get((f, y - 2), set())) - cur
        fut = on.get((f, y + 1), set()) - cur - on.get((f, y - 1), set()) - on.get((f, y - 2), set())
        rows.append({"pfr_id": pid, "current": linked(cur, k, y), "former": linked(prior, k, y),
                     "future": linked(fut, k, y),
                     "known": (f, y - 1) in on and (f, y + 1) in on})
    L = pd.DataFrame(rows)
    d = v.merge(L, on="pfr_id")
    d = d[d.known & (d.n_teams_conn > 0)]
    fe = pd.get_dummies(d.cls, prefix="c", drop_first=True, dtype=float)
    d = pd.concat([d.reset_index(drop=True), fe.reset_index(drop=True)], axis=1)
    covs = ("log_nconn", *fe.columns)
    none = ~d.current & ~d.former & ~d.future
    print(f"   fair-group picks with staff data for Y-1..Y+1: {len(d)}; current {d.current.sum()}, former only "
          f"{(d.former & ~d.current).sum()}, future only {(d.future & ~d.current & ~d.former).sum()}")
    for lab, flag in [("current (on staff at the draft)", d.current),
                      ("former only (left in the 2 prior seasons)", d.former & ~d.current),
                      ("future only (joined after the draft)", d.future & ~d.current & ~d.former)]:
        g = d[flag | none].assign(f=flag[flag | none])
        compare(g, "f", "exp_slot", covs=covs, label=lab, min_n=5)


def main():
    v, st, car, dp = load()
    links = part_a(v, st, car)
    part_b(v, dp, links)
    part_c(v, st, car)


if __name__ == "__main__":
    main()
