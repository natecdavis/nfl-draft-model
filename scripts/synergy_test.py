"""Position synergies: do units complement each other, and does draft order matter?

Folk beliefs tested ("build the trenches first", "get the QB weapons", "pass rush and
coverage work together", "the OL is only as good as its worst player"):

A. Team production (2006-2025, 640 team-seasons). Offensive / defensive points ratings
   (opponent-adjusted from schedules) on PFF unit grades (snap-weighted, z-scored within
   season). Additive vs pairwise interactions. Positive interaction = complements (each unit
   is worth more when the other is good: build both / build the base first); negative =
   substitutes (diminishing returns: spread resources). Also within-team (team FE) and
   out-of-sample (fit 2006-2017, predict 2018-2025).
B. Weakest link: OL rating on mean vs worst starter vs best starter.
C. Drafted players (drafts 2014-2022): does a rookie's 4-year value depend on the
   complementary unit he joins (WR/TE/RB with last season's QB and OL grades, QB with OL
   and receivers, ED/IDL with coverage, CB/S with pass rush)? Controls as in roster_need.
D. Sequencing: teams' early-round draft mix (trenches vs skill/secondary) over drafts
   t-2..t vs strength change, controlling for draft capital and start strength. Descriptive.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PFF = Path.home() / "cfdarko/data/raw/pff_nfl"
RAW, OUT = Path("data/raw"), Path("data/derived")
YEARS = range(2006, 2026)
TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "STL": "LA",
        "SD": "LAC", "OAK": "LV"}
RELOC = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def wmean(d, g, w):
    d = d[d[w] > 0]
    return (d[g] * d[w]).groupby([d.season, d.team]).sum() / d[w].groupby([d.season, d.team]).sum()


def units():
    rows = []
    for y in YEARS:
        rd = lambda f: pd.read_csv(PFF / str(y) / f"{f}.csv").assign(
            season=y, team=lambda d: d.team_name.replace(TEAM))
        pas, blk, rec, rus, dfn = (rd(f) for f in ["passing_summary", "offense_blocking", "receiving_summary",
                                                  "rushing_summary", "defense_summary"])
        u = {}
        q = pas[pas.position == "QB"]
        u["qb"] = wmean(q, "grades_pass", "dropbacks")
        ol = blk[blk.position.isin(["T", "G", "C"])]
        u["ol"] = wmean(ol, "grades_offense", "snap_counts_offense")
        u["olpb"] = wmean(ol, "grades_pass_block", "snap_counts_pass_block")
        u["olrb"] = wmean(ol, "grades_run_block", "snap_counts_run_block")
        top5 = ol.sort_values("snap_counts_offense", ascending=False).groupby(["season", "team"]).head(5)
        g5 = top5.groupby(["season", "team"]).grades_offense
        u["ol_min"], u["ol_max"] = g5.min(), g5.max()
        r = rec[rec.position.isin(["WR", "TE"])]
        u["rec"] = wmean(r, "grades_offense", "routes")
        rb = rus[rus.position == "HB"]
        u["rb"] = wmean(rb, "grades_offense", "attempts")
        u["prush"] = wmean(dfn[dfn.position.isin(["ED", "DI"])], "grades_pass_rush_defense", "snap_counts_pass_rush")
        u["rdef"] = wmean(dfn[dfn.position.isin(["ED", "DI", "LB"])], "grades_run_defense", "snap_counts_run_defense")
        u["cov"] = wmean(dfn[dfn.position.isin(["CB", "S"])], "grades_coverage_defense", "snap_counts_coverage")
        u["lb"] = wmean(dfn[dfn.position == "LB"], "grades_defense", "snap_counts_defense")
        rows.append(pd.DataFrame(u))
    U = pd.concat(rows).reset_index()
    return U


def ratings():
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.home_score.notna() & s.season.isin(YEARS)].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(RELOC)
    out = []
    for y, g in s.groupby("season"):
        teams = sorted(set(g.home_team) | set(g.away_team))
        ix = {t: i for i, t in enumerate(teams)}
        n, k = len(g), len(teams)
        # rows: points scored by home, by away. cols: off(k), def(k), home, const
        X = np.zeros((2 * n, 2 * k + 2))
        a = np.arange(n)
        X[a, g.home_team.map(ix)] = 1; X[a, k + g.away_team.map(ix)] = 1
        X[n + a, g.away_team.map(ix)] = 1; X[n + a, k + g.home_team.map(ix)] = 1
        X[a, 2 * k] = (g.location != "Neutral").astype(float); X[:, -1] = 1
        yv = np.r_[g.home_score, g.away_score].astype(float)
        C = np.zeros((2, 2 * k + 2)); C[0, :k] = 100; C[1, k:2 * k] = 100
        b, *_ = np.linalg.lstsq(np.vstack([X, C]), np.r_[yv, 0, 0], rcond=None)
        out.append(pd.DataFrame({"season": y, "team": teams, "off": b[:k], "dfn": -b[k:2 * k]}))
    R = pd.concat(out)
    R["net"] = R.off + R.dfn  # dfn: higher = better (fewer points allowed)
    return R


def fit(f, d, fe=False):
    if fe:
        f += " + C(team)"
    return smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.team})


def show(m, terms):
    ci = m.conf_int()
    return "  ".join(f"{t} {m.params[t]:+.2f} [{ci.loc[t, 0]:+.2f},{ci.loc[t, 1]:+.2f}]" for t in terms)


def oos(f_add, f_int, d):
    tr, te = d[d.season <= 2017], d[d.season >= 2018]
    y = f_add.split("~")[0].strip()
    e = {}
    for nm, f in [("additive", f_add), ("interactions", f_int)]:
        m = smf.ols(f, tr).fit()
        e[nm] = np.sqrt(((te[y] - m.predict(te)) ** 2).mean())
    return e


def part_a(D):
    print("=== A. Team production: unit grades (z within season) -> points ratings, 2006-2025 ===")
    print(f"team-seasons: {len(D)};  rating sd: off {D.off.std():.2f}, def {D.dfn.std():.2f} pts/game\n")
    specs = [
        ("off", ["qb", "ol", "rec", "rb"], ["qb:ol", "qb:rec", "ol:rb", "ol:rec"]),
        ("off", ["qb", "olpb", "olrb", "rec", "rb"], ["qb:olpb", "olrb:rb"]),
        ("dfn", ["prush", "cov", "rdef"], ["prush:cov", "rdef:cov"]),
    ]
    res = {}
    for y, main, inter in specs:
        fa = f"{y} ~ " + " + ".join(main)
        fi = fa + " + " + " + ".join(inter)
        for fe in [False, True]:
            ma, mi = fit(fa, D, fe), fit(fi, D, fe)
            tag = "team FE" if fe else "pooled "
            print(f"[{y}, {tag}] additive R2 {ma.rsquared:.3f} -> with interactions {mi.rsquared:.3f}")
            print("   main:", show(ma, main))
            print("   inter:", show(mi, inter))
            res[(y, tuple(main), fe)] = mi
        print("   out-of-sample RMSE 2018-25:", {k: round(v, 3) for k, v in oos(fa, fi, D).items()})
        print()
    # nonlinearity check: is the QB grade's payoff convex (stars matter more) / concave?
    m = fit("off ~ qb + I(qb**2) + ol + I(ol**2) + rec + rb", D)
    print("Curvature (offense):", show(m, ["I(qb ** 2)", "I(ol ** 2)"]))
    m = fit("dfn ~ prush + I(prush**2) + cov + I(cov**2) + rdef", D)
    print("Curvature (defense):", show(m, ["I(prush ** 2)", "I(cov ** 2)"]))
    # Does the QB "elevate" the OL / receivers? split by QB tercile
    D["qbt"] = pd.qcut(D.qb, 3, labels=["low", "mid", "high"])
    print("\nSlope of offense on OL and receivers, by QB tercile:")
    for t, g in D.groupby("qbt", observed=True):
        m = smf.ols("off ~ ol + rec + rb + qb", g).fit(cov_type="HC1")
        print(f"   QB {t:4s} (n={len(g)}): " + show(m, ["ol", "rec"]))
    return res


def part_b(D):
    print("\n=== B. Weakest link: OL ===")
    d = D.dropna(subset=["ol_min", "ol_max"])
    for f in ["off ~ ol + qb + rec + rb", "off ~ ol_min + ol_max + ol + qb + rec + rb",
              "off ~ ol_min + ol_max + qb + rec + rb"]:
        m = fit(f, d)
        terms = [t for t in ["ol", "ol_min", "ol_max"] if t in m.params]
        print(f"   {f}: R2 {m.rsquared:.3f}  " + show(m, terms))
    # sacks-allowed style: pass-block min vs mean against QB pressure? keep simple


def part_c(Uraw):
    print("\n=== C. Rookies: does the unit he joins change his 4-year value? (drafts 2014-2022) ===")
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[R.year.between(2014, 2022)].copy()
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pid = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    R["pfr"] = [pid.get((y, k)) for y, k in zip(R.year, R.pick)]
    L = pd.read_parquet(OUT / "labels_pickadj_narrow.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    R["v14"] = R.pfr.map(L.set_index("pfr_id").value_y14) * 100
    R = R.dropna(subset=["v14"])
    # complementary unit = last season's grade on the drafting team (z within season)
    U = Uraw.copy()
    U["season"] += 1  # known at draft time
    R = R.merge(U.rename(columns={"season": "year"}), on=["year", "team"], how="left")
    R["lp"] = np.log(R.pick + 10)
    R["off_b"] = R.board_rank.isna().astype(int)
    R["lr"] = np.log(R.board_rank.fillna(330) + 10)
    R["cl"] = R.year.astype(str) + R.team
    ctrl = "lp + lr + off_b"
    tests = [("WR/TE", ["WR", "TE"], ["qb", "ol"]), ("RB", ["RB"], ["ol", "qb"]),
             ("QB", ["QB"], ["ol", "rec"]), ("OL", ["T", "G", "C"], ["qb", "rec"]),
             ("ED/IDL", ["ED", "IDL"], ["cov"]), ("CB/S", ["CB", "S"], ["prush"])]
    print(f"   rookie value y1-4, % of cap; mean {R.v14.mean():.2f}. Effect per +1 SD of the unit last season")
    rows = []
    for nm, grps, us in tests:
        d = R[R.grp.isin(grps)].dropna(subset=us)
        m = smf.ols(f"v14 ~ {' + '.join(us)} + C(grp) + C(grp):({ctrl})", d).fit(
            cov_type="cluster", cov_kwds={"groups": d.cl})
        print(f"   {nm:7s} n={len(d):4d} mean {d.v14.mean():5.2f}: " + show(m, us))
        for u in us:
            rows.append({"rookies": nm, "unit": u, "n": len(d), "effect": m.params[u],
                         "lo": m.conf_int().loc[u, 0], "hi": m.conf_int().loc[u, 1], "mean": d.v14.mean()})
    return pd.DataFrame(rows)


def part_d(D):
    print("\n=== D. Draft order: trench-heavy vs skill-heavy early picks (drafts 2008-2022) ===")
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dp = dp[dp.season.between(2006, 2022) & (dp["round"] <= 2)].copy()
    dp["team"] = dp.team.replace(RELOC | {"SDG": "LAC", "STL": "LA", "OAK": "LV", "SFO": "SF", "GNB": "GB",
                                          "KAN": "KC", "LAR": "LA", "LVR": "LV", "NWE": "NE", "NOR": "NO", "TAM": "TB", "RAM": "LA"})
    pos = dp.position.str.upper()
    trench = pos.isin(["T", "G", "C", "OL", "OT", "OG", "DE", "DT", "NT", "DL", "OLB"])
    skill = pos.isin(["WR", "TE", "RB", "CB", "S", "SAF", "DB", "FS", "SS"])
    w = 1 / np.log(dp.pick + 5)  # early picks weigh more
    dp["tr"], dp["sk"], dp["cap"] = w * trench, w * skill, w
    g = dp.groupby(["season", "team"])[["tr", "sk", "cap"]].sum().reset_index()
    rows = []
    for (t, grp) in g.groupby("team"):
        grp = grp.set_index("season")[["tr", "sk", "cap"]].reindex(range(2006, 2023), fill_value=0)
        roll = grp.rolling(3).sum().dropna()
        roll["team"] = t
        rows.append(roll.reset_index())
    S = pd.concat(rows)
    S["trench_share"] = S.tr / (S.tr + S.sk)
    N = D.set_index(["season", "team"]).net
    S["start"] = [N.get((y - 3, t), np.nan) for y, t in zip(S.season, S.team)]  # before the 3 drafts
    S["after2"] = [N.get((y + 2, t), np.nan) for y, t in zip(S.season, S.team)]  # 2 seasons after last draft
    S = S.dropna(subset=["trench_share", "start", "after2"])
    S = S[S.season.isin([2010, 2013, 2016, 2019, 2022 - 0])]  # non-overlapping windows
    m = smf.ols("after2 ~ trench_share + start + cap + C(season)", S).fit(cov_type="cluster",
                                                                           cov_kwds={"groups": S.team})
    print(f"   n={len(S)} team-windows; trench share of early-pick capital mean {S.trench_share.mean():.2f} sd {S.trench_share.std():.2f}")
    print("   net rating 2 yrs after window ~ trench share + start + capital: " + show(m, ["trench_share", "start", "cap"]))
    print(f"   -> going from 25% to 75% trench share changes net rating by {0.5 * m.params.trench_share:+.2f} pts/game")
    return S


def main():
    U = units()
    unit_cols = [c for c in U.columns if c not in ("season", "team")]
    Z = U.copy()
    Z[unit_cols] = U.groupby("season")[unit_cols].transform(lambda x: (x - x.mean()) / x.std())
    D = Z.merge(ratings(), on=["season", "team"])
    assert len(D) >= 630, len(D)
    print("Unit correlations (z):")
    print(D[["qb", "ol", "rec", "rb", "prush", "cov", "rdef"]].corr().round(2).to_string(), "\n")
    part_a(D)
    part_b(D)
    C = part_c(Z)
    part_d(D)
    D.to_parquet(OUT / "synergy_units.parquet")
    C.to_csv(OUT / "synergy_rookies.csv", index=False)


if __name__ == "__main__":
    main()
