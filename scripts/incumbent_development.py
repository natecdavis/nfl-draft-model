"""Does age / years in the league help project incumbents' snaps (and quality) next season?

The simulator's room check counts draft-day incumbents at last season's snap share (se), weighted by quality
tier (good 1.0, middling 0.6, weak 0). Young players may grow into more snaps and better play; older players
may fade, get cut or retire after the draft. Incumbents as in scripts/displacement.py (drafts 2014-24).
Years in the league: seasons since entry (draft year, else first season with snaps; entry before 2012 counts
as 2012, so long-tenured players are censored, which only matters inside the 7+ bin's lower edge).

A  Incumbent side, out of sample. Target: his se for the same team this season (0 if gone or not playing).
   Each season Y in 2017-24 is predicted from coefficients fit on earlier seasons only.
     base   now ~ group x se + tier x se                         (what the room check knows)
     +age   base + age band x se + age band                       (bands <=24, 25-27, 28-30, 31-33, 34+)
     +exp   base + years-in-league band x se + band               (1, 2, 3, 4-6, 7+)
     +both  base + both
   RMSE per player and per team-group total (the room check's unit).
B  Quality: next season's quality percentile ~ this season's, by age / years band (players with q both years).
C  Rookie side (as displacement.py A): rookie year-1 se ~ draft controls + incumbent se split by tier x
   career stage (young: years 1-3; prime; old: age 31+). Do young good incumbents hold rookies back more?

Usage: python scripts/incumbent_development.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["src", "scripts"]
import displacement as dz  # noqa: E402
import roster_need as rn  # noqa: E402
from outlook import OUT  # noqa: E402

AGE_BINS, AGE_LAB = [0, 24.99, 27.99, 30.99, 33.99, 99], ["a<=24", "a25-27", "a28-30", "a31-33", "a34+"]
EXP_BINS, EXP_LAB = [0, 1, 2, 3, 6, 99], ["y1", "y2", "y3", "y4-6", "y7+"]


def build():
    s, ps, games, dp = rn.load()
    A = dz.attributes()
    I = dz.types(pd.concat([dz.incumbents(ps, y, A) for y in dz.YEARS], ignore_index=True))
    me = ps.set_index(["season", "team", "pfr_player_id"]).se
    I["now"] = [float(me.get((y, tm, p), 0.0)) for y, tm, p in zip(I.year, I.team, I.pfr_player_id)]
    first = ps.groupby("pfr_player_id").season.min()
    dy = dict(zip(dp.pfr_player_id, dp.season))
    entry = np.array([dy.get(p, first.get(p, np.nan)) for p in I.pfr_player_id], float)
    I["exp"] = I.year - entry  # seasons played before this one (1 = entering year 2)
    I["age_"] = I.age.fillna(I.age.median())
    I["ab"] = pd.cut(I.age_, AGE_BINS, labels=AGE_LAB).astype(str)
    I["eb"] = pd.cut(I.exp.fillna(7), EXP_BINS, labels=EXP_LAB).astype(str)
    p2g, p2pff, born, drafted, q, cap = A
    I["q_next"] = [float(np.ravel(q.get((p2pff.get(p), y), np.nan))[-1]) for p, y in zip(I.pfr_player_id, I.year)]
    I["cl"] = I.year.astype(str) + I.team
    I = add_availability(I, s, games)
    return I, ps, dp


def add_availability(I, s, games):
    """Last season's availability (any team): gp_frac = games with snaps at his role / his team's games; rate =
    median snap share in those games (robust to the game he got hurt in); miss = rate x (1 - gp_frac), the share
    he'd have added playing every game at that rate (injuries, also benchings / healthy scratches); se2 = his
    starter-equivalents two seasons ago (any team)."""
    g = s[s.pct > 0].groupby(["season", "pfr_player_id"])
    A = pd.DataFrame({"gp": g.game_id.nunique(), "rate": g.pct.median(), "pct": g.pct.sum()})
    G = games.groupby(level="season").max()
    A["gp_frac"] = (A.gp / A.index.get_level_values("season").map(G)).clip(upper=1)
    A["se_all"] = A.pct / A.index.get_level_values("season").map(G)
    key = list(zip(I.year - 1, I.pfr_player_id))
    I["gp_frac"] = [A.gp_frac.get(k, 0.0) for k in key]
    I["rate"] = [A.rate.get(k, 0.0) for k in key]
    I["miss"] = I.rate * (1 - I.gp_frac)
    I["se2"] = [A.se_all.get((y - 2, p), 0.0) for y, p in zip(I.year, I.pfr_player_id)]
    return I


FORMS = {
    "base": "now ~ C(grp):se + C(qt):se + C(qt) - 1",
    "+age": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) - 1",
    "+exp": "now ~ C(grp):se + C(qt):se + C(qt) + C(eb):se + C(eb) - 1",
    "+both": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) + C(eb):se + C(eb) - 1",
    "+miss": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) + C(eb):se + C(eb) + miss + C(ab):miss - 1",
    "+se2": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) + C(eb):se + C(eb) + se2 - 1",
    "+miss+se2": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) + C(eb):se + C(eb) + miss + C(ab):miss + se2 - 1",
    # adopted for the simulator's room check (Nov 16 2026): missed time by quality tier and age, plus two seasons ago
    "+inj": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) + C(eb):se + C(eb) + miss + C(ab):miss + C(qt):miss"
            " + se2 - 1",
    # room check v2 (Nov 16 2026, scripts/room_crowding.py): + players with no snaps last season (out1 = missed /
    # red12 / red3; displacement.missed), by kind and their se two seasons ago
    "+out": "now ~ C(grp):se + C(qt):se + C(qt) + C(ab):se + C(ab) + C(eb):se + C(eb) + miss + C(ab):miss + C(qt):miss"
            " + se2 + C(out1, Treatment('no')) + C(out1, Treatment('no')):se2 - 1",
}


def part_a(I):
    P = []
    for y in range(2017, 2025):
        tr, te = I[I.year < y], I[I.year == y].copy()
        for k, f in ((k, f) for k, f in FORMS.items() if k != "+out"):  # +out: room_crowding.py
            te[k] = smf.ols(f, tr).fit().predict(te).clip(lower=0)
        P.append(te)
    P = pd.concat(P)
    # the room check's current rule (good 1, middling 0.6, weak/unknown 0) is a "held back" weight, not a forecast;
    # shown for reference as a forecast of snaps
    P["rule"] = P.se * P.qt.map({"high": 1.0, "mid": 0.6}).fillna(0.0)
    P["last"] = P.se
    cands = ["last", "rule"] + [k for k in FORMS if k != "+out"]
    pl = {c: np.sqrt(((P.now - P[c]) ** 2).mean()) for c in cands}
    G = P.groupby(["year", "team", "grp"])[["now"] + cands].sum()
    gr = {c: np.sqrt(((G.now - G[c]) ** 2).mean()) for c in cands}
    print(f"A. Out-of-sample 2017-24: {len(P)} incumbents, {len(G)} team-groups")
    print(pd.DataFrame({"RMSE player": pl, "RMSE team-group": gr}).round(4).to_string())
    # paired bootstrap by team-draft for +both minus base (team-group level)
    rng = np.random.default_rng(0)
    d = ((G.now - G["+both"]) ** 2 - (G.now - G["base"]) ** 2).groupby(level=[0, 1]).sum()
    n = (G.now * 0 + 1).groupby(level=[0, 1]).sum()
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(d), len(d))
        bs.append(d.to_numpy()[k].sum() / n.to_numpy()[k].sum())
    print(f"  +both minus base, team-group MSE {d.sum() / n.sum():+.4f} (95% {np.percentile(bs, 2.5):+.4f} to {np.percentile(bs, 97.5):+.4f})")
    print("  by group, team-group RMSE base -> +both:",
          {g: f"{np.sqrt(((x.now - x.base) ** 2).mean()):.3f}->{np.sqrt(((x.now - x['+both']) ** 2).mean()):.3f}"
           for g, x in G.reset_index().groupby("grp")})
    # what age / experience do, fit on all seasons: ratio of next season's se to last season's, starters (se >= 0.5)
    S = I[I.se >= 0.5]
    print("\n  Starters last season (se >= 0.5): mean se this season / last season, and share gone to 0")
    for col, lab in [("ab", AGE_LAB), ("eb", EXP_LAB)]:
        t = S.groupby(col).apply(lambda x: pd.Series({"n": len(x), "kept": x.now.sum() / x.se.sum(),
                                                      "zero": (x.now == 0).mean()}), include_groups=False).reindex(lab)
        print(t.round(2).T.to_string())
    S2 = I[(I.se > 0.05) & (I.se < 0.5)]
    print("\n  Part-timers (0.05 < se < 0.5): this season / last season")
    for col, lab in [("ab", AGE_LAB), ("eb", EXP_LAB)]:
        t = S2.groupby(col).apply(lambda x: pd.Series({"n": len(x), "ratio": x.now.sum() / x.se.sum()}),
                                  include_groups=False).reindex(lab)
        print(t.round(2).T.to_string())
    return P


def part_b(I):
    Q = I[I.q.notna() & I.q_next.notna()]
    m = smf.ols("q_next ~ q + C(ab) + C(eb)", Q).fit(cov_type="cluster", cov_kwds={"groups": Q.pfr_player_id})
    print(f"\nB. Next season's quality percentile ~ this season's + age band + years band (n={len(Q)}; reference a<=24, y1)")
    print("  " + "  ".join(f"{k.replace('C(ab)[T.', '').replace('C(eb)[T.', '').rstrip(']')} {v:+.3f} ({m.bse[k]:.3f})"
                           for k, v in m.params.items() if k != "Intercept"))
    # same quality tercile changes: tier transitions by career stage for good / middling
    Q = Q.assign(stage=np.where(Q.exp <= 3, "young(y1-3)", np.where(Q.age_ >= 31, "old(31+)", "prime")))
    Q["tn"] = np.select([Q.q_next >= 2 / 3, Q.q_next >= 1 / 3], ["high", "mid"], "low")
    print("  tier next season by tier now and stage (row shares):")
    print(pd.crosstab([Q.qt, Q.stage], Q.tn, normalize="index").round(2).to_string())


def part_c(I, ps, dp):
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[R.year.isin(dz.YEARS) & R.grp.isin(rn.GROUPS)].copy()
    pid = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    R["pfr"] = [pid.get((y, k)) for y, k in zip(R.year, R.pick)]
    me = ps.set_index(["season", "team", "pfr_player_id"]).se
    R["s1"] = [float(me.get((y, tm, p), 0.0)) for y, tm, p in zip(R.year, R.team, R.pfr)]
    R["lp"], R["lr"] = np.log(R.pick + 10), np.log(R.board_rank.fillna(330) + 10)
    R["off"] = R.board_rank.isna().astype(int)
    R["cl"] = R.year.astype(str) + R.team
    R = R.sort_values("pick")
    R["earlier"] = R.groupby(["year", "team", "grp"]).cumcount()
    I = I.assign(stage=np.where(I.exp <= 3, "young", np.where(I.age_ >= 31, "old", "prime")),
                 tier=I.qt.map({"high": "good", "mid": "mid"}).fillna("weak"))
    I["cell"] = I.tier + "_" + I.stage
    X = I.pivot_table(index=["year", "team", "grp"], columns="cell", values="se", aggfunc="sum", fill_value=0).reset_index()
    cols = [c for c in X.columns if c not in ("year", "team", "grp")]
    D = R.merge(X, on=["year", "team", "grp"], how="left").fillna({c: 0 for c in cols})
    base = "C(grp):lp + C(grp):lr + off + C(grp) + earlier"
    m = smf.ols(f"s1 ~ {base} + " + " + ".join(cols), D).fit(cov_type="cluster", cov_kwds={"groups": D.cl})
    print(f"\nC. Rookie year-1 se per incumbent se, by tier x stage (n={len(D)} rookies; young = years 1-3, old = 31+)")
    for c in sorted(cols):
        print(f"  {c:12s} {m.params[c]:+.3f} ({m.bse[c]:.3f})   mean se per room {X[c].mean():.2f}")
    for t in ["good", "mid"]:
        a, b = f"{t}_young", f"{t}_old"
        if a in cols and b in cols:
            r = m.t_test(f"{a} - {b} = 0")
            print(f"  {t}: young minus old {float(np.ravel(r.effect)[0]):+.3f} (p {float(r.pvalue):.2f})")


def main():
    I, ps, dp = build()
    pd.set_option("display.width", 200)
    part_a(I)
    part_b(I)
    part_c(I, ps, dp)


if __name__ == "__main__":
    main()
