"""Per-unit team check for quality-over-replacement (follow-up to scripts/replacement_level.py).

For each team-season 2014-2025 and position group g (cfdarko pricing groups):
  base_g = sum over the team's players of price_g(share, Q_REF)       (playing-time part)
  qual_g = sum of price_g(share, q) - base_g                          (quality part)
with q either the same season's quality (leaks: grades partly reflect team results) or the player's
PRIOR-season quality (any team; neutral if he has none). All in % of cap / 1.10 (this project's units).

Questions: does each unit's playing-time part buy ~0 points? Is the quality slope the same across units
(if so, the pay curves' quality pricing is consistent and the reshuffle stands; if DB/LB/RB quality buys
more per 1% than OL/WR, their flat curves understate them)?
Offense units vs offensive rating, defense units vs defensive rating (opponent-adjusted points).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_level as rl  # noqa: E402  (sets up cfdarko import)
from synergy_test import ratings, RELOC  # noqa: E402

rv = rl.rv
Q_REF, UNITS = 0.1, 1.10
OFF = ["QB", "RB", "WR", "TE", "OL"]
DEF = ["DL", "EDGE", "LB", "DB"]


def capture():
    rv.quality_table_pff, rv.pay_curves_wide = rl.grab_quality, rl.grab_curves
    try:
        rv.build_label(out_path=ROOT / "data/derived/_unused.parquet")
    except rl.Stop:
        pass
    return rl.CAP


def player_seasons(C):
    sc = pd.read_parquet(ROOT / "data/raw/snap_counts.parquet")
    sc = sc[(sc.game_type == "REG") & sc.season.between(2013, 2025)].copy()
    sc["team"] = sc.team.replace(RELOC)
    sc["G"] = np.where(sc.season >= 2021, 17, 16)
    ps = (sc.groupby(["pfr_player_id", "season", "team"])
          .agg(off=("offense_pct", "sum"), dfn=("defense_pct", "sum"), G=("G", "first")).reset_index())
    ps["share"] = (ps[["off", "dfn"]].max(axis=1) / ps.G).clip(0, 1)
    ps["gsis"] = ps.pfr_player_id.map(C["pfr2gsis"])
    ps["g"] = ps.gsis.map(C["pos"])
    ps = ps[ps.g.isin(C["curves"]) & (ps.share > 0)].copy()
    q, qv, neutral = C["q"], C["qv"], C["neutral"]
    ps["q_now"] = [qv(a, g, y) for a, g, y in zip(ps.gsis, ps.g, ps.season)]
    # prior season quality; players with none (rookies, prior season unmeasured) -> neutral for the group
    ps["has_prior"] = [(a, y - 1) in q for a, y in zip(ps.gsis, ps.season)]
    ps["q_prior"] = [q[(a, y - 1)] if h else neutral.get((g, y), neutral[(g, None)])
                     for a, g, y, h in zip(ps.gsis, ps.g, ps.season, ps.has_prior)]
    cur = C["curves"]
    ps["base"] = [cur[g](s, Q_REF) for g, s in zip(ps.g, ps.share)]
    ps["v_now"] = [cur[g](s, qq) for g, s, qq in zip(ps.g, ps.share, ps.q_now)]
    ps["v_prior"] = [cur[g](s, qq) for g, s, qq in zip(ps.g, ps.share, ps.q_prior)]
    return ps


def table(ps, which):
    ps = ps.assign(qual=ps[f"v_{which}"] - ps.base)
    W = ps.pivot_table(index=["season", "team"], columns="g", values=["base", "qual"], aggfunc="sum",
                       fill_value=0) * 100 / UNITS
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index().merge(ratings(), on=["season", "team"])
    cols = [c for c in W.columns if c not in ("season", "team")]
    W[cols] = W[cols] - W.groupby("season")[cols].transform("mean")
    return W


def fit(W, y, groups, parts=("base", "qual")):
    terms = [f"{p}_{g}" for g in groups for p in parts]
    m = smf.ols(f"{y} ~ " + " + ".join(terms), W).fit(cov_type="cluster", cov_kwds={"groups": W.team})
    return m


def main():
    C = capture()
    ps = player_seasons(C)
    ps = ps[ps.season >= 2014]
    print(f"\nplayer-team-seasons {len(ps)}; with prior-season quality {ps.has_prior.mean():.0%} "
          f"(snap-weighted {np.average(ps.has_prior, weights=ps.share):.0%})")
    rows = []
    for which in ["now", "prior"]:
        W = table(ps, which)
        for y, groups in [("off", OFF), ("dfn", DEF)]:
            m = fit(W, y, groups)
            ci = m.conf_int()
            for g in groups:
                r = {"quality": which, "unit": g}
                for p in ("base", "qual"):
                    t = f"{p}_{g}"
                    r[p] = m.params[t]
                    r[f"{p}_lo"], r[f"{p}_hi"] = ci.loc[t, 0], ci.loc[t, 1]
                    r[f"{p}_sd"] = W[t].std()
                rows.append(r)
            # pooled: one base slope, one quality slope for the side
            W["B"] = W[[f"base_{g}" for g in groups]].sum(axis=1)
            W["Q"] = W[[f"qual_{g}" for g in groups]].sum(axis=1)
            mp = smf.ols(f"{y} ~ B + Q", W).fit(cov_type="cluster", cov_kwds={"groups": W.team})
            # does letting quality slopes differ by unit help? F-test of equal quality slopes
            eq = " = ".join(f"qual_{g}" for g in groups)
            ft = m.f_test([f"qual_{groups[0]} = qual_{g}" for g in groups[1:]])
            print(f"[{which}, {y}] pooled base {mp.params.B:+.3f} [{mp.conf_int().loc['B', 0]:+.3f},"
                  f"{mp.conf_int().loc['B', 1]:+.3f}], quality {mp.params.Q:+.3f} [{mp.conf_int().loc['Q', 0]:+.3f},"
                  f"{mp.conf_int().loc['Q', 1]:+.3f}]  R2 {mp.rsquared:.2f} vs by-unit {m.rsquared:.2f};"
                  f" equal quality slopes p = {float(ft.pvalue):.3f}")
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nBy unit, pts/game per 1% of cap (95%), sd of each part across teams in % of cap:")
    for which in ["now", "prior"]:
        print(f"\n  quality = {which}")
        for r in T[T.quality == which].itertuples():
            print(f"   {r.unit:5s} playing time {r.base:+.3f} [{r.base_lo:+.3f},{r.base_hi:+.3f}] (sd {r.base_sd:.2f})   "
                  f"quality {r.qual:+.3f} [{r.qual_lo:+.3f},{r.qual_hi:+.3f}] (sd {r.qual_sd:.2f})")
    T.to_csv(ROOT / "data/derived/replacement_units.csv", index=False)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def drafted_points():
    """Points per drafted player 2014-22 (4 seasons) under: current rule; pooled quality-over-replacement
    (0.48, replacement_level.py); unit-specific same-season slopes; unit-specific prior-quality slopes
    (playing-time part + quality part)."""
    C = rl.CAP if "curves" in rl.CAP else capture()
    cur, pos = C["curves"], C["pos"]
    T = pd.read_csv(ROOT / "data/derived/replacement_units.csv").set_index(["quality", "unit"])
    s = rv.load_snaps()
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["gfrac"] = (s["n_games"] / s["season"].map(rv.reg_season_games)).clip(upper=1.0)
    s = s.sort_values("share").groupby(["pfr_player_id", "season"])[["share", "gfrac"]].last()
    L = pd.read_parquet(ROOT / "data/derived/labels_pickadj_narrow.parquet")
    L = L[L.drafted & L.draft_year.between(2014, 2022)].dropna(subset=["value_y14"])
    out = []
    for r in L.itertuples():
        g = pos.get(r.gsis) if isinstance(r.gsis, str) else None
        if g not in cur:
            continue
        rec = {"g": g, "round": r.round, "old": 0.0, "pooled": 0.0, "now": 0.0, "prior": 0.0}
        for k in range(4):
            v = getattr(r, f"value_y{k + 1}") / UNITS
            if pd.isna(v):
                continue
            key = (r.pfr_id, r.draft_year + k)
            a, f = (s.loc[key, "share"], s.loc[key, "gfrac"]) if key in s.index else (0.0, 0.0)
            base = cur[g](a, Q_REF) / UNITS * f if a > 0 else 0.0
            qual = v - base
            rec["old"] += (v - 0.005 * a * f) * (0.34 if g == "QB" else 0.19)
            rec["pooled"] += qual * 0.48
            rec["now"] += base * T.loc[("now", g), "base"] + qual * T.loc[("now", g), "qual"]
            rec["prior"] += base * T.loc[("prior", g), "base"] + qual * T.loc[("prior", g), "qual"]
        out.append(rec)
    O = pd.DataFrame(out)
    for c in ["old", "pooled", "now", "prior"]:
        O[c] *= 100
    O["rd"] = np.where(O["round"] == 1, "R1", np.where(O["round"] <= 3, "R2-3", "R4-7"))
    print("\nPoints of team strength per drafted player 2014-22, 4 seasons (current / pooled quality-over-"
          "replacement / unit slopes same-season / unit slopes prior-quality):")
    print(O.groupby(["rd", "g"])[["old", "pooled", "now", "prior"]].mean().unstack(0).round(2).to_string())
    print(O.groupby("rd")[["old", "pooled", "now", "prior"]].mean().round(2).to_string())
    G = O.groupby("g")[["old", "pooled", "now", "prior"]].mean()
    print("\nRatio to current rule, all rounds:")
    print(G.div(G.old, axis=0).round(2).to_string())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "points":
    drafted_points()
