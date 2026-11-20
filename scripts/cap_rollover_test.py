"""Does cap a team leaves unspent turn into team strength later? (decides the cap price: scripts/cap_shadow_picks.py)

The shadow-price version charges a rebuilding team's rookie contracts at what that cap would buy later, assuming
unused cap rolls over and is then spent at the free-agent rate (~0.12 points of strength per 1% of cap,
scripts/vet_value.py). The current model's 0.10 rests on scripts/qb_savings.py (QB savings turned into ~0.05).
Specified before running:
  unused_t   = 100% - cap used by the team's players in season t (OTC season_history cap_percent, summed by team;
               excludes dead money and practice squad, so it overstates true space, the same way for every team)
  outcomes   team rating (simulate.season_ratings) in seasons t+1, t+2, t+3, and their sum (seasons t = 2014-2022)
  controls   rating t and t-1; roster curve value in t (veterans and rookie-deal players separately: a cheap young
             core both uses less cap and improves); season fixed effects. Variant: + franchise fixed effects.
  reading    sum of the t+1..t+3 coefficients ~ 0.12 per 1% of cap supports the shadow price (rolled cap is spent
             at the free-agent rate); ~0.05 or less supports the current charge.
  mechanism  cap used in t+1 ~ unused_t (same controls): does unspent cap get spent the next season (+1 = all of it)?
Run: .venv/bin/python scripts/cap_rollover_test.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import age_pricing as ap  # noqa: E402
import roster_need as rn  # noqa: E402

RAW = ap.RAW


def cap_used():
    c = pd.read_parquet(RAW / "contracts.parquet")
    c = c[c.season_history.notna()]
    c = c.assign(n=c.season_history.map(len)).sort_values("n").drop_duplicates("otc_id", keep="last")
    nick = dict(zip(*pd.read_parquet(RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS"})
    rows = [(int(x["year"]), rn.team(nick.get(x["team"], x["team"])), x["cap_percent"] or 0.0)
            for sh in c.season_history for x in sh if str(x.get("year", "")).isdigit()]
    d = pd.DataFrame(rows, columns=["season", "team", "pct"])
    d = d.groupby(["season", "team"]).pct.sum().reset_index()
    return d[d.team.str.len() <= 3]


def roster_curve():
    ps = ap.player_seasons()
    v = pd.read_parquet("data/derived/vet_season_values.parquet")
    v = v.sort_values("value_ours").drop_duplicates(["pfr_id", "season"], keep="last")
    tot = ps.groupby(["pfr_player_id", "season"]).share.transform("sum")
    ps["w"] = np.where(tot > 0, ps.share / tot, 0)
    ps = ps.merge(v[["pfr_id", "season", "value_ours"]].rename(columns={"pfr_id": "pfr_player_id"}),
                  on=["pfr_player_id", "season"], how="left")
    ps["curve"] = ps.value_ours.fillna(0) * ps.w * 100
    ps["kind"] = np.where(ps.vet, "vet_val", "young_val")
    return ps.pivot_table(index=["season", "team"], columns="kind", values="curve", aggfunc="sum", fill_value=0).reset_index()


def main():
    u = cap_used()
    print("cap used by season (share of cap, mean / sd across teams):")
    print(u[u.season.between(2013, 2025)].groupby("season").pct.agg(["mean", "std", "count"]).round(3).T.to_string())
    rat = ap.ratings()
    D = u.merge(rat, on=["season", "team"]).merge(roster_curve(), on=["season", "team"])
    D["unused"] = (1 - D.pct) * 100
    key = D.set_index(["team", "season"])
    for h in (-1, 1, 2, 3):
        D[f"r{h:+d}"] = [key.rating.get((t, s + h), np.nan) for t, s in zip(D.team, D.season)]
    D["used_next"] = [key.pct.get((t, s + 1), np.nan) * 100 for t, s in zip(D.team, D.season)]
    D = D[D.season.between(2014, 2022)].dropna(subset=["r-1", "r+1", "r+2", "r+3", "used_next"])
    D["rsum"] = D["r+1"] + D["r+2"] + D["r+3"]
    print(f"\nteam-seasons {len(D)} (2014-2022); unused cap mean {D.unused.mean():.1f}%, sd {D.unused.std():.1f}")
    ctrl = "rating + Q('r-1') + vet_val + young_val + C(season)"
    rows = []
    for fe in ("", " + C(team)"):
        for y in ["Q('r+1')", "Q('r+2')", "Q('r+3')", "rsum", "used_next", "rating"]:
            f = f"{y} ~ unused + {ctrl}{fe}" if y != "rating" else f"rating ~ unused + Q('r-1') + vet_val + young_val + C(season){fe}"
            m = smf.ols(f, D).fit(cov_type="cluster", cov_kwds={"groups": D.team})
            lo, hi = m.conf_int().loc["unused"]
            rows.append({"franchise FE": bool(fe), "outcome": y.replace("Q('", "").replace("')", ""),
                         "per 1% unused": m.params["unused"], "lo": lo, "hi": hi, "p": m.pvalues["unused"]})
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nCoefficient on unused cap in season t (points of rating; used_next in % of cap; 'rating' = same season):")
    print(T.round(3).to_string(index=False))
    for fe in (False, True):
        r = T[(T["franchise FE"] == fe) & (T.outcome == "rsum")].iloc[0]
        print(f"  {'with' if fe else 'without'} franchise FE: t+1..t+3 sum {r['per 1% unused']:+.3f} "
              f"({r.lo:+.3f} to {r.hi:+.3f}); shadow price assumes ~+0.12, current charge ~+0.05")
    D.to_parquet("data/derived/cap_rollover_test.parquet")


if __name__ == "__main__":
    main()
