"""How the model values linebackers: does LB season value turn into team strength like other positions, and does
the pay curve cap top LBs below what they're worth?

Prompt (Nov 22 2026): SF's optimizer cuts Fred Warner. His healthy seasons price at 5.0% of cap (quality percentile
1.00, ~93% of snaps) - the LB pay curve's ceiling - against a 9.2% cap hit; top LB contracts run ~8-9% of cap.

Season value (scripts/vet_season_values.py) = pay curve price_g(snap share, quality) x games: what the market pays
that role at that position. The model converts value to team strength at one non-QB rate.

1 rate by group: team-season rating (2013-25, season means removed, clustered by team) ~ curve value summed by
  veteran pricing group (QB, RB, WR, TE, OL, DL, EDGE, LB, DB). If the curve prices LBs right, the LB slope matches
  the other non-QB groups (test LB = pooled others).
2 the ceiling: per group, the most a full-time season can be worth on the curve (99th percentile of value per full
  season among share >= 0.8 seasons) vs what the top of the market pays (90th / 99th percentile of veteran APY),
  and for the 10% best-paid veterans the ratio curve value / APY in force. LB-specific if LB's ratio is lowest.
3 pay above the curve: rating ~ curve by group + each group's pay above its curve (snap share x (APY in force -
  curve price per full season), veterans). If teams' extra pay for top LBs buys strength the curve misses, the LB
  excess slope is > 0 (and near the curve's rate).
Run: .venv/bin/python scripts/lb_value_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import age_pricing as ap  # noqa: E402

GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


def data():
    ps = ap.player_seasons()  # pfr_player_id, season, team, share, grp, price (APY in force, % cap), value, vet, age
    v = pd.read_parquet(os.environ.get("VET_VALUES", "data/derived/vet_season_values.parquet"))
    v = v.sort_values("value_ours").drop_duplicates(["pfr_id", "season"], keep="last")
    tot = ps.groupby(["pfr_player_id", "season"]).share.transform("sum")
    ps["w"] = np.where(tot > 0, ps.share / tot, 0)
    ps = ps.merge(v[["pfr_id", "season", "value_ours", "group", "gfrac"]].rename(columns={"pfr_id": "pfr_player_id"}),
                  on=["pfr_player_id", "season"], how="left")
    ps["curve"] = ps.value_ours.fillna(0) * ps.w * 100
    ps["g"] = ps.group.fillna(ps.grp.map({"qb": "QB", "ol": "OL", "db": "DB"}))
    ps = ps[ps.g.isin(GROUPS)].copy()
    # curve price per full season (value / (share x games)), and pay above it for veterans
    full = (ps.share * ps.gfrac.fillna(1)).replace(0, np.nan)
    ps["curve_full"] = (ps.value_ours * 100 / full).where(ps.share >= 0.2)
    ps["excess"] = np.where(ps.vet & ps.curve_full.notna(), ps.share * ps.w.clip(upper=1) * (ps.price - ps.curve_full), 0.0)
    return ps


def regress(ps, cols_money, label, test_lb=None):
    W = pd.concat([ps.pivot_table(index=["season", "team"], columns="g", values=m, aggfunc="sum", fill_value=0)
                   .add_prefix(f"{m}_") for m in cols_money], axis=1).reset_index()
    rat = ap.ratings()
    df = W.merge(rat, on=["season", "team"])
    cols = [c for c in W.columns if c not in ("season", "team")]
    df[cols] = df[cols] - df.groupby("season")[cols].transform("mean")
    m = smf.ols("rating ~ " + " + ".join(f"Q('{c}')" for c in cols), df).fit(cov_type="cluster",
                                                                            cov_kwds={"groups": df.team})
    ci = m.conf_int()
    T = pd.DataFrame({"pts per 1%": m.params[1:].values, "lo": ci.iloc[1:, 0].values, "hi": ci.iloc[1:, 1].values,
                      "mean % cap": W[cols].mean().values}, index=cols)
    print(f"\n--- {label}; team-seasons {len(df)}, R2 {m.rsquared:.2f} ---")
    print(T.round(3).to_string())
    if test_lb:
        pre = test_lb
        others = [f"Q('{pre}_{g}')" for g in GROUPS if g not in ("QB", "LB")]
        R = np.zeros(len(m.params))
        R[list(m.params.index).index(f"Q('{pre}_LB')")] = 1
        for o in others:
            R[list(m.params.index).index(o)] = -1 / len(others)
        t = m.t_test(R)
        lo, hi = np.asarray(t.conf_int()).ravel()
        print(f"LB minus mean of other non-QB groups ({pre}): {float(np.ravel(t.effect)[0]):+.3f} "
              f"({lo:+.3f} to {hi:+.3f}), p {float(np.ravel(t.pvalue)[0]):.3f}")
    return m, T


def ceiling(ps):
    v = pd.read_parquet(os.environ.get("VET_VALUES", "data/derived/vet_season_values.parquet"))
    v = v[(v.share >= 0.8) & (v.gfrac >= 0.8)]
    v["per_full"] = v.value_ours * 100 / (v.share * v.gfrac)
    vet = ps[ps.vet & (ps.share >= 0.5) & ps.curve_full.notna()]
    rows = []
    for g in GROUPS:
        a = vet[vet.g == g]
        top = a[a.price >= a.price.quantile(0.9)]
        rows.append({"group": g, "curve max (p99, full season)": v[v.group == g].per_full.quantile(0.99),
                     "APY p90": a.price.quantile(0.9), "APY p99": a.price.quantile(0.99),
                     "top-10%-paid: curve / APY": top.curve_full.sum() / top.price.sum(),
                     "all vets: curve / APY": a.curve_full.sum() / a.price.sum(), "n top": len(top)})
    print("\n--- 2 the ceiling: curve value per full season vs pay (% of cap), veterans with share >= 0.5 ---")
    print(pd.DataFrame(rows).set_index("group").round(2).to_string())


def main():
    pd.set_option("display.width", 200)
    ps = data()
    regress(ps, ["curve"], "1 curve value by group", test_lb="curve")
    ceiling(ps)
    regress(ps, ["curve", "excess"], "3 curve value + veterans' pay above the curve, by group", test_lb="excess")
    lb = ps[(ps.g == "LB") & ps.vet & (ps.share >= 0.5)]
    top = lb[lb.price >= lb.price.quantile(0.9)]
    print(f"\nWarner-type LBs (top 10% paid, share >= 0.5): mean APY {top.price.mean():.2f}, curve {top.curve_full.mean():.2f}")


if __name__ == "__main__":
    main()
