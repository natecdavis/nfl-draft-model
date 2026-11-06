"""Does the model overrate WRs? (53% of its round-1 recommendations are WRs; 3% of board-best players.)

1. Calibration out of sample (leave-one-draft-out review, 2014-2022): delivered / predicted titles, by
   position and round, for the model's recommendations and for teams' actual picks.
2. Does WR pay buy team strength like other non-QB pay? Bridge regression (src/bridge.py) with WRs split
   out: team rating ~ QB value + WR value + other non-QB value (snap share x market price, % of cap),
   team-seasons 2013-2025, career-best and leak-free pre-season prices, with and without team effects.
   The model prices every non-QB at the same 0.19 points per 1% of cap, so a WR coefficient well below
   the others means WR pay overstates what WRs add, and the model (whose titles are in that currency,
   predicted and realized alike) overrates them.

Run:  python scripts/wr_check.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

import bridge as B  # noqa: E402
from outlook import OUT  # noqa: E402
from simulate import load_schedules, season_ratings  # noqa: E402

rng = np.random.default_rng(0)


def ratio(pred, real):
    pred, real = np.asarray(pred, float), np.asarray(real, float)
    if len(pred) < 5:
        return f"n={len(pred)}"
    b = []
    for _ in range(2000):
        i = rng.integers(0, len(pred), len(pred))
        b.append(real[i].sum() / pred[i].sum())
    return f"{real.sum() / pred.sum():.2f} ({np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f}) n={len(pred)}"


def calibration():
    d = pd.read_parquet(OUT / "review_picks.parquet")
    d = d[d.year.between(2014, 2022) & d.real_team.notna()]
    print("1. delivered / predicted titles (leave-one-draft-out), by position")
    for lab, f in [("round 1", d["round"] == 1), ("rounds 2-4", d["round"].between(2, 4))]:
        x = d[f]
        rec = x.drop_duplicates(["year", "rec_player"])
        print(f" {lab}")
        for g in ["WR", "QB", "ED", "T", "CB", "IDL", "TE", "LB", "S"]:
            r, t = rec[rec.rec_grp == g], x[x.grp == g]
            print(f"   {g:4s} model recs {ratio(r.value_rec, r.real_model):28s} | team picks {ratio(t.value_actual, t.real_team)}")
        o = rec[~rec.rec_grp.isin(["WR", "QB"])]
        t = x[~x.grp.isin(["WR", "QB"])]
        print(f"   rest model recs {ratio(o.value_rec, o.real_model):28s} | team picks {ratio(t.value_actual, t.real_team)}")


def bridge_split():
    c = pl.read_parquet(B.RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    snaps = pl.read_parquet(B.RAW / "snap_counts.parquet")
    players = pl.read_parquet(B.RAW / "players.parquet")
    sched = load_schedules()
    rat = []
    for yr in B.SEASONS:
        g = sched[(sched.season == yr) & (sched.game_type == "REG") & sched.result.notna()]
        r, *_ = season_ratings(g, sorted(set(g.home_team) | set(g.away_team)))
        rat.append(pd.DataFrame({"season": yr, "team": r.index, "rating": r.values}))
    rat = pd.concat(rat)
    B.POS_GROUP["WR"] = "wr"
    print("\n2. team rating per 1% of cap in on-field value (model: every non-QB +0.19)")
    for mode, prices, by in [("career-best prices", B.player_prices(c), False),
                             ("pre-season prices (no leak)", B.preseason_prices(c, players), True)]:
        vals, _ = B.roster_values(snaps, players, prices, by)
        df = vals.merge(rat, on=["season", "team"])
        df["other"] = df[["ol", "skill", "front", "db"]].sum(axis=1)
        cols = ["qb", "wr", "other", "rating"]
        df[cols] = df[cols] - df.groupby("season")[cols].transform("mean")
        print(f"  {mode} ({len(df)} team-seasons; sd of WR value {df.wr.std():.1f}%, other non-QB {df.other.std():.1f}%)")
        for fe in ("", " + C(team)"):
            m = smf.ols("rating ~ qb + wr + other" + fe, df).fit(cov_type="cluster", cov_kwds={"groups": df["team"]})
            ci = m.conf_int()
            w = m.params.wr - m.params.other
            # difference CI from the covariance
            v = m.cov_params()
            se = np.sqrt(v.loc["wr", "wr"] + v.loc["other", "other"] - 2 * v.loc["wr", "other"])
            print(f"    {'within team' if fe else 'pooled     '}: " + " | ".join(
                f"{t} {m.params[t]:+.2f} ({ci.loc[t, 0]:+.2f} to {ci.loc[t, 1]:+.2f})" for t in ["qb", "wr", "other"])
                + f" | WR minus other {w:+.2f} ({w - 1.96 * se:+.2f} to {w + 1.96 * se:+.2f})")
        # out of sample: does a separate WR rate predict 2021-25 better?
        tr, te = df[df.season <= 2020], df[df.season >= 2021]
        for f in ["rating ~ qb + wr + other", "rating ~ qb + I(wr + other)"]:
            m = smf.ols(f, tr).fit()
            print(f"    2021-25 error, {f}: {np.sqrt(((te.rating - m.predict(te)) ** 2).mean()):.3f}")


if __name__ == "__main__":
    calibration()
    bridge_split()
