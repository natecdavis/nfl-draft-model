"""Does a cheap QB's cap savings turn into team strength, as the model assumes?

The model values a rookie-deal QB as (a) his on-field value (snap share x market price, 0.31-0.34
points of team strength per 1% of cap) plus (b) the cap he saves, spent on non-QB veterans at 0.19
points per 1% (src/bridge.py). (a) is fit to real team ratings; (b) never was. Tested here on
team-seasons 2013-2025, everything relative to that season's league average:

  1. non-QB roster value ~ main QB's cap hit: the model implies -1 (each 1% of cap the QB doesn't
     take becomes 1% of non-QB value)
  2. team rating ~ QB on-field value + QB cap hit: the model implies -0.19 per 1% on the cap hit
     (-0.19 x the slope from 1 if savings are only partly spent)
Both with and without team fixed effects (within-team changes, e.g. a QB's rookie deal -> extension).

Run:  python scripts/qb_savings.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

import bridge as B  # noqa: E402
from simulate import RELOCATED, load_schedules, season_ratings  # noqa: E402


def qb_cap() -> pd.DataFrame:
    """Main QB per team-season (most offensive snaps) and his cap hit, % of cap."""
    s = pd.read_parquet(B.RAW / "snap_counts.parquet")
    s = s[(s.game_type == "REG") & (s.position == "QB") & s.season.isin(B.SEASONS)].copy()
    s["team"] = s.team.replace(RELOCATED)
    m = s.groupby(["season", "team", "pfr_player_id"], as_index=False).offense_pct.sum()
    m = m.sort_values("offense_pct").groupby(["season", "team"]).tail(1)
    pl_ = pd.read_parquet(B.RAW / "players.parquet")[["pfr_id", "gsis_id", "otc_id", "draft_year"]]
    m = m.merge(pl_.rename(columns={"pfr_id": "pfr_player_id"}), on="pfr_player_id", how="left")
    c = pd.read_parquet(B.RAW / "contracts.parquet")
    hist = {}
    for oid, sh in zip(c.otc_id, c.season_history):
        if sh is None:
            continue
        for h in sh:
            try:
                k = (int(oid), int(h["year"]))
            except (TypeError, ValueError):
                continue
            if h.get("cap_percent") is not None:
                hist[k] = max(hist.get(k, 0.0), float(h["cap_percent"]))
    m["qbcap"] = [hist.get((int(o), int(y))) if pd.notna(o) else None
                  for o, y in zip(pd.to_numeric(m.otc_id, errors="coerce"), m.season)]
    m["qbcap"] = m.qbcap * 100
    m["rookie"] = (m.season - pd.to_numeric(m.draft_year, errors="coerce")).between(0, 3)
    return m[["season", "team", "qbcap", "rookie", "offense_pct"]]


def main():
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
    Q = qb_cap()
    print(f"main QB cap hit found for {Q.qbcap.notna().mean():.0%} of team-seasons; "
          f"rookie-deal QBs: mean {Q[Q.rookie].qbcap.mean():.1f}% of cap, others {Q[~Q.rookie].qbcap.mean():.1f}%")
    for mode, prices, by in [("career-best prices", B.player_prices(c), False),
                             ("pre-season prices (no leak)", B.preseason_prices(c, players), True)]:
        vals, _ = B.roster_values(snaps, players, prices, by)
        df = vals.merge(rat, on=["season", "team"]).merge(Q, on=["season", "team"]).dropna(subset=["qbcap"])
        df["nonqb"] = df[["ol", "skill", "front", "db"]].sum(axis=1)
        cols = ["qb", "nonqb", "qbcap", "rating"]
        df[cols] = df[cols] - df.groupby("season")[cols].transform("mean")
        print(f"\n=== {mode}: {len(df)} team-seasons ===")
        for lab, f, term, implied in [
            ("1. non-QB value per 1% of QB cap hit", "nonqb ~ qbcap", "qbcap", "-1.00"),
            ("2. rating per 1% of QB cap hit, QB value held fixed", "rating ~ qb + qbcap", "qbcap", "-0.19"),
            ("   (check) rating per 1% of QB value", "rating ~ qb + qbcap", "qb", "+0.31 to +0.34"),
            ("   (check) rating per 1% of non-QB value", "rating ~ qb + nonqb", "nonqb", "+0.19")]:
            out = []
            for fe in ("", " + C(team)"):
                m = smf.ols(f + fe, df).fit(cov_type="cluster", cov_kwds={"groups": df["team"]})
                lo, hi = m.conf_int().loc[term]
                out.append(f"{m.params[term]:+.2f} ({lo:+.2f} to {hi:+.2f})")
            print(f"  {lab:52s} model {implied:>14s} | data {out[0]} | within team {out[1]}")
        g = df.assign(good=df.qb > df.qb.median()).groupby(["good", "rookie"])[["rating", "qbcap", "nonqb"]].mean()
        print("  mean by QB on-field value (above/below median) and rookie deal:\n" + g.round(2).to_string())


if __name__ == "__main__":
    main()
