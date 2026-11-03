"""Bridge: how many points of team strength does 1% of cap in player value buy?

For every team-season 2013-2025:
  roster value = sum over players of (snap share for that team) x (market price)
  market price = the player's best veteran contract (share of cap), veteran-minimum
                 floor; the same definition the pick values (value.py) use, so the
                 answer converts those numbers directly into points.
Then regress each team's strength rating (simulate.py) on roster value, split into
position groups, with each season's league average removed.

If the salary market priced winning correctly, every group would buy about the
same points per 1% of cap. Differences show where the market is off.

Known biases, both worth remembering when reading the results:
  - Some leakage: a player's best contract partly rewards seasons in this sample,
    and good teams make players look good. This pushes the slopes up.
  - Market price is a noisy measure of real ability, which pushes slopes down.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from simulate import RELOCATED, load_schedules, season_ratings

RAW = Path("data/raw")
OUT = Path("data/derived")
SEASONS = list(range(2013, 2026))
VET_MIN = 0.005

POS_GROUP = {"QB": "qb",
             "T": "ol", "G": "ol", "C": "ol", "OL": "ol", "OT": "ol", "OG": "ol",
             "WR": "skill", "TE": "skill", "RB": "skill", "FB": "skill",
             "DE": "front", "DT": "front", "NT": "front", "DL": "front",
             "LB": "front", "ILB": "front", "OLB": "front", "MLB": "front",
             "CB": "db", "S": "db", "SS": "db", "FS": "db", "DB": "db"}
GROUPS = ["qb", "ol", "skill", "front", "db"]


def player_prices(c: pl.DataFrame) -> pl.DataFrame:
    """Best veteran contract per OTC player id (rookie deals of drafted players excluded)."""
    vets = c.filter(~(pl.col("year_signed") == pl.col("draft_year")).fill_null(False))
    return vets.group_by("otc_id").agg(price=pl.col("apy_cap_pct").max())


# players.position -> outlook.py position group, for slot-based expected prices
DRAFT_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
               "T": "T", "OT": "T", "OL": "T", "G": "G", "OG": "G", "C": "C",
               "DE": "ED", "OLB": "ED", "DT": "IDL", "NT": "IDL", "DL": "IDL",
               "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "S": "S", "SAF": "S", "SS": "S", "FS": "S", "DB": "S"}


def preseason_prices(c: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Price per (player, season) using only what was known before that season:
    veterans -> pay of their latest veteran contract signed by then;
    players still on a rookie deal -> expected market price for their draft slot
    and position (outlook tier chances x average price of each tier)."""
    curves = pd.read_parquet(OUT / "outlook_curves.parquet")
    lib = pd.read_parquet(OUT / "value_library.parquet")
    tier_price = lib.groupby(["grp", "tier"], observed=True)["price"].mean().unstack()
    curves["exp_price"] = sum(curves[t] * curves.grp.map(tier_price[t]) for t in ["bust", "starter", "star"])
    slot = pl.from_pandas(curves[["grp", "pick", "exp_price"]]).cast({"pick": pl.Int32})

    vets = (c.filter(~(pl.col("year_signed") == pl.col("draft_year")).fill_null(False))
            .select("otc_id", "year_signed", "apy_cap_pct"))
    grid = pl.DataFrame({"season": SEASONS}).cast(pl.Int32)
    ids = players.select(otc_id=pl.col("otc_id").cast(pl.Int32, strict=False), pfr_player_id="pfr_id",
                         draft_year=pl.col("draft_year").cast(pl.Int32),
                         pick=pl.col("draft_pick").cast(pl.Int32),
                         grp=pl.col("position").replace_strict(DRAFT_GROUP, default=None)).drop_nulls("pfr_player_id")
    latest = (ids.select("pfr_player_id", "otc_id").drop_nulls().join(grid, how="cross")
              .join(vets, on="otc_id").filter(pl.col("year_signed") <= pl.col("season"))
              .sort("year_signed").group_by("pfr_player_id", "season").last()
              .select("pfr_player_id", "season", vet_price="apy_cap_pct"))
    rookie = (ids.drop_nulls(["draft_year", "pick", "grp"]).join(grid, how="cross")
              .filter((pl.col("season") >= pl.col("draft_year")) & (pl.col("season") <= pl.col("draft_year") + 4))
              .join(slot, on=["grp", "pick"], how="left")
              .select("pfr_player_id", "season", "exp_price"))
    return (latest.join(rookie, on=["pfr_player_id", "season"], how="full", coalesce=True)
            # on a rookie deal with no veteran contract yet -> slot price; else latest contract
            .with_columns(price=pl.coalesce("vet_price", "exp_price"))
            .select("pfr_player_id", "season", "price"))


def roster_values(s: pl.DataFrame, players: pl.DataFrame, prices: pl.DataFrame,
                  by_season: bool = False) -> pd.DataFrame:
    games = pl.when(pl.col("season") >= 2021).then(17).otherwise(16)
    ps = (s.filter((pl.col("game_type") == "REG") & pl.col("season").is_in(SEASONS))
          .with_columns(team=pl.col("team").replace(RELOCATED))
          .group_by("pfr_player_id", "season", "team")
          .agg(off=pl.col("offense_pct").sum(), dfn=pl.col("defense_pct").sum(),
               pos=pl.col("position").mode().first())
          .with_columns(share=(pl.max_horizontal("off", "dfn") / games).clip(0, 1),
                        grp=pl.col("pos").replace_strict(POS_GROUP, default=None))
          .filter(pl.col("grp").is_not_null()))
    ids = (players.select(pfr_player_id="pfr_id", otc_id=pl.col("otc_id").cast(pl.Int32, strict=False))
           .drop_nulls().unique("pfr_player_id"))
    ps = ps.join(ids, on="pfr_player_id", how="left")
    ps = (ps.join(prices, on=["pfr_player_id", "season"], how="left") if by_season
          else ps.join(prices, on="otc_id", how="left"))
    ps = (ps
          .with_columns(price=pl.col("price").fill_null(VET_MIN).clip(lower_bound=VET_MIN))
          .with_columns(value=pl.col("share") * pl.col("price") * 100))  # % of cap
    linked = ps.select((pl.col("otc_id").is_not_null() * pl.col("share")).sum() / pl.col("share").sum()).item()
    wide = (ps.group_by("season", "team", "grp").agg(pl.col("value").sum()).to_pandas()
            .pivot_table(index=["season", "team"], columns="grp", values="value", fill_value=0)
            .reset_index())
    return wide, linked


def main():
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    s = pl.read_parquet(RAW / "snap_counts.parquet")
    players = pl.read_parquet(RAW / "players.parquet")
    sched = load_schedules()

    rat = []
    for yr in SEASONS:
        g = sched[(sched.season == yr) & (sched.game_type == "REG") & sched.result.notna()]
        r, *_ = season_ratings(g, sorted(set(g.home_team) | set(g.away_team)))
        rat.append(pd.DataFrame({"season": yr, "team": r.index, "rating": r.values}))
    rat = pd.concat(rat)
    modes = {"career-best prices (may leak)": roster_values(s, players, player_prices(c)),
             "pre-season prices (no leak, noisier)": roster_values(s, players, preseason_prices(c, players), True)}
    saved = {}
    for mode, (vals, linked) in modes.items():
        print(f"=== {mode} ===")
        saved[mode] = analyze(vals, linked, rat)
    q = saved["career-best prices (may leak)"]
    p = saved["pre-season prices (no leak, noisier)"]
    json.dump({"career_best": q, "preseason": p}, open(OUT / "bridge.json", "w"), indent=2)


def analyze(vals, linked, rat) -> dict:
    df = vals.merge(rat, on=["season", "team"])
    df["nonqb"] = df[["ol", "skill", "front", "db"]].sum(axis=1)
    df["total"] = df[GROUPS].sum(axis=1)
    cols = GROUPS + ["nonqb", "total"]
    df[cols] = df[cols] - df.groupby("season")[cols].transform("mean")  # vs that season's average

    print(f"Team-seasons: {len(df)}; snaps linked to a contract record: {linked:.0%}\n")
    print("Roster value spread across teams (sd, % of cap):", df[cols].std().round(1).to_dict(), "\n")

    fits = {"one number for everyone": "rating ~ total",
            "QB vs everyone else": "rating ~ qb + nonqb",
            "five position groups": "rating ~ " + " + ".join(GROUPS)}
    res = {}
    for name, f in fits.items():
        m = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": df["team"]})
        res[name] = m
        terms = [t for t in m.params.index if t != "Intercept"]
        print(f"{name}: R² = {m.rsquared:.2f}")
        for t in terms:
            lo, hi = m.conf_int().loc[t]
            print(f"   {t:6s} {m.params[t]:+.2f} pts per 1% of cap  (95% range {lo:+.2f} to {hi:+.2f})")
        print()

    # Out-of-sample check: fit on 2013-2020, predict 2021-2025 ratings
    tr, te = df[df.season <= 2020], df[df.season >= 2021]
    for name, f in fits.items():
        m = smf.ols(f, tr).fit()
        err = te.rating - m.predict(te)
        print(f"2021-25 prediction error ({name}): {np.sqrt((err ** 2).mean()):.2f} pts "
              f"(guessing average: {np.sqrt((te.rating ** 2).mean()):.2f})")

    print()
    q = res["QB vs everyone else"]
    return {"qb": q.params["qb"], "nonqb": q.params["nonqb"],
            "all": res["one number for everyone"].params["total"]}


if __name__ == "__main__":
    main()
