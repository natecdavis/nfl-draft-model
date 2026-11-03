"""Milestone 2: what a pick is worth, per season, in salary-cap terms.

For every pick in the 2013-2021 drafts (snap data starts in 2013), seasons 1-5:

  value_t   = snap share in season t  x  his market price
              market price = best veteran contract he signed after his rookie
              deal (share of cap); veteran minimum (0.5%) if he never got one.
              The idea: the league's later pay is its verdict on how good he was.
  cost_t    = rookie contract pay (share of cap), seasons 1-4. Rounds 1-2 count it
              every year (guaranteed money); rounds 3+ only in seasons he played,
              since a cut late-round pick is replaced at about the same price.
  season 5  = round 1 only: the team holds a fifth-year option and uses it only if
              he's worth more than its price (approximated as the position's star
              pay bar for picks 1-10, 70% of it for picks 11-32). Other rounds: he
              is on a market-rate deal or gone, so no surplus.
  surplus_t = value_t - cost_t

Expected values by pick and position combine the M1 tier chances with what each
tier is worth: E[value] = sum over tiers of P(tier | pick, pos) x average value of
that tier (market price by tier and position; playing-time pattern by tier and
pick range). This keeps the tier mix, which the title-odds simulator will need.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from outlook import GROUPS, OUT, RAW, TIERS, pos_group, season_shares, star_thresholds

LIB_DRAFTS = list(range(2013, 2022))
VET_MIN = 0.005
OPTION_SHARE = {True: 1.0, False: 0.7}  # pick <= 10 ?
BUCKETS = ([0, 32, 100, 262], ["1-32", "33-100", "101+"])
YEARS = [1, 2, 3, 4, 5]


def rookie_cost_curve(c: pl.DataFrame) -> pd.Series:
    """Typical rookie-deal pay (share of cap) for each pick, 2013-2026 drafts, smoothed."""
    r = (c.filter((pl.col("year_signed") == pl.col("draft_year")) & (pl.col("years") == 4)
                  & pl.col("draft_overall").is_not_null() & pl.col("draft_year").is_between(2013, 2026))
         .group_by("draft_overall").agg(pl.col("apy_cap_pct").median()).sort("draft_overall").to_pandas())
    r = r.set_index("draft_overall")["apy_cap_pct"].reindex(range(1, 263)).interpolate().bfill().ffill()
    return r.rolling(5, center=True, min_periods=1).median().rename("rookie_cost")


def build_library(outcomes: pd.DataFrame, shares: pl.DataFrame, cost: pd.Series,
                  thr: pd.Series) -> pd.DataFrame:
    """One row per historical pick with season-by-season value, cost and surplus."""
    lib = outcomes[outcomes.draft_year.isin(LIB_DRAFTS) & outcomes.grp.isin(GROUPS)].copy()
    lib["price"] = lib["best_pay"].fillna(VET_MIN).clip(lower=VET_MIN)
    lib["rookie_cost"] = lib["pick"].map(cost)
    sh = shares.to_pandas().set_index(["pfr_player_id", "season"])["share"]
    for t in YEARS:
        key = list(zip(lib["pfr_id"], lib["draft_year"] + t - 1))
        lib[f"share_{t}"] = sh.reindex(key).fillna(0).to_numpy()
        lib[f"value_{t}"] = lib[f"share_{t}"] * lib["price"]
    early = lib["pick"] <= 64  # rounds 1-2, roughly
    for t in YEARS[:4]:
        lib[f"cost_{t}"] = np.where(early | (lib[f"share_{t}"] > 0), lib["rookie_cost"], 0.0)
    # Fifth-year option, round 1 only
    bar = thr.reindex(list(zip(lib["draft_year"] + 4, lib["grp"]))).to_numpy()
    opt = bar * np.where(lib["pick"] <= 10, OPTION_SHARE[True], OPTION_SHARE[False])
    first = lib["pick"] <= 32
    lib["cost_5"] = np.where(first & (lib["value_5"] > opt), opt, 0.0)
    lib["value_5"] = np.where(first & (lib["value_5"] > opt), lib["value_5"], 0.0)  # others: market deal
    for t in YEARS:
        lib[f"surplus_{t}"] = lib[f"value_{t}"] - lib[f"cost_{t}"]
    lib["value_5yr"] = lib[[f"value_{t}" for t in YEARS]].sum(axis=1)
    lib["cost_5yr"] = lib[[f"cost_{t}" for t in YEARS]].sum(axis=1)
    lib["surplus_5yr"] = lib["value_5yr"] - lib["cost_5yr"]
    lib["bucket"] = pd.cut(lib["pick"], BUCKETS[0], labels=BUCKETS[1])
    return lib


def tier_profiles(lib: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What each tier is worth.
    price:  average market price by tier x position
    share:  average snap share by season, tier x pick range (pooled over positions)
    opt:    fifth-year-option surplus per unit of price, by tier (round 1 only)
    """
    price = lib.groupby(["grp", "tier"], observed=True)["price"].agg(["mean", "size"])
    share = lib.groupby(["tier", "bucket"], observed=True)[[f"share_{t}" for t in YEARS[:4]]].mean()
    r1 = lib[lib.pick <= 32]
    opt = (r1["surplus_5"] / r1["price"]).groupby(r1["tier"], observed=True).mean()
    onroster = lib.groupby(["tier", "bucket"], observed=True)[[f"cost_{t}" for t in YEARS[:4]]] \
                  .apply(lambda d: (d > 0).mean())
    return price, share, opt, onroster


def expected_values(curves: pd.DataFrame, lib: pd.DataFrame, cost: pd.Series) -> pd.DataFrame:
    price, share, opt, onroster = tier_profiles(lib)
    rows = []
    for _, r in curves.iterrows():
        pick, grp = int(r.pick), r.grp
        b = pd.cut([pick], BUCKETS[0], labels=BUCKETS[1])[0]
        out = {"grp": grp, "pick": pick, "rookie_cost": cost[pick]}
        for t in YEARS:
            v = c_ = 0.0
            for tier in TIERS:
                p, q = r[tier], price.loc[(grp, tier), "mean"]
                if t <= 4:
                    v += p * share.loc[(tier, b), f"share_{t}"] * q
                    c_ += p * onroster.loc[(tier, b), f"cost_{t}"] * cost[pick]
                elif pick <= 32:
                    v += p * opt.get(tier, 0) * q  # option-year surplus (value net of option cost)
            out[f"value_{t}"], out[f"cost_{t}"] = v, c_
        rows.append(out)
    ev = pd.DataFrame(rows)
    ev["value_5yr"] = ev[[f"value_{t}" for t in YEARS]].sum(axis=1)
    ev["cost_5yr"] = ev[[f"cost_{t}" for t in YEARS]].sum(axis=1)
    ev["surplus_5yr"] = ev["value_5yr"] - ev["cost_5yr"]
    return ev


def main():
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    s = pl.read_parquet(RAW / "snap_counts.parquet")
    outcomes = pd.read_parquet(OUT / "pick_outcomes.parquet")
    curves = pd.read_parquet(OUT / "outlook_curves.parquet")

    thr = star_thresholds(c)
    cost = rookie_cost_curve(c)
    lib = build_library(outcomes, season_shares(s), cost, thr)
    lib.to_parquet(OUT / "value_library.parquet")
    ev = expected_values(curves, lib, cost)
    ev.to_parquet(OUT / "pick_value.parquet")

    pd.set_option("display.width", 170, "display.max_rows", 200)
    pct = lambda x: (x * 100).round(1)
    price, share, opt, _ = tier_profiles(lib)
    print("Market price by tier (% of cap):\n", pct(price["mean"].unstack()), "\n")
    print("Snap share by season, tier x pick range:\n", share.round(2), "\n")
    print("Option-year surplus per unit of price (round 1):", opt.round(2).to_dict(), "\n")

    picks = [1, 5, 10, 20, 32, 50, 64, 100, 150, 220]
    print("Expected 5-year SURPLUS (% of one season's cap) by position and pick:")
    print(pct(ev[ev.pick.isin(picks)].pivot(index="grp", columns="pick", values="surplus_5yr")), "\n")

    # Check: composed expectation vs the actual average of historical picks
    lib = lib.merge(ev[["grp", "pick", "surplus_5yr"]].rename(columns={"surplus_5yr": "expected"}),
                    on=["grp", "pick"])
    lib["slot"] = pd.cut(lib.pick, [0, 10, 32, 64, 100, 150, 262],
                         labels=["1-10", "11-32", "33-64", "65-100", "101-150", "151+"])
    chk = lib.groupby("slot", observed=True).agg(expected=("expected", "mean"), actual=("surplus_5yr", "mean"),
                                                 value=("value_5yr", "mean"), cost=("cost_5yr", "mean"),
                                                 n=("pick", "size"))
    print("All positions, expected vs actual 5-year surplus (% of cap):\n",
          chk.assign(**{k: pct(chk[k]) for k in ["expected", "actual", "value", "cost"]}), "\n")
    chk2 = lib.groupby("grp").agg(expected=("expected", "mean"), actual=("surplus_5yr", "mean"))
    print("By position, expected vs actual (% of cap):\n", pct(chk2).T)


if __name__ == "__main__":
    main()
