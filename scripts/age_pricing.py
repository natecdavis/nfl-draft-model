"""Lead #3 of the contract-efficiency list: does the market overpay older veterans?

Bridge-style team test (src/bridge.py, pre-season prices = no leak): team-season rating (simulate.season_ratings,
season-demeaned) on roster value split by contract type and age at the season's start. Value = snap share x
pre-season price (% of cap): latest veteran contract signed by then, or for players still on a rookie deal the
expected price for their draft slot (bridge.preseason_prices). If the market prices age right, every veteran
bucket buys the same points per 1% of cap.

Two money measures:
  value  = share x price (what bridge / the model use: dollars that played)
  cap    = price for every player with a snap that season (dollars spent, including those benched)
Buckets: QB (rookie deal / vet <30 / vet 30+); non-QB rookie deal; non-QB vets by age <=26, 27-29, 30-32, 33+.
Pre-specified test: equal slopes across non-QB vet age buckets; the lead predicts falling slopes with age.
Also non-QB vets 30+ by position group (skill / ol / front / db), as a descriptive split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402
os.chdir(ROOT)
import bridge  # noqa: E402
from simulate import RELOCATED, load_schedules, season_ratings  # noqa: E402

RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
SEASONS = bridge.SEASONS


def prices_typed(c, players):
    """bridge.preseason_prices, keeping whether the price is a veteran contract, and its signing year."""
    p = bridge.preseason_prices(c, players)
    vets = (c.filter(~(pl.col("year_signed") == pl.col("draft_year")).fill_null(False))
            .select("otc_id", "year_signed"))
    ids = players.select(otc_id=pl.col("otc_id").cast(pl.Int32, strict=False), pfr_player_id="pfr_id").drop_nulls()
    grid = pl.DataFrame({"season": SEASONS}).cast(pl.Int32)
    vet = (ids.join(grid, how="cross").join(vets, on="otc_id").filter(pl.col("year_signed") <= pl.col("season"))
           .group_by("pfr_player_id", "season").agg(signed=pl.col("year_signed").max()))
    return p.join(vet, on=["pfr_player_id", "season"], how="left").to_pandas()


def player_seasons():
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    players = pl.read_parquet(RAW / "players.parquet")
    s = pl.read_parquet(RAW / "snap_counts.parquet")
    games = pl.when(pl.col("season") >= 2021).then(17).otherwise(16)
    ps = (s.filter((pl.col("game_type") == "REG") & pl.col("season").is_in(SEASONS))
          .with_columns(team=pl.col("team").replace(RELOCATED))
          .group_by("pfr_player_id", "season", "team")
          .agg(off=pl.col("offense_pct").sum(), dfn=pl.col("defense_pct").sum(), pos=pl.col("position").mode().first())
          .with_columns(share=(pl.max_horizontal("off", "dfn") / games).clip(0, 1),
                        grp=pl.col("pos").replace_strict(bridge.POS_GROUP, default=None))
          .filter(pl.col("grp").is_not_null())).to_pandas()
    pr = prices_typed(c, players)
    ps = ps.merge(pr, on=["pfr_player_id", "season"], how="left")
    bd = players.select(pfr_player_id="pfr_id", birth_date="birth_date").drop_nulls().unique("pfr_player_id").to_pandas()
    ps = ps.merge(bd, on="pfr_player_id", how="left")
    ps["age"] = (pd.to_datetime(ps.season.astype(str) + "-09-01") - pd.to_datetime(ps.birth_date)).dt.days / 365.25
    ps["vet"] = ps.signed.notna()
    ps["price"] = ps.price.fillna(bridge.VET_MIN).clip(lower=bridge.VET_MIN) * 100
    ps["value"] = ps.share * ps.price
    ps["cap"] = np.where(ps.share > 0, ps.price, 0)
    return ps


def bucket(r):
    if r.grp == "qb":
        if not r.vet:
            return "qb_rookie"
        return "qb_vet_30plus" if r.age >= 30 else "qb_vet_u30"
    if not r.vet:
        return "rookie_deal"
    if np.isnan(r.age):
        return "vet_unknown_age"
    a = r.age
    return "vet_le26" if a < 27 else "vet_27_29" if a < 30 else "vet_30_32" if a < 33 else "vet_33plus"


def ratings():
    sched = load_schedules()
    out = []
    for yr in SEASONS:
        g = sched[(sched.season == yr) & (sched.game_type == "REG") & sched.result.notna()]
        r, *_ = season_ratings(g, sorted(set(g.home_team) | set(g.away_team)))
        out.append(pd.DataFrame({"season": yr, "team": r.index, "rating": r.values}))
    return pd.concat(out)


def fit(ps, rat, money, key, label, test=None):
    W = ps.pivot_table(index=["season", "team"], columns=key, values=money, aggfunc="sum", fill_value=0).reset_index()
    cols = [c for c in W.columns if c not in ("season", "team")]
    df = W.merge(rat, on=["season", "team"])
    df[cols] = df[cols] - df.groupby("season")[cols].transform("mean")
    m = smf.ols("rating ~ " + " + ".join(f"Q('{c}')" for c in cols), df).fit(
        cov_type="cluster", cov_kwds={"groups": df.team})
    ci = m.conf_int()
    sd = df[cols].std()
    T = pd.DataFrame({"pts per 1%": m.params[1:].values, "lo": ci.iloc[1:, 0].values, "hi": ci.iloc[1:, 1].values,
                      "sd across teams": sd.values,
                      "mean % cap": W[cols].mean().values}, index=cols)
    print(f"\n--- {label} ({money}); team-seasons {len(df)}, R2 {m.rsquared:.2f} ---")
    print(T.round(3).to_string())
    if test:
        names = [f"Q('{c}')" for c in test]
        hyp = ", ".join(f"{a} = {b}" for a, b in zip(names[:-1], names[1:]))
        print(f"equal slopes {test}: p {float(m.f_test(hyp).pvalue):.3f}")
        # linear trend in age across the buckets (contrast -1.5,-0.5,0.5,1.5 / 5)
        w = np.array([-3, -1, 1, 3]) / 10 if len(test) == 4 else None
        if w is not None:
            R = np.zeros(len(m.params))
            for wi, n in zip(w, names):
                R[list(m.params.index).index(n)] = wi
            t = m.t_test(R)
            print(f"slope trend per bucket step: {float(np.ravel(t.effect)[0]):+.3f} "
                  f"({float(np.asarray(t.conf_int()).ravel()[0]):+.3f} to {float(np.asarray(t.conf_int()).ravel()[1]):+.3f}), "
                  f"p {float(np.ravel(t.pvalue)[0]):.3f}")
    return T


def main():
    ps = player_seasons()
    ps["b"] = ps.apply(bucket, axis=1)
    rat = ratings()
    print("share of snap-weighted veteran value by bucket:")
    print((ps.groupby("b").value.sum() / ps.value.sum()).round(3).to_string())
    print(f"vets with unknown age: {(ps.b == 'vet_unknown_age').mean():.2%} of rows")
    vb = ["vet_le26", "vet_27_29", "vet_30_32", "vet_33plus"]
    for money in ["value", "cap"]:
        fit(ps, rat, money, "b", "by contract type and age", test=vb)
    # pooled bucket version: non-QB vets <30 vs 30+
    ps["b2"] = np.where(ps.b.isin(["vet_le26", "vet_27_29"]), "vet_u30",
                        np.where(ps.b.isin(["vet_30_32", "vet_33plus"]), "vet_30plus", ps.b))
    fit(ps, rat, "value", "b2", "non-QB vets under 30 vs 30+", test=["vet_u30", "vet_30plus"])
    # 30+ by position group
    ps["b3"] = np.where(ps.b2.isin(["vet_u30", "vet_30plus"]), ps.b2 + "_" + ps.grp, ps.b2)
    fit(ps, rat, "value", "b3", "non-QB vets by age and position group")
    # stale contracts: vets on a deal signed 2+ seasons ago vs this offseason / last
    ps["b4"] = np.where(ps.b.str.startswith("vet_") & ps.vet,
                        np.where(ps.season - ps.signed >= 2, "vet_old_deal", "vet_new_deal"), ps.b)
    fit(ps, rat, "value", "b4", "non-QB vets by contract age", test=["vet_new_deal", "vet_old_deal"])
    ps.to_parquet(OUT / "age_pricing_player_seasons.parquet")


if __name__ == "__main__":
    main()
