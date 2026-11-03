"""Milestone 1: player outlook v1 — what each draft slot tends to produce, by position.

Outcome tiers for every pick in the 2012-2021 drafts:

  star     - best contract after the rookie deal paid like a top player at his
             position (top 10 for one-per-team spots, top 20 for two-per-team,
             top 24 for WR/CB), measured against veteran contracts only
  starter  - not a star, but played >= 50% of his side's snaps in 2+ seasons
  bust     - neither

Snap data starts in 2013, so 2012 picks are judged from their second season on.

Smoothing: two logistic curves on log(pick), P(star) and P(starter or better),
fit separately so the star curve can fall off faster than the starter curve.
Positions shift the curves up or down but share one slope (tested: fully
separate curves per position overfit and lose to ignoring position).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

RAW = Path("data/raw")
OUT = Path("data/derived")
OUT.mkdir(parents=True, exist_ok=True)

DRAFTS = list(range(2012, 2022))
TIERS = ["bust", "starter", "star"]
START_SHARE, START_SEASONS = 0.50, 2
# Curves use log(pick + 10): flattens the very top, where plain log(pick) overstated
# picks 1-3. Offsets 0/2/5/10/20 tested on rolling drafts; gains stop around 10.
PICK_OFFSET = 10

GROUP = {"LT": "T", "RT": "T", "LG": "G", "RG": "G"}
STAR_RANK = {"QB": 10, "RB": 10, "TE": 10, "C": 10, "WR": 24, "CB": 24,
             "T": 20, "G": 20, "ED": 20, "IDL": 20, "LB": 20, "S": 20}
GROUPS = list(STAR_RANK)


def pos_group(col: str) -> pl.Expr:
    return pl.col(col).replace(GROUP)


def star_thresholds(c: pl.DataFrame) -> pd.Series:
    """Pay (share of cap) of the Nth-highest running veteran contract, per position
    and season. A contract is judged against the market in the year it was signed,
    because pay at some positions (QB especially) has outgrown the cap over time."""
    vets = c.filter(~(pl.col("year_signed") == pl.col("draft_year")).fill_null(False))
    rows = []
    for season in range(2013, 2027):
        running = (vets.filter((pl.col("year_signed") <= season)
                               & (pl.col("year_signed") + pl.col("years") > season))
                   .sort("year_signed").group_by("otc_id").last()
                   .with_columns(grp=pos_group("position")))
        for grp, n in STAR_RANK.items():
            pay = running.filter(pl.col("grp") == grp)["apy_cap_pct"].sort(descending=True)
            rows.append({"season": season, "grp": grp, "star_pay": pay[n - 1]})
    return pd.DataFrame(rows).set_index(["season", "grp"])["star_pay"]


def season_shares(s: pl.DataFrame) -> pl.DataFrame:
    """Share of his side's regular-season snaps, per player and season.
    Share = sum of per-game snap % / games in the season (missed games count as 0)."""
    games = pl.when(pl.col("season") >= 2021).then(17).otherwise(16)
    return (s.filter(pl.col("game_type") == "REG")
            .group_by("pfr_player_id", "season")
            .agg(off=pl.col("offense_pct").sum(), dfn=pl.col("defense_pct").sum())
            .with_columns(share=(pl.max_horizontal("off", "dfn") / games).clip(0, 1))
            .select("pfr_player_id", "season", "share"))


def starter_seasons(s: pl.DataFrame) -> pl.DataFrame:
    """Seasons per player with >= START_SHARE of his side's snaps."""
    return (season_shares(s).filter(pl.col("share") >= START_SHARE)
            .group_by("pfr_player_id").agg(start_seasons=pl.len()))


def label_picks(p, c, s, thr: pd.Series) -> pd.DataFrame:
    picks = p.filter(pl.col("season").is_in(DRAFTS)).select(
        draft_year="season", pick="pick", name="pfr_player_name", pfr_id="pfr_player_id")
    drafted = c.filter(pl.col("draft_year").is_in(DRAFTS) & pl.col("draft_overall").is_not_null())

    # Position at draft time = position on the rookie contract
    rookie = (drafted.filter(pl.col("year_signed") == pl.col("draft_year"))
              .sort("apy").group_by("draft_year", "draft_overall").last()
              .select("draft_year", pick="draft_overall", rookie_pos="position"))

    # Was any later contract star pay for that contract's position?
    later = (drafted.filter(pl.col("year_signed") > pl.col("draft_year"))
             .with_columns(grp=pos_group("position"))
             .join(pl.from_pandas(thr.reset_index()).cast({"season": pl.Int32}),
                   left_on=["year_signed", "grp"], right_on=["season", "grp"], how="left")
             .with_columns(is_star=pl.col("apy_cap_pct") >= pl.col("star_pay"))
             .group_by("draft_year", "draft_overall")
             .agg(any_star=pl.col("is_star").any(), best_pay=pl.col("apy_cap_pct").max())
             .rename({"draft_overall": "pick"}))

    df = (picks.join(rookie, on=["draft_year", "pick"], how="left")
          .join(later, on=["draft_year", "pick"], how="left")
          .join(starter_seasons(s), left_on="pfr_id", right_on="pfr_player_id", how="left")
          .with_columns(grp=pos_group("rookie_pos"),
                        any_star=pl.col("any_star").fill_null(False),
                        start_seasons=pl.col("start_seasons").fill_null(0))
          .with_columns(score=pl.when(pl.col("any_star")).then(2)
                        .when(pl.col("start_seasons") >= START_SEASONS).then(1).otherwise(0))
          .to_pandas())
    df["tier"] = pd.Categorical([TIERS[x] for x in df["score"]], categories=TIERS, ordered=True)
    return df


def _fit(train: pd.DataFrame, by_position: bool):
    """Two logistic curves on log(pick): P(star) and P(starter or better).
    by_position=True lets each position shift its curve up or down but share one
    slope; this beat both 'pick only' and fully separate curves on unseen drafts."""
    d = train.assign(lp=np.log(train["pick"] + PICK_OFFSET), star=(train["score"] == 2).astype(int),
                     plus=(train["score"] >= 1).astype(int))
    rhs = "lp + C(grp)" if by_position else "lp"
    return (smf.logit(f"star ~ {rhs}", d).fit(disp=False),
            smf.logit(f"plus ~ {rhs}", d).fit(disp=False))


def _predict(models, new: pd.DataFrame) -> np.ndarray:
    """Probabilities [bust, starter, star] for rows with columns pick, grp."""
    new = new.assign(lp=np.log(new["pick"].to_numpy(float) + PICK_OFFSET))
    p_star = models[0].predict(new).to_numpy()
    p_plus = np.maximum(models[1].predict(new).to_numpy(), p_star)
    return np.column_stack([1 - p_plus, p_plus - p_star, p_star])


def fit_curves(df: pd.DataFrame) -> pd.DataFrame:
    grid = pd.DataFrame([(g, k) for g in GROUPS for k in range(1, 263)], columns=["grp", "pick"])
    return pd.concat([grid, pd.DataFrame(_predict(_fit(df, True), grid), columns=TIERS)], axis=1)


def holdout_check(df: pd.DataFrame) -> dict:
    """Each draft 2016-2021 predicted from earlier drafts only, vs simpler baselines."""
    ll = lambda pr, y: -np.log(np.clip(pr[np.arange(len(y)), y], 1e-4, 1))
    losses = {"by position + pick": [], "pick only": [], "no info": []}
    rows = []
    for yr in range(2016, 2022):
        tr, te = df[df.draft_year < yr], df[df.draft_year == yr]
        y = te["score"].to_numpy()
        p = _predict(_fit(tr, True), te)
        losses["by position + pick"] += list(ll(p, y))
        losses["pick only"] += list(ll(_predict(_fit(tr, False), te), y))
        flat = tr["score"].value_counts(normalize=True).reindex(range(3)).to_numpy()
        losses["no info"] += list(ll(np.tile(flat, (len(y), 1)), y))
        rows.append(te.assign(p_star=p[:, 2], p_plus=p[:, 1] + p[:, 2], star=y == 2, plus=y >= 1))
    return {"log_loss": {k: float(np.mean(v)) for k, v in losses.items()},
            "n": len(losses["no info"]), "rows": pd.concat(rows)}


SLOTS = ([0, 10, 32, 64, 100, 150, 200, 262],
         ["1-10", "11-32", "33-64", "65-100", "101-150", "151-200", "201+"])


def main():
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    p = pl.read_parquet(RAW / "draft_picks.parquet")
    s = pl.read_parquet(RAW / "snap_counts.parquet")

    thr = star_thresholds(c)
    df = label_picks(p, c, s, thr)
    df.to_parquet(OUT / "pick_outcomes.parquet")
    model_df = df[df.grp.isin(GROUPS)]
    curves = fit_curves(model_df)
    curves.to_parquet(OUT / "outlook_curves.parquet")

    pd.set_option("display.width", 160, "display.max_rows", 200)
    print("Star pay bar (% of cap) by signing year:\n", (thr.unstack() * 100).round(1), "\n")
    print("Left out (K/P/LS/FB):", df.loc[~df.grp.isin(GROUPS), "grp"].value_counts().to_dict(), "\n")
    print("Tier share by draft year:\n", pd.crosstab(df.draft_year, df.tier, normalize="index").round(2), "\n")
    df["slot"] = pd.cut(df.pick, *SLOTS[:1], labels=SLOTS[1])
    print("Tier share by pick range:\n",
          pd.crosstab(df.slot, df.tier, normalize="index", margins=True).round(2), "\n")

    show = curves[curves.pick.isin([1, 5, 15, 32, 64, 100, 150, 220])]
    print("Smoothed STAR chance (%) by position and pick:")
    print((show.pivot(index="grp", columns="pick", values="star") * 100).round(0), "\n")
    show = show.assign(plus=show.star + show.starter)
    print("Smoothed STARTER-OR-BETTER chance (%):")
    print((show.pivot(index="grp", columns="pick", values="plus") * 100).round(0), "\n")

    h = holdout_check(model_df)
    print(f"Rolling test, 2016-21 drafts (n={h['n']}), avg log loss (lower is better):",
          {k: round(v, 3) for k, v in h["log_loss"].items()})
    r = h["rows"].assign(slot=lambda d: pd.cut(d.pick, *SLOTS[:1], labels=SLOTS[1]))
    print("Holdout calibration (predicted vs actual):\n",
          r.groupby("slot", observed=True)[["p_star", "star", "p_plus", "plus"]].mean().round(2))


if __name__ == "__main__":
    main()
