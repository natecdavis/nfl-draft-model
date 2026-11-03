"""Milestone 0: confirm the free data needed for the pick-value chart exists."""
import nflreadpy as nfl
import polars as pl

out = "data/raw"
picks = nfl.load_draft_picks()
picks.write_parquet(f"{out}/draft_picks.parquet")
print("draft picks:", picks.shape, picks["season"].min(), "-", picks["season"].max())
print(picks.columns)

contracts = nfl.load_contracts()
contracts.write_parquet(f"{out}/contracts.parquet")
print("\ncontracts:", contracts.shape)
print(contracts.columns)

games = nfl.load_schedules()
games.write_parquet(f"{out}/schedules.parquet")
print("\ngames:", games.shape, games["season"].min(), "-", games["season"].max())

# Second-contract label feasibility: drafted players with a later, non-rookie deal
c = contracts.filter(pl.col("draft_year").is_not_null() & (pl.col("year_signed") > pl.col("draft_year")))
print("\ncontracts signed after draft year:", c.height,
      "| distinct players:", c["player"].n_unique())
print(c.select("player","position","draft_year","draft_overall","year_signed","apy_cap_pct").head(5))
