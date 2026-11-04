"""College production from free college play-by-play (sportsdataverse / ESPN), 2008-2025.

One row per player, team and season:
  passing:   attempts, completions, yards, TDs, interceptions, sacks taken, EPA
  rushing:   carries, yards, TDs, EPA
  receiving: targets, catches, yards, TDs, EPA
  defense:   sacks (half credit when shared), interceptions, pass breakups, forced fumbles
  games:     games with any recorded involvement
Plus team totals per season, so production can be expressed as a share of the team's.

Players are identified by name only (no IDs in this data); prospects.py matches
them to draft picks by name and school.

Run:  python src/college.py [first_season last_season]
"""
import sys
from pathlib import Path

import polars as pl
import sportsdataverse.cfb as cfb

OUT = Path("data/derived")
COLS = ["game_id", "season", "pos_team", "def_pos_team", "start.pos_team.name", "start.def_pos_team.name",
        "pass_attempt", "completion", "pass_td", "rush", "rush_td", "sack", "target", "EPA",
        "statYardage", "yds_rushed", "yds_receiving", "passer_player_name", "rusher_player_name",
        "receiver_player_name", "sack_player_name", "sack_player_name2", "interception_player_name",
        "pass_breakup_player_name", "fumble_forced_player_name"]


def season_stats(year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = cfb.load_cfb_pbp(seasons=[year])
    d = d.select([c for c in COLS if c in d.columns]).rename(
        {"start.pos_team.name": "off_team", "start.def_pos_team.name": "def_team"})
    # Some seasons have a name column that is entirely empty and loads as a number
    d = d.with_columns([pl.col(c).cast(pl.Utf8) for c in d.columns if c.endswith("_name") or c.endswith("_team")])
    for c in COLS:
        if c not in d.columns and c.endswith("_name"):
            d = d.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
    f = lambda c: pl.col(c).fill_null(False).cast(pl.Boolean)
    num = lambda c: pl.col(c).fill_null(0).cast(pl.Float64)

    passing = (d.filter(pl.col("passer_player_name").is_not_null())
               .group_by(name="passer_player_name", team="off_team")
               .agg(pass_att=f("pass_attempt").sum(), pass_cmp=f("completion").sum(),
                    pass_yds=pl.when(f("completion")).then(num("yds_receiving")).otherwise(0).sum(),
                    pass_td=f("pass_td").sum(), sacked=f("sack").sum(),
                    pass_int=pl.col("interception_player_name").is_not_null().sum(),
                    pass_epa=num("EPA").sum(), g_pass=pl.col("game_id").n_unique()))
    rushing = (d.filter(pl.col("rusher_player_name").is_not_null() & f("rush"))
               .group_by(name="rusher_player_name", team="off_team")
               .agg(rush_att=pl.len(), rush_yds=num("yds_rushed").sum(), rush_td=f("rush_td").sum(),
                    rush_epa=num("EPA").sum(), g_rush=pl.col("game_id").n_unique()))
    receiving = (d.filter(pl.col("receiver_player_name").is_not_null())
                 .group_by(name="receiver_player_name", team="off_team")
                 .agg(targets=pl.len(), rec=f("completion").sum(),
                      rec_yds=pl.when(f("completion")).then(num("yds_receiving")).otherwise(0).sum(),
                      rec_td=f("pass_td").sum(), rec_epa=num("EPA").sum(), g_rec=pl.col("game_id").n_unique()))
    sacks = pl.concat([
        d.filter(pl.col("sack_player_name").is_not_null()).select(
            name="sack_player_name", team="def_team", game_id="game_id",
            v=pl.when(pl.col("sack_player_name2").is_not_null()).then(0.5).otherwise(1.0)),
        d.filter(pl.col("sack_player_name2").is_not_null()).select(
            name="sack_player_name2", team="def_team", game_id="game_id", v=pl.lit(0.5))])
    defense = [sacks.group_by("name", "team").agg(sacks=pl.col("v").sum(), g_sack=pl.col("game_id").n_unique())]
    for col, stat in [("interception_player_name", "def_int"), ("pass_breakup_player_name", "pbu"),
                      ("fumble_forced_player_name", "ff")]:
        defense.append(d.filter(pl.col(col).is_not_null())
                       .group_by(name=col, team="def_team")
                       .agg(pl.len().alias(stat), pl.col("game_id").n_unique().alias(f"g_{stat}")))
    out = passing
    for t in [rushing, receiving] + defense:
        out = out.join(t, on=["name", "team"], how="full", coalesce=True)
    gcols = [c for c in out.columns if c.startswith("g_")]
    out = (out.with_columns(games=pl.max_horizontal(gcols)).drop(gcols)
           .with_columns(pl.exclude("name", "team").fill_null(0), season=pl.lit(year)))

    team = (d.group_by(team="off_team")
            .agg(team_pass_att=f("pass_attempt").sum(),
                 team_pass_yds=pl.when(f("completion")).then(num("yds_receiving")).otherwise(0).sum(),
                 team_rush_yds=pl.when(f("rush")).then(num("yds_rushed")).otherwise(0).sum(),
                 team_rush_att=f("rush").sum(), team_games=pl.col("game_id").n_unique(),
                 team_off_epa_play=num("EPA").mean())
            .with_columns(season=pl.lit(year)))
    team_def = (d.group_by(team="def_team").agg(team_def_epa_play=num("EPA").mean(),
                                                team_sacks=f("sack").sum()))
    return out, team.join(team_def, on="team", how="left")


def main(first=2008, last=2025):
    OUT.mkdir(parents=True, exist_ok=True)
    players, teams = [], []
    for y in range(first, last + 1):
        try:
            p, t = season_stats(y)
        except Exception as e:  # the host has no file for some seasons (2022+ as of Nov 2026)
            print(f"{y}: not available ({type(e).__name__})", flush=True)
            continue
        players.append(p); teams.append(t)
        print(f"{y}: {p.height} player-team rows", flush=True)
    pl.concat(players, how="diagonal_relaxed").write_parquet(OUT / "cfb_player_seasons.parquet")
    pl.concat(teams, how="diagonal_relaxed").write_parquet(OUT / "cfb_team_seasons.parquet")
    print("saved")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(int(x) for x in a)) if a else main()
