"""Match drafted players to college production and build per-position features.

Matching: by normalized name, in the 6 seasons before the draft. If the name
appears at more than one school, keep seasons at schools that look like the
player's listed college (string similarity), so a common name doesn't pull in a
stranger's stats. Transfers with unique names keep all their schools.

Features (final season = last college season; best = best season):
  QB         EPA per dropback (final, best), completion %, yards/attempt, INT rate,
             sack rate, career attempts, rushing yards/game
  RB         share of team rushing yards (final, best), yards/carry, rush EPA/carry,
             share of team receiving yards
  WR, TE     share of team receiving yards (final, best), age at breakout (first season
             with share >= 20% WR / 15% TE), yards/target, receiving EPA/target
  ED, IDL, LB  sacks per team game (final, best), forced fumbles per game
  CB, S      interceptions + pass breakups per team game (final, best)
  T, G, C    none (play-by-play has no individual line stats)
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from outlook import OUT, RAW

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " "))
    return re.sub(r"\s+", " ", SUFFIX.sub("", s)).strip()


def norm_school(s: str) -> str:
    s = str(s).lower().replace("st.", "state").replace("&", "and")
    s = re.sub(r"\(.*?\)", "", s)
    return re.sub(r"[^a-z ]", "", s).strip()


def match(picks: pd.DataFrame, cfb: pd.DataFrame) -> pd.DataFrame:
    """Rows of cfb player-seasons belonging to each pick (adds pick keys)."""
    cfb = cfb.assign(key=cfb.name.map(norm_name), school=cfb.team.map(norm_school))
    by_name = dict(tuple(cfb.groupby("key")))
    out = []
    for _, p in picks.iterrows():
        cand = by_name.get(norm_name(p["name"]))
        if cand is None:
            continue
        cand = cand[(cand.season < p.draft_year) & (cand.season >= p.draft_year - 6)]
        if cand.empty:
            continue
        if cand.school.nunique() > 1:
            sim = cand.school.map(lambda s: SequenceMatcher(None, s, norm_school(p.college)).ratio())
            if sim.max() < 0.6:
                continue
            cand = cand[sim >= 0.6]
        out.append(cand.assign(draft_year=p.draft_year, pick=p.pick))
    return pd.concat(out, ignore_index=True)


def features(picks: pd.DataFrame) -> pd.DataFrame:
    cfb = pd.read_parquet(OUT / "cfb_player_seasons.parquet")
    teams = pd.read_parquet(OUT / "cfb_team_seasons.parquet")
    m = match(picks, cfb).merge(teams, on=["team", "season"], how="left")
    tg = m.team_games.clip(lower=1)
    m["dropbacks"] = m.pass_att + m.sacked
    m["qb_epa_db"] = m.pass_epa / m.dropbacks.clip(lower=1)
    m["rush_share"] = m.rush_yds / m.team_rush_yds.clip(lower=1)
    m["rec_share"] = m.rec_yds / m.team_pass_yds.clip(lower=1)
    m["sacks_pg"] = m.sacks / tg
    m["ff_pg"] = m.ff / tg
    m["cov_pg"] = (m.def_int + m.pbu) / tg
    m["rush_ypg"] = m.rush_yds / tg

    birth = (pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
             .drop_duplicates("pfr_id").set_index("pfr_id").birth_date)
    picks = picks.assign(birth=pd.to_datetime(picks.pfr_id.map(birth), errors="coerce"))
    rows = []
    for (dy, pk), g in m.groupby(["draft_year", "pick"]):
        p = picks[(picks.draft_year == dy) & (picks.pick == pk)].iloc[0]
        g = g.sort_values("season")
        last = g.iloc[-1]
        r = {"draft_year": dy, "pick": pk, "cfb_seasons": g.season.nunique()}
        grp = p.grp
        if grp == "QB":
            q = g[g.dropbacks >= 100]
            lastq = q.iloc[-1] if len(q) else last
            r.update(qb_epa_final=lastq.qb_epa_db, qb_epa_best=q.qb_epa_db.max() if len(q) else np.nan,
                     qb_cmp=lastq.pass_cmp / max(lastq.pass_att, 1), qb_ypa=lastq.pass_yds / max(lastq.pass_att, 1),
                     qb_int_rate=lastq.pass_int / max(lastq.pass_att, 1),
                     qb_sack_rate=lastq.sacked / max(lastq.dropbacks, 1),
                     qb_career_att=g.pass_att.sum(), qb_rush_ypg=lastq.rush_ypg)
        elif grp == "RB":
            r.update(rb_share_final=last.rush_share, rb_share_best=g.rush_share.max(),
                     rb_ypc=last.rush_yds / max(last.rush_att, 1),
                     rb_epa_carry=last.rush_epa / max(last.rush_att, 1), rb_rec_share=g.rec_share.max())
        elif grp in ("WR", "TE"):
            thr = 0.20 if grp == "WR" else 0.15
            hit = g[g.rec_share >= thr]
            bo_age = ((pd.Timestamp(f"{int(hit.season.iloc[0])}-09-01") - p.birth).days / 365.25
                      if len(hit) and pd.notna(p.birth) else 24.0)  # never broke out -> old
            r.update(rec_share_final=last.rec_share, rec_share_best=g.rec_share.max(), breakout_age=bo_age,
                     rec_ypt=last.rec_yds / max(last.targets, 1), rec_epa_tgt=last.rec_epa / max(last.targets, 1))
        elif grp in ("ED", "IDL", "LB"):
            r.update(sacks_pg_final=last.sacks_pg, sacks_pg_best=g.sacks_pg.max(), ff_pg=g.ff_pg.mean())
        elif grp in ("CB", "S"):
            r.update(cov_pg_final=last.cov_pg, cov_pg_best=g.cov_pg.max())
        rows.append(r)
    return pd.DataFrame(rows)


FEATURES = {
    "QB": ["qb_epa_final", "qb_epa_best", "qb_cmp", "qb_ypa", "qb_int_rate", "qb_sack_rate", "qb_career_att", "qb_rush_ypg"],
    "RB": ["rb_share_final", "rb_share_best", "rb_ypc", "rb_epa_carry", "rb_rec_share"],
    "WR": ["rec_share_final", "rec_share_best", "breakout_age", "rec_ypt", "rec_epa_tgt"],
    "TE": ["rec_share_final", "rec_share_best", "breakout_age", "rec_ypt", "rec_epa_tgt"],
    "ED": ["sacks_pg_final", "sacks_pg_best", "ff_pg"],
    "IDL": ["sacks_pg_final", "sacks_pg_best", "ff_pg"],
    "LB": ["sacks_pg_final", "sacks_pg_best", "ff_pg"],
    "CB": ["cov_pg_final", "cov_pg_best"],
    "S": ["cov_pg_final", "cov_pg_best"],
}
LOWER_IS_BETTER = {"breakout_age", "qb_int_rate", "qb_sack_rate"}
