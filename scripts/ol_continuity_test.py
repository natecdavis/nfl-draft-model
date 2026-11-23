"""Offensive line continuity: does a line that has played together play better than its talent says?

Outside work (SIS, Vigderman Nov 2024: 82 groups with 500+ snaps together since 2018) found 'not much of anything'
once line quality is accounted for; fantasy-side trackers (RotoViz) report correlations without talent controls.
Specified before looking (Nov 23 2026).

GAME LEVEL (REG 2014-2025; snap counts start 2013, so 2013 only seeds shared history).
  Starting five = the team's top 5 OL (T/G/C/OL/G-T/C-G/G-C) by offensive snaps in the game.
  Continuity (PRIMARY) = mean over the 10 pairs in the five of prior games both started for this team (any season
    since 2013), per 17 games. SECONDARY measures: starters new vs the previous game (in-season, weeks 2+);
    prior games this exact five started together (log1p).
  Outcome (PRIMARY): offensive EPA/play (pass + run plays, no special teams). SECONDARY: dropback EPA, sack rate per
    dropback, run EPA/play, points scored, margin minus closing spread (does the market miss it).
  Model: outcome ~ continuity + OL talent (mean darko_pff talent_pre of the five; unmatched = 10th pct of that
    season's OL) + QB talent (game's main passer) + team-season FE + opponent-season FE + week FE, clustered by
    team. Team-season FE means identification is within team-season: shuffles from injuries, benchings.
  Caveat stated up front: a backup's talent_pre is shrunk toward the prior, so the talent control may understate
    what is lost when a starter goes down. That pushes the continuity term UP; a null is the robust reading.
FOLLOW-UP (after looking): first-scrimmage-play line, see first_play_followup.
SEASON LEVEL (2014-2025): returning OL snap share (this season's OL snaps by players with 300+ OL snaps for the
  team last season) -> offensive rating (synergy_test.ratings 'off') given 9 talent groups + season FE.
"""
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from synergy_test import ratings

RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
RELOC = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
OLPOS = {"T", "G", "C", "OL", "G/T", "C/G", "G/C", "T/G", "OT", "OG"}
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


def fives():
    sc = pd.read_parquet(RAW / "snap_counts.parquet")
    sc = sc[(sc.game_type == "REG") & sc.position.isin(OLPOS) & (sc.offense_snaps > 0)].copy()
    sc["team"] = sc.team.replace(RELOC)
    sc = sc.sort_values(["game_id", "team", "offense_snaps"], ascending=[True, True, False])
    top = sc.groupby(["game_id", "team"]).head(5)
    g = top.groupby(["game_id", "team", "season", "week"]).pfr_player_id.apply(lambda x: tuple(sorted(x))).reset_index()
    g = g[g.pfr_player_id.str.len() == 5].rename(columns={"pfr_player_id": "five"})
    return g.sort_values(["team", "season", "week"]).reset_index(drop=True)


def continuity(g):
    rows = []
    for team, d in g.groupby("team", sort=False):
        pair, exact, prev, prev_season = {}, {}, None, None
        for r in d.itertuples():
            pairs = list(itertools.combinations(r.five, 2))
            same_season = prev_season == r.season
            rows.append({"game_id": r.game_id, "team": team,
                         "pair_games": np.mean([pair.get(p, 0) for p in pairs]) / 17,
                         "exact_prior": exact.get((r.season, r.five), 0),
                         "new_vs_prev": len(set(r.five) - set(prev)) if same_season else np.nan})
            for p in pairs:
                pair[p] = pair.get(p, 0) + 1
            exact[(r.season, r.five)] = exact.get((r.season, r.five), 0) + 1
            prev, prev_season = r.five, r.season
    return g.merge(pd.DataFrame(rows), on=["game_id", "team"])


def talent_lookup():
    s = pd.read_parquet(OUT / "darko_pff.parquet")[["player_id", "season", "g", "talent_pre", "snaps"]]
    p = pd.read_parquet(RAW / "players.parquet")[["gsis_id", "pfr_id", "pff_id"]]
    p["pff_id"] = pd.to_numeric(p.pff_id, errors="coerce")
    s = s.merge(p.dropna(subset=["pff_id"]), left_on="player_id", right_on="pff_id")
    s = s.sort_values("snaps").drop_duplicates(["pfr_id", "season"], keep="last")
    return s


def play_outcomes():
    out = []
    for f in sorted(glob.glob(str(RAW / "nflverse_pbp/play_by_play_*.parquet"))):
        y = int(f[-12:-8])
        if y < 2014:
            continue
        cols = ["game_id", "season", "season_type", "posteam", "pass", "rush", "epa", "sack", "qb_dropback",
                "passer_player_id", "special"]
        d = pd.read_parquet(f, columns=cols)
        d = d[(d.season_type == "REG") & ((d["pass"] == 1) | (d.rush == 1)) & (d.special != 1) & d.epa.notna()]
        d["posteam"] = d.posteam.replace(RELOC).replace({"JAC": "JAX"})
        drop = d.qb_dropback == 1
        a = d.groupby(["game_id", "posteam"]).agg(epa=("epa", "mean"), plays=("epa", "size")).reset_index()
        b = d[drop].groupby(["game_id", "posteam"]).agg(pass_epa=("epa", "mean"), sack_rate=("sack", "mean"),
                                                         qb=("passer_player_id", lambda x: x.mode().iloc[0]
                                                             if x.notna().any() else None)).reset_index()
        c = d[~drop].groupby(["game_id", "posteam"]).agg(run_epa=("epa", "mean")).reset_index()
        out.append(a.merge(b, how="left").merge(c, how="left"))
    return pd.concat(out).rename(columns={"posteam": "team"})


def run(lab, f, D, term):
    m = smf.ols(f, D).fit(cov_type="cluster", cov_kwds={"groups": D.team})
    ci = m.conf_int().loc[term]
    print(f"   {lab:52s} n={len(D):5d} | {term} {m.params[term]:+.4f} ({ci[0]:+.4f} to {ci[1]:+.4f}, p {m.pvalues[term]:.2f})")
    return m


def game_level():
    g = continuity(fives())
    g = g[g.season >= 2014]
    T = talent_lookup()
    ol = T[T.g == "OL"]
    q10 = ol.groupby("season").talent_pre.quantile(0.10)
    tl = ol.set_index(["pfr_id", "season"]).talent_pre
    ex = g[["game_id", "team", "season", "five"]].explode("five")
    ex["t"] = [tl.get((p, s), np.nan) for p, s in zip(ex.five, ex.season)]
    ex["miss"] = ex.t.isna()
    ex["t"] = ex.t.fillna(ex.season.map(q10))
    g = g.merge(ex.groupby(["game_id", "team"]).agg(ol_t=("t", "mean"), ol_miss=("miss", "sum")).reset_index())
    po = play_outcomes()
    g = g.merge(po, on=["game_id", "team"])
    qt = T[T.g == "QB"].set_index(["gsis_id", "season"]).talent_pre
    qmin = T[T.g == "QB"].groupby("season").talent_pre.quantile(0.10)
    g["qb_t"] = [qt.get((q, s), np.nan) for q, s in zip(g.qb, g.season)]
    g["qb_t"] = g.qb_t.fillna(g.season.map(qmin))
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.result.notna()]
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(RELOC)
    sg = pd.concat([s.assign(team=s.home_team, opp=s.away_team, pts=s.home_score, ms=s.result - s.spread_line),
                    s.assign(team=s.away_team, opp=s.home_team, pts=s.away_score, ms=-(s.result - s.spread_line))])
    g = g.merge(sg[["game_id", "team", "opp", "pts", "ms"]], on=["game_id", "team"])
    g["ts"] = g.team + g.season.astype(str)
    g["os"] = g.opp + g.season.astype(str)
    g["log_exact"] = np.log1p(g.exact_prior)
    print(f"GAME LEVEL: team-games {len(g)} ({g.season.min()}-{g.season.max()}); unmatched OL talent "
          f"{g.ol_miss.sum() / (5 * len(g)):.1%} of slots")
    print(f"   pair games together/17: mean {g.pair_games.mean():.2f}, sd {g.pair_games.std():.2f}; "
          f"within team-season sd {(g.pair_games - g.groupby('ts').pair_games.transform('mean')).std():.2f}")
    print(f"   new starters vs previous game: {g.new_vs_prev.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    fe = "+ ol_t + qb_t + C(ts) + C(os) + C(week)"
    print("   PRIMARY (EPA/play per +17 shared games per pair)")
    m = run("offense EPA/play", f"epa ~ pair_games {fe}", g, "pair_games")
    print(f"     (for scale: OL talent coef {m.params['ol_t']:+.4f}/pt; within-ts sd of ol_t "
          f"{(g.ol_t - g.groupby('ts').ol_t.transform('mean')).std():.2f})")
    print("   SECONDARY outcomes")
    run("dropback EPA", f"pass_epa ~ pair_games {fe}", g, "pair_games")
    run("sack rate", f"sack_rate ~ pair_games {fe}", g, "pair_games")
    run("run EPA", f"run_epa ~ pair_games {fe}", g, "pair_games")
    run("points scored", f"pts ~ pair_games {fe}", g, "pair_games")
    run("margin minus spread", f"ms ~ pair_games {fe}", g.dropna(subset=["ms"]), "pair_games")
    print("   SECONDARY measures (offense EPA/play)")
    run("starters new vs previous game (per starter)", f"epa ~ new_vs_prev {fe}", g.dropna(subset=["new_vs_prev"]),
        "new_vs_prev")
    run("sack rate, new vs previous game", f"sack_rate ~ new_vs_prev {fe}", g.dropna(subset=["new_vs_prev"]),
        "new_vs_prev")
    run("log1p prior games this exact five", f"epa ~ log_exact {fe}", g, "log_exact")
    run("no team-season FE (talent + season + opp-season)",
        "epa ~ pair_games + ol_t + qb_t + C(season) + C(os) + C(week)", g, "pair_games")
    g.drop(columns=["five"]).to_parquet(OUT / "ol_continuity_games.parquet")
    return g


def season_level():
    sc = pd.read_parquet(RAW / "snap_counts.parquet")
    sc = sc[(sc.game_type == "REG") & sc.position.isin(OLPOS)].copy()
    sc["team"] = sc.team.replace(RELOC)
    ps = sc.groupby(["season", "team", "pfr_player_id"]).offense_snaps.sum().reset_index()
    last = ps[ps.offense_snaps >= 300].assign(season=lambda x: x.season + 1)[["season", "team", "pfr_player_id"]]
    last["ret"] = 1
    ps = ps.merge(last, how="left").fillna({"ret": 0})
    r = ps.assign(w=ps.ret * ps.offense_snaps).groupby(["season", "team"])[["w", "offense_snaps"]].sum()
    r = (r.w / r.offense_snaps * 10).rename("ret_share").reset_index()
    s = pd.read_parquet(OUT / "darko_pff.parquet")
    s["w"] = s.talent_pre * s.snaps
    t = s.groupby(["season", "tm", "g"])[["w", "snaps"]].sum()
    t = (t.w / t.snaps).unstack("g")[GROUPS]
    t.columns = [f"t_{x}" for x in GROUPS]
    t = t.reset_index().rename(columns={"tm": "team"})
    D = r.merge(t, on=["season", "team"]).merge(ratings(), on=["season", "team"])
    D = D[D.season.between(2014, 2025)]
    tal = " + ".join(f"t_{x}" for x in GROUPS)
    print(f"\nSEASON LEVEL: team-seasons {len(D)}; returning OL snap share mean {D.ret_share.mean() * 10:.0f}% "
          f"(sd {D.ret_share.std() * 10:.0f})")
    print("   PRIMARY (offensive pts/game per +10 points of returning OL snap share)")
    run("off rating", f"off ~ ret_share + {tal} + C(season)", D, "ret_share")
    print("   SECONDARY")
    run("off rating, team FE", f"off ~ ret_share + {tal} + C(season) + C(team)", D, "ret_share")
    run("off rating, no talent control", "off ~ ret_share + C(season)", D, "ret_share")
    run("net rating", f"net ~ ret_share + {tal} + C(season)", D, "ret_share")


def first_play_followup(g):
    """FOLLOW-UP (added after the primary results, Nov 23 2026). The snap-based 'new vs previous game' term
    (-0.0078 EPA/play, p 0.05; margin minus spread -0.76, p 0.01) counts in-game injury replacements as new starters.
    Redo it with the five OL on each team's first scrimmage play (participation data 2016+), which is mostly
    known before kickoff, and keep the residual in-game change as a separate term."""
    print("\nFOLLOW-UP: first-play line (2016-2025)")
    g = g.copy()
    g = g[g.season >= 2016]
    r = pd.read_parquet(ROOT / 'data/raw/rosters_weekly.parquet')
    r = r[r.position.isin(['OL','T','G','C'])][['season','gsis_id']].drop_duplicates()
    olset = set(zip(r.season, r.gsis_id))
    out=[]
    for y in range(2016,2026):
        pa = pd.read_parquet(ROOT / f'data/raw/nflverse_pbp/pbp_participation_{y}.parquet',columns=['nflverse_game_id','play_id','possession_team','offense_players'])
        pb = pd.read_parquet(ROOT / f'data/raw/nflverse_pbp/play_by_play_{y}.parquet',columns=['game_id','play_id','pass','rush','season_type'])
        pb = pb[(pb.season_type=='REG')&((pb['pass']==1)|(pb.rush==1))]
        d = pa.merge(pb, left_on=['nflverse_game_id','play_id'], right_on=['game_id','play_id']).sort_values(['game_id','play_id'])
        d['ol'] = [tuple(sorted(p for p in str(s).split(';') if (y,p) in olset)) for s in d.offense_players]
        d = d[d.ol.str.len()==5]
        f = d.groupby(['game_id','possession_team']).head(1)
        out.append(f[['game_id','possession_team','ol']].assign(season=y))
    F = pd.concat(out).rename(columns={'possession_team':'team'})
    F['team']=F.team.replace(RELOC)
    F = F.merge(g[['game_id','team','week']], on=['game_id','team']).sort_values(['team','season','week'])
    rows=[]
    for team,d in F.groupby('team',sort=False):
        prev=None; ps=None
        for x in d.itertuples():
            same = ps==x.season and prev is not None
            rows.append(dict(game_id=x.game_id,team=team,pre_new=len(set(x.ol)-set(prev)) if same else np.nan))
            prev=x.ol; ps=x.season
    G = g.merge(pd.DataFrame(rows),on=['game_id','team']).dropna(subset=['pre_new','new_vs_prev'])
    print("games", len(G), "first-play line new vs prev first-play line:", G.pre_new.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("corr(pre_new, snap-based new)", round(G.pre_new.corr(G.new_vs_prev),2))
    fe="+ ol_t + qb_t + C(ts) + C(os) + C(week)"
    for y,lab in [('epa','offense EPA/play'),('pass_epa','dropback EPA'),('sack_rate','sack rate'),('run_epa','run EPA'),('pts','points'),('ms','margin minus spread')]:
        run(f"pre-game new starters -> {lab}", f"{y} ~ pre_new {fe}", G.dropna(subset=[y]), "pre_new")
    G['ingame']=G.new_vs_prev-G.pre_new
    run("both: in-game residual change -> EPA", f"epa ~ pre_new + ingame {fe}", G, "ingame")
    run("both: in-game residual change -> margin-spread", f"ms ~ pre_new + ingame {fe}", G.dropna(subset=['ms']), "ingame")


if __name__ == "__main__":
    first_play_followup(game_level())
    season_level()
