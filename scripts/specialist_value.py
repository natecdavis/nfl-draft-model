"""Kickers and punters: NFL value in points, and what drafting one buys over an undrafted one.

Specialists aren't in the label (PFF doesn't grade them, no pay curve) and too few are on the consensus board to
fit a rank slope, so they get their own path, in points (the unit of team strength in the engine).

STEP 1, NFL value (nflverse play-by-play 2006-25, regular season, all plays):
  Field goals  P(make) = logistic(spline(distance) x era + indoor + wind + cold), fit on all attempts;
               points over expected = 3 x (made - P). Blocked = miss.
  Extra points P(make) by era (distance 20 -> 33 in 2015) x indoor + wind; 1 x (made - P).
  Punts        EPA of the punt play (kicking team's view) on spline(yardline) x era + indoor + wind;
               residual = punt value over expected (includes coverage/returns: team noise, which shows up as
               low year-to-year stability rather than bias).
  Kickoffs are left out: touchback rules changed several times (2011, 2016, 2018, 2024) and coverage dominates.
  Persistence: year-to-year correlation of per-kick value (>= 20 kicks both seasons), all and team changers; signal
  sd = sqrt(cov(t, t+1)) per kick, times kicks per season = sd of true talent in points per season.
STEP 2, draft (classes 2006-2022, years 1-4 = class .. class+3, all seasons played by 2025):
  Value over replacement in points: replacement = per-kick value of mid-season pickups (a kicker's first game for
  a team after week 1, not his rookie season), a street free agent. Per player, years 1-4:
  sum over seasons of (season per-kick value, shrunk to the mean with the step-1 reliability - replacement) x kicks.
  Compare drafted (by round) with undrafted entrants: all who kicked in the NFL, and the best k of each class where
  k = the number drafted that class (the undrafted alternative teams actually compete for).
Usage: python scripts/specialist_value.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nflverse_pbp"
OUT = ROOT / "data/derived/specialist_seasons.parquet"
SEASONS = range(2006, 2026)
CLASSES = (2006, 2022)
COLS = ["season", "season_type", "week", "game_id", "posteam", "play_type", "field_goal_attempt", "field_goal_result",
        "extra_point_attempt", "extra_point_result", "punt_attempt", "kick_distance", "yardline_100", "epa",
        "kicker_player_id", "punter_player_id", "roof", "wind", "temp"]


def load():
    d = pd.concat([pd.read_parquet(RAW / f"play_by_play_{y}.parquet", columns=COLS) for y in SEASONS], ignore_index=True)
    d = d[d.season_type == "REG"].copy()
    d["indoor"] = d.roof.isin(["dome", "closed"]).astype(float)
    d["windy"] = np.where(d.indoor == 1, 0.0, d.wind.fillna(d.wind.median()).clip(0, 40))
    d["cold"] = ((d.indoor == 0) & (d.temp.fillna(60) < 40)).astype(float)
    d["era"] = np.select([d.season < 2015, d.season < 2020], ["a", "b"], "c")
    return d


def fg_xp(d):
    f = d[(d.field_goal_attempt == 1) & d.kicker_player_id.notna() & d.kick_distance.notna()].copy()
    f["made"] = (f.field_goal_result == "made").astype(float)
    m = smf.logit("made ~ bs(kick_distance, df=5) * era + indoor + windy + cold", f).fit(disp=0)
    f["value"] = 3 * (f.made - m.predict(f))
    x = d[(d.extra_point_attempt == 1) & d.kicker_player_id.notna()].copy()
    x["made"] = (x.extra_point_result == "good").astype(float)
    x["post15"] = (x.season >= 2015).astype(float)
    mx = smf.logit("made ~ post15 * (indoor + windy) + C(season)", x).fit(disp=0)
    x["value"] = x.made - mx.predict(x)
    print(f"FG attempts {len(f):,} (make {f.made.mean():.3f}), XP {len(x):,} (make {x.made.mean():.3f})")
    k = pd.concat([f.assign(kind="fg"), x.assign(kind="xp")])
    return k.rename(columns={"kicker_player_id": "pid"})[["pid", "season", "week", "posteam", "kind", "value"]]


def punts(d):
    p = d[(d.punt_attempt == 1) & d.punter_player_id.notna() & d.epa.notna()].copy()
    m = smf.ols("epa ~ bs(yardline_100, df=6) * era + indoor + windy", p).fit()
    p["value"] = p.epa - m.predict(p)
    print(f"punts {len(p):,}; EPA sd {p.epa.std():.2f}, residual sd {p.value.std():.2f}")
    return p.rename(columns={"punter_player_id": "pid"}).assign(kind="punt")[["pid", "season", "week", "posteam", "kind", "value"]]


def seasons(k):
    s = k.groupby(["pid", "season"]).agg(n=("value", "size"), pts=("value", "sum"),
                                         tm=("posteam", lambda x: x.mode().iat[0])).reset_index()
    s["rate"] = s.pts / s.n
    return s


def persistence(s, lab, min_n=20):
    a = s[s.n >= min_n]
    p = a.merge(a.assign(season=a.season - 1), on=["pid", "season"], suffixes=("", "_n"))
    w = np.sqrt(p.n * p.n_n)
    def wcorr(x, y, w):
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        c = np.average((x - mx) * (y - my), weights=w)
        return c / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)), c
    r, cov = wcorr(p.rate, p.rate_n, w)
    mv = p.tm != p.tm_n
    r_mv, _ = wcorr(p.rate[mv], p.rate_n[mv], w[mv]) if mv.sum() > 20 else (np.nan, 0)
    sig = np.sqrt(max(cov, 0))
    typ = s[s.n >= min_n].n.median()
    noise = np.average(s[s.n >= min_n].rate.var() - cov)  # per-season-rate noise variance at typical n
    print(f"  {lab}: pairs {len(p)} (team changers {mv.sum()}); yoy corr {r:.2f}, changers {r_mv:.2f}; "
          f"talent sd {sig:.3f} pts/kick x {typ:.0f} kicks = {sig * typ:.1f} pts/season ({sig * typ / 17:.2f}/game)")
    return dict(sig2=max(cov, 0), noise_n=noise * typ, typ=typ)


def shrink(s, prm):
    # reliability of a season with n kicks: sig2 / (sig2 + noise_per_kick / n)
    rel = prm["sig2"] / (prm["sig2"] + prm["noise_n"] / s.n)
    return s.rate * rel


def replacement(k, s):
    """Per-kick value of mid-season pickups: first game for a team after week 1, not in the player's first season."""
    first = k.groupby(["pid", "season", "posteam"]).week.min().reset_index()
    rook = s.groupby("pid").season.min()
    pu = first[(first.week > 1) & (first.season > first.pid.map(rook))]
    kk = k.merge(pu[["pid", "season", "posteam"]], on=["pid", "season", "posteam"])
    return kk.value.mean(), len(pu), len(kk)


def main():
    pd.set_option("display.width", 200)
    d = load()
    out = {}
    for lab, k in [("K (FG + XP)", fg_xp(d)), ("P (punts)", punts(d))]:
        print(f"\n== {lab} ==")
        s = seasons(k)
        prm = persistence(s, lab)
        s["fg_share"] = np.nan
        repl, npu, nk = replacement(k, s)
        print(f"  replacement (mid-season pickups: {npu} player-team-seasons, {nk} kicks): {repl:+.3f} pts/kick")
        s["est"] = shrink(s, prm)
        s["por"] = (s.est - repl) * s.n  # points over replacement in the season
        s["pos"] = lab[0]
        out[lab[0]] = (s, prm, repl)
    S = pd.concat([v[0] for v in out.values()])
    S.to_parquet(OUT, index=False)
    draft(S)


def draft(S):
    pl = pd.read_parquet(ROOT / "data/raw/players.parquet")
    pl = pl[pl.position.isin(["K", "P"])][["gsis_id", "display_name", "position", "draft_year", "draft_round",
                                             "draft_pick", "rookie_season", "college_name"]]
    dp = pd.read_parquet(ROOT / "data/raw/draft_picks.parquet")
    dp = dp[dp.position.isin(["K", "P"])][["season", "round", "pick", "gsis_id", "pfr_player_name", "position"]]
    print(f"\n== STEP 2: draft, classes {CLASSES[0]}-{CLASSES[1]} ==")
    rows = []
    for pos in ["K", "P"]:
        s = S[S.pos == pos]
        first = s.groupby("pid").season.min()
        # entrants: drafted (draft_picks) or undrafted with an NFL kick; class = draft year / first NFL season
        dr = dp[(dp.position == pos) & dp.season.between(*CLASSES)].rename(columns={"season": "cls"})
        und = pl[(pl.position == pos) & pl.draft_round.isna()][["gsis_id", "display_name", "rookie_season"]]
        # class = rookie season (players list); first season with a kick if missing; veterans who entered before
        # 2006 are excluded (their pre-2006 seasons aren't in the play-by-play)
        und = und.assign(cls=und.rookie_season.fillna(und.gsis_id.map(first))).dropna(subset=["cls"])
        und = und[und.gsis_id.isin(first.index) & (und.gsis_id.map(first) >= und.cls)]
        und = und[und.cls.between(*CLASSES)]
        E = pd.concat([dr.assign(drafted=True, name=dr.pfr_player_name)[["gsis_id", "name", "cls", "round", "pick", "drafted"]],
                       und.assign(drafted=False, name=und.display_name, round=np.nan, pick=np.nan)[["gsis_id", "name", "cls", "round", "pick", "drafted"]]])
        E["cls"] = E.cls.astype(int)
        y = s.merge(E[["gsis_id", "cls"]], left_on="pid", right_on="gsis_id")
        y = y[y.season.between(y.cls, y.cls + 3)]
        a = y.groupby("pid").agg(por=("por", "sum"), kicks=("n", "sum"), raw=("pts", "sum"),
                                 seasons=("n", lambda x: int((x >= (40 if pos == "K" else 40)).sum())))
        E = E.join(a, on="gsis_id").fillna({"por": 0.0, "kicks": 0, "seasons": 0})
        E["pos"] = pos
        rows.append(E)
    E = pd.concat(rows, ignore_index=True)
    E["band"] = np.select([E.drafted & (E["round"] <= 4), E.drafted & (E["round"] == 5), E.drafted & (E["round"] == 6),
                           E.drafted], ["R2-4", "R5", "R6", "R7"], "undrafted (kicked in NFL)")
    print("Years 1-4, points over replacement (shrunk per-kick value x kicks), full-time seasons (>= 40 kicks):")
    for pos in ["K", "P"]:
        e = E[E.pos == pos]
        e = e.assign(per_kick=(e.raw / e.kicks.where(e.kicks >= 40)))
        t = e.groupby("band").agg(n=("por", "size"), mean_pts=("por", "mean"), median=("por", "median"),
                                  raw_per_kick=("per_kick", "mean"),
                                  full_time_seasons=("seasons", "mean"), share_any_ft=("seasons", lambda x: (x > 0).mean()))
        # best-k undrafted per class, k = number drafted that class
        kd = e[e.drafted].groupby("cls").size()
        u = e[~e.drafted].sort_values("por", ascending=False)
        bestk = pd.concat([g.head(kd.get(c, 0)) for c, g in u.groupby("cls")])
        t.loc["undrafted best-k of class (hindsight)"] = [len(bestk), bestk.por.mean(), bestk.por.median(), bestk.per_kick.mean(),
                                                          bestk.seasons.mean(), (bestk.seasons > 0).mean()]
        print(f"\n  {pos}:")
        print(t.round(2).to_string())
        top = e.sort_values("por", ascending=False).head(10)[["name", "cls", "round", "por", "seasons"]]
        print(f"  top 10 {pos} by years 1-4 value:\n" + top.round(1).to_string(index=False))
    E.to_parquet(ROOT / "data/derived/specialist_entrants.parquet", index=False)
    # yardstick: what a typical non-QB pick adds, same unit (team points per game, per season, years 1-4 + option)
    t = pd.read_parquet(ROOT / "data/derived/pick_value_titles.parquet")
    t = t[(t.grp != "QB") & (t.situation == "Middle of the pack")]
    print("\nYardstick, non-QB picks (middle of the pack), avg team points/game added per season over 5 seasons:")
    print(t[t.pick.isin([100, 140, 180, 220, 256])].groupby("pick").pts_per_season.mean().round(3).to_string())


if __name__ == "__main__":
    main()
