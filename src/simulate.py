"""Milestone 3: team strength -> playoff and Super Bowl odds.

Team strength is one number: points per game better than an average team.
A game's margin = home rating - away rating + home-field edge + noise (normal).

Estimated from 2002-2025 regular seasons (the current 32-team, 8-division era):
  - game noise (spread of margins around the ratings)
  - home-field edge (it has shrunk over time; recent seasons are used for scenarios)
  - how spread out true team strength is, once luck is removed
  - how much true strength carries over from one season to the next

The season simulator plays a real schedule, picks division winners and wild
cards, seeds the playoffs (6 teams per conference before 2020, 7 after), and plays
the bracket with re-seeding. Tiebreakers are random (real ones are complicated
and matter little for odds). Ratings can differ per simulated season, so later
steps can feed in uncertain draft outcomes.

Checks against history:
  1. Playoff games: regular-season ratings vs actual winners and vs the betting line
  2. Champions: how highly rated real champions were, vs how highly rated
     simulated champions are
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/derived")
SEASONS = list(range(2002, 2026))
RECENT = list(range(2021, 2026))  # for the home-field edge used in scenarios
RELOCATED = {"STL": "LA", "SD": "LAC", "OAK": "LV"}  # same franchise, current code


def load_schedules() -> pd.DataFrame:
    s = pd.read_parquet(RAW / "schedules.parquet")
    for col in ["home_team", "away_team"]:
        s[col] = s[col].replace(RELOCATED)
    return s


# ---------- estimation ----------

def season_ratings(g: pd.DataFrame, teams: list[str]) -> tuple[pd.Series, float, float, np.ndarray]:
    """Least-squares ratings from one season's regular-season margins.
    Returns ratings (mean 0), home edge, residual sd, and each rating's sampling variance."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(g)
    X = np.zeros((n, len(teams) + 1))
    X[np.arange(n), g.home_team.map(idx)] = 1
    X[np.arange(n), g.away_team.map(idx)] = -1
    X[:, -1] = (g.location != "Neutral").astype(float)
    # Sum-to-zero constraint as an extra heavily weighted row
    Xc = np.vstack([X, np.r_[np.ones(len(teams)), 0] * 100])
    yc = np.r_[g.result.to_numpy(float), 0]
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    resid = g.result.to_numpy(float) - X @ beta
    sd = resid.std(ddof=len(teams))
    cov = sd ** 2 * np.linalg.pinv(Xc.T @ Xc)
    return pd.Series(beta[:-1], index=teams), beta[-1], sd, np.diag(cov)[:-1]


def league_params(sched: pd.DataFrame) -> dict:
    reg = sched[(sched.game_type == "REG") & sched.season.isin(SEASONS) & sched.result.notna()]
    rows, rat = [], {}
    for yr, g in reg.groupby("season"):
        teams = sorted(set(g.home_team) | set(g.away_team))
        r, hfa, sd, v = season_ratings(g, teams)
        rat[yr] = r
        rows.append({"season": yr, "hfa": hfa, "sd": sd, "obs_var": r.var(), "noise_var": v.mean()})
    p = pd.DataFrame(rows).set_index("season")
    true_var = (p.obs_var - p.noise_var).mean()
    reliability = true_var / p.obs_var.mean()
    # Year-to-year carryover of TRUE strength = observed correlation / reliability
    pairs = [(rat[y][t], rat[y + 1][t]) for y in SEASONS[:-1] for t in rat[y].index]
    a = np.array(pairs)
    carry = np.corrcoef(a[:, 0], a[:, 1])[0, 1] / reliability
    return {"game_sd": p.sd.mean(), "hfa_recent": p.loc[RECENT, "hfa"].mean(), "hfa_by_season": p.hfa,
            "true_sd": np.sqrt(true_var), "reliability": reliability, "carryover": carry,
            "ratings": rat}


# ---------- simulation ----------

def _play(r, home, away, hfa, sd, rng):
    """Vectorized games. r: (N, T) ratings; home/away: (N,) team indices. Returns winners."""
    i = np.arange(len(home))
    margin = r[i, home] - r[i, away] + hfa + sd * rng.standard_normal(len(home))
    return np.where(margin > 0, home, away)


def simulate(r: np.ndarray, teams: list[str], conf: dict, div: dict, sched: pd.DataFrame,
             hfa: float, sd: float, n_playoff: int, rng) -> dict:
    """Simulate N seasons. r: (N, T) ratings. sched: regular-season games (home, away, neutral).
    Returns per-team arrays (T,) of playoff, conf-title and Super Bowl chances."""
    N, T = r.shape
    idx = {t: i for i, t in enumerate(teams)}
    h = sched.home_team.map(idx).to_numpy()
    a = sched.away_team.map(idx).to_numpy()
    edge = np.where(sched.location.to_numpy() == "Neutral", 0.0, hfa)
    margin = r[:, h] - r[:, a] + edge + sd * rng.standard_normal((N, len(h)))
    wins = np.zeros((N, T))
    np.add.at(wins, (slice(None), h), (margin > 0))
    np.add.at(wins, (slice(None), a), (margin < 0))
    key = wins + rng.random((N, T)) * 0.5  # random tiebreak

    made = np.zeros(T); conf_win = np.zeros(T); sb_win = np.zeros(T)
    champs = []
    for c in ["AFC", "NFC"]:
        ct = np.array([idx[t] for t in teams if conf[t] == c])
        divs = sorted({div[t] for t in teams if conf[t] == c})
        dw = np.stack([np.array([idx[t] for t in teams if div[t] == d])[np.argmax(
            key[:, [idx[t] for t in teams if div[t] == d]], axis=1)] for d in divs], axis=1)  # (N, 4)
        dw = np.take_along_axis(dw, np.argsort(-np.take_along_axis(key, dw, 1), axis=1), 1)
        rest_key = key[:, ct].copy()
        for j in range(4):
            rest_key[ct[None, :] == dw[:, [j]]] = -1
        wc = ct[np.argsort(-rest_key, axis=1)[:, : n_playoff - 4]]
        seeds = np.concatenate([dw, wc], axis=1)  # (N, n_playoff): seed 1 first
        np.add.at(made, seeds.ravel(), 1)

        # Wild-card round: byes for top (8 - n_playoff) seeds; rest paired high vs low
        byes = 8 - n_playoff
        alive = [np.full(N, s) for s in range(byes)]
        lo, hi = byes, n_playoff - 1
        while lo < hi:
            w = _play(r, seeds[:, lo], seeds[:, hi], hfa, sd, rng)
            alive.append(np.where(w == seeds[:, lo], lo, hi)); lo += 1; hi -= 1
        # Later rounds: re-seed, best remaining seed hosts worst remaining
        alive = np.stack(alive, axis=1)
        while alive.shape[1] > 1:
            alive = np.sort(alive, axis=1)
            k = alive.shape[1]
            nxt = []
            for j in range(k // 2):
                hs, as_ = alive[:, j], alive[:, k - 1 - j]
                ht, at = seeds[np.arange(N), hs], seeds[np.arange(N), as_]
                w = _play(r, ht, at, hfa, sd, rng)
                nxt.append(np.where(w == ht, hs, as_))
            alive = np.stack(nxt, axis=1)
        champ = seeds[np.arange(N), alive[:, 0]]
        np.add.at(conf_win, champ, 1)
        champs.append(champ)
    sb = _play(r, champs[0], champs[1], 0.0, sd, rng)  # neutral site
    np.add.at(sb_win, sb, 1)
    return {"playoffs": made / N, "conf": conf_win / N, "sb": sb_win / N, "sb_winner": sb}


# ---------- checks against history ----------

def check_playoff_games(sched, params):
    po = sched[(sched.game_type != "REG") & sched.season.isin(SEASONS) & sched.result.notna()].copy()
    po["pred"] = [params["ratings"][s][h] - params["ratings"][s][a]
                  + (0 if loc == "Neutral" else params["hfa_by_season"][s])
                  for s, h, a, loc in zip(po.season, po.home_team, po.away_team, po.location)]
    po["pred_shrunk"] = po.pred * params["reliability"]
    from scipy.stats import norm
    sd = params["game_sd"]
    y = (po.result > 0).astype(float)
    out = {}
    for name, col in [("raw ratings", "pred"), ("luck-adjusted ratings", "pred_shrunk"), ("betting line", "spread_line")]:
        p = np.clip(norm.cdf(po[col] / sd), 1e-3, 1 - 1e-3)
        out[name] = {"brier": np.mean((p - y) ** 2), "picked winner": np.mean((p > 0.5) == (y == 1))}
    out["n"] = len(po)
    return pd.DataFrame({k: v for k, v in out.items() if k != "n"}).T.round(3), out["n"]


def check_champions(sched, params, teams_df, n=4000, seed=0):
    """For each season: simulate it with that year's luck-adjusted ratings. Compare how
    real champions ranked by rating with how simulated champions rank."""
    rng = np.random.default_rng(seed)
    conf = dict(zip(teams_df.team_abbr, teams_df.team_conf))
    div = dict(zip(teams_df.team_abbr, teams_df.team_division))
    real_rank, sim_rank, ll = [], [], []
    for yr in SEASONS:
        r = params["ratings"][yr] * params["reliability"]
        teams = list(r.index)
        g = sched[(sched.season == yr) & (sched.game_type == "REG")]
        sb = sched[(sched.season == yr) & (sched.game_type == "SB")].iloc[0]
        champ = sb.home_team if sb.result > 0 else sb.away_team
        rank = r.rank(ascending=False)
        res = simulate(np.tile(r.to_numpy(), (n, 1)), teams, conf, div, g,
                       params["hfa_by_season"][yr], params["game_sd"], 6 if yr < 2020 else 7, rng)
        real_rank.append(rank[champ])
        sim_rank += list(rank.iloc[res["sb_winner"]].to_numpy())
        ll.append(np.log(max(res["sb"][teams.index(champ)], 1e-4)))
    bins = [0, 1, 3, 5, 10, 32]
    labels = ["#1", "#2-3", "#4-5", "#6-10", "#11+"]
    tab = pd.DataFrame({"real champions": pd.Series(pd.cut(real_rank, bins, labels=labels)).value_counts(normalize=True),
                        "simulated champions": pd.Series(pd.cut(sim_rank, bins, labels=labels)).value_counts(normalize=True)})
    return tab.sort_index().round(2), -np.mean(ll), np.log(32)


# ---------- generic league ----------

def title_curve(params, teams_df, sched, levels, n=20000, seed=1):
    """Chance one team wins the Super Bowl at each strength level, when the other 31
    teams are drawn from the league's normal spread of true strength."""
    rng = np.random.default_rng(seed)
    g = sched[(sched.season == 2025) & (sched.game_type == "REG")]
    teams = sorted(set(g.home_team) | set(g.away_team))
    conf = dict(zip(teams_df.team_abbr, teams_df.team_conf))
    div = dict(zip(teams_df.team_abbr, teams_df.team_division))
    me = 0  # any team; schedules are roughly balanced
    rows = []
    for x in levels:
        r = rng.normal(0, params["true_sd"], (n, len(teams)))
        r -= r.mean(axis=1, keepdims=True)
        r[:, me] = x
        res = simulate(r, teams, conf, div, g, params["hfa_recent"], params["game_sd"], 7, rng)
        rows.append({"strength": x, "playoffs": res["playoffs"][me], "conf_title": res["conf"][me],
                     "super_bowl": res["sb"][me]})
    return pd.DataFrame(rows)


def main():
    sched = load_schedules()
    teams_df = pd.read_parquet(RAW / "teams.parquet")
    params = league_params(sched)

    print(f"Game noise (sd of margin): {params['game_sd']:.1f} pts")
    print(f"Home-field edge: {params['hfa_by_season'].loc[2002:2006].mean():.1f} pts in 2002-06, "
          f"{params['hfa_recent']:.1f} in 2021-25")
    print(f"Spread of true team strength (sd): {params['true_sd']:.1f} pts "
          f"(a season's point differential is {params['reliability']:.0%} skill, rest luck)")
    print(f"Carryover of true strength to next season: {params['carryover']:.2f}\n")

    tab, n = check_playoff_games(sched, params)
    print(f"Check 1 - playoff games 2002-25 (n={n}), predicted from regular-season ratings:\n{tab}\n")
    tab, ll, ll0 = check_champions(sched, params, teams_df)
    print("Check 2 - where champions ranked in luck-adjusted rating:\n", tab)
    print(f"  avg log loss on actual champion: {ll:.2f} (knowing nothing: {ll0:.2f})\n")

    curve = title_curve(params, teams_df, sched, levels=[-6, -3, 0, 2, 4, 6, 8, 10, 12])
    curve.to_parquet(OUT / "title_curve.parquet")
    print("Generic league, one team's odds by true strength (pts/game vs average):")
    print(curve.assign(**{c: (curve[c] * 100).round(1) for c in ["playoffs", "conf_title", "super_bowl"]})
          .to_string(index=False))
    pd.Series({k: v for k, v in params.items() if np.isscalar(v)}).to_json(OUT / "league_params.json")


if __name__ == "__main__":
    main()
