"""Betting-line absence test: what is a starter worth over his replacement, by position, in points?

Games 2013-2025 (regular season, closing spread from schedules: spread_line = expected home margin).
Regular starter of a team at game g: in the team's previous up-to-4 games this season (at least 2),
played in >= 75% of them, averaging >= 60% of snaps when playing. Absent: no offensive/defensive snaps in g
(the market knows inactives before close). Long absences count only until the replacement becomes the
regular.

Model (one row per game):
  spread = hfa x (not neutral) + r[home team-season] - r[away team-season]
           + sum_pos b_pos x (absent starters home - away)
Team-season ratings absorb team strength, so b_pos comes from within-season changes in who plays:
the market's estimate of starter minus backup. Same model on the actual margin (noisier; checks whether
the market misjudges absences) and on margin - spread.

Pay version: b scaled by the absent starter's pre-season price (bridge.preseason_prices: latest veteran
contract, or expected price for draft slot while on a rookie deal; % of cap), QB and non-QB, to compare
with the model's 0.19 / 0.34 pts per 1% of cap x snap share, net of the 0.5% replacement.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
RELOC = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
GRP = {"QB": "QB", "RB": "RB", "FB": None, "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
       "OT": "OL", "OG": "OL", "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "OLB": "LB", "ILB": "LB",
       "MLB": "LB", "LB": "LB", "CB": "CB", "S": "S", "SS": "S", "FS": "S", "DB": "CB", "SAF": "S"}
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S"]


CARRY = len(sys.argv) > 1 and sys.argv[1] == "carry"


def absences():
    s = pd.read_parquet(RAW / "snap_counts.parquet")
    s = s[s.game_type == "REG"].copy()
    s["team"] = s.team.replace(RELOC)
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["grp"] = s.position.map(GRP)
    s = s.dropna(subset=["grp"])
    # team game order
    tg = s[["season", "team", "week", "game_id"]].drop_duplicates().sort_values(["season", "team", "week"])
    tg["k"] = tg.groupby(["season", "team"]).cumcount()
    s = s.merge(tg[["game_id", "team", "k"]], on=["game_id", "team"])
    play = s[s.share > 0]
    # player-game grid over each team-season's games
    rows = []
    for (yr, tm), G in play.groupby(["season", "team"]):
        games = tg[(tg.season == yr) & (tg.team == tm)].set_index("k").game_id
        n = len(games)
        P = G.pivot_table(index="pfr_player_id", columns="k", values="share", aggfunc="max").reindex(
            columns=range(n))
        grp = G.groupby("pfr_player_id").grp.agg(lambda x: x.mode().iloc[0])
        A = P.to_numpy()
        keep_avg = np.zeros(len(A))
        for k in range(2, n):
            lo = max(0, k - 4)
            W = A[:, lo:k]
            played = ~np.isnan(W)
            frac = played.mean(axis=1)
            avg = np.where(played.any(axis=1), np.nansum(W, axis=1) / np.maximum(played.sum(axis=1), 1), 0)
            reg = (frac >= 0.75) & (avg >= 0.6)
            if CARRY:  # still out since he was last a regular: keep counting him absent
                if k == 2:
                    out = np.zeros(len(A), bool)
                reg = reg | out
            absent = reg & np.isnan(A[:, k])
            if CARRY:
                out = absent.copy()
            for i in np.where(reg)[0]:
                if CARRY and avg[i] > 0:
                    keep_avg[i] = avg[i]
                rows.append((games[k], tm, yr, P.index[i], grp[P.index[i]],
                             bool(absent[i]), float(keep_avg[i] if CARRY else avg[i])))
    R = pd.DataFrame(rows, columns=["game_id", "team", "season", "pfr", "grp", "absent", "avg_share"])
    return R


def prices():
    import bridge
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    players = pl.read_parquet(RAW / "players.parquet")
    bridge.SEASONS = list(range(2013, 2026))
    p = bridge.preseason_prices(c, players).to_pandas()
    return p.set_index(["pfr_player_id", "season"]).price


def design(sched, A, value_cols):
    g = sched.copy()
    g["hts"] = g.home_team + g.season.astype(str)
    g["ats"] = g.away_team + g.season.astype(str)
    ts = sorted(set(g.hts) | set(g.ats))
    ix = {t: i for i, t in enumerate(ts)}
    X = np.zeros((len(g), len(ts)))
    X[np.arange(len(g)), g.hts.map(ix)] = 1
    X[np.arange(len(g)), g.ats.map(ix)] = -1
    X = X[:, 1:]  # drop one for identification (sum of ratings arbitrary per season handled by differences)
    agg = A.groupby(["game_id", "team"])[value_cols].sum()
    H = agg.reindex(list(zip(g.game_id, g.home_team))).fillna(0).to_numpy()
    W = agg.reindex(list(zip(g.game_id, g.away_team))).fillna(0).to_numpy()
    D = H - W
    hfa = (g.location != "Neutral").astype(float).to_numpy()[:, None]
    M = np.hstack([D, hfa, X])
    names = value_cols + ["hfa"]
    return M, names, g


def fit(y, M, names, groups):
    m = sm.OLS(y, M).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = m.conf_int()
    return pd.DataFrame({"b": m.params[:len(names)], "lo": ci[:len(names), 0], "hi": ci[:len(names), 1]},
                        index=names)


def main():
    A = absences()
    pr = prices()
    A["price"] = [pr.get((p, y), np.nan) for p, y in zip(A.pfr, A.season)]
    print(f"starter-games: {len(A)}; absent {A.absent.sum()} ({A.absent.mean():.1%}); price linked "
          f"{A.price.notna().mean():.0%}")
    print("absences by group:", A[A.absent].grp.value_counts().to_dict())
    for gname in GROUPS:
        A[f"abs_{gname}"] = (A.absent & (A.grp == gname)).astype(float)
    A["price_f"] = A.price.fillna(0.005).clip(lower=0.005) * 100
    A["share_f"] = A.avg_share
    # model-implied points of the absent starter: rate x share x (price - 0.5%)
    A["model_pts"] = np.where(A.absent, np.where(A.grp == "QB", 0.34, 0.19) * A.share_f * (A.price_f - 0.5), 0)
    A["abs_pay_QB"] = np.where(A.absent & (A.grp == "QB"), A.price_f, 0)
    A["abs_pay_nonQB"] = np.where(A.absent & (A.grp != "QB"), A.price_f, 0)
    A["abs_nonQB"] = (A.absent & (A.grp != "QB")).astype(float)

    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    print(f"games with spreads: {len(s)}")

    cols = [f"abs_{g}" for g in GROUPS]
    M, names, g = design(s, A, cols)
    cl = g.home_team.to_numpy()
    res = {}
    for y, lab in [("spread_line", "spread"), ("result", "margin")]:
        res[lab] = fit(g[y].to_numpy(float), M, names, cl)
    res["margin - spread"] = fit((g.result - g.spread_line).to_numpy(float), M, names, cl)
    T = pd.concat(res, axis=1)
    # model-implied average points of an absent starter at each group
    imp = A[A.absent].groupby("grp").model_pts.mean()
    T[("model", "implied")] = [imp.get(n.replace("abs_", ""), np.nan) for n in T.index]
    T[("model", "n_absent")] = [int(A[A.absent].grp.eq(n.replace("abs_", "")).sum()) for n in T.index]
    T[("model", "avg_price")] = [A[A.absent & (A.grp == n.replace("abs_", ""))].price_f.mean() for n in T.index]
    pd.set_option("display.width", 220)
    print("\nPoints per absent starter (home minus away), team-season fixed effects; model = what pick_value's rule "
          "credits that starter per game (rate x share x (pre-season price - 0.5%)):")
    print(T.round(2).to_string())

    # pay-scaled: count + pay of absent starters
    cols2 = ["abs_QB", "abs_pay_QB", "abs_nonQB", "abs_pay_nonQB"]
    M2, n2, g2 = design(s, A, cols2)
    for y in ["spread_line", "result"]:
        print(f"\nPay-scaled ({y}): points per absent starter + per 1% of cap of his pre-season price")
        print(fit(g2[y].to_numpy(float), M2, n2, cl).round(3).to_string())
    print("   model: per 1% of cap x share ~0.9 -> QB 0.31, non-QB 0.17 pts per 1%, minus 0.5% replacement")

    # direct calibration: spread on model-implied points of absent starters
    M3, n3, g3 = design(s, A, ["model_pts"])
    for y in ["spread_line", "result"]:
        print(f"\nCalibration ({y} on model-implied points lost, 1.0 = model right):")
        print(fit(g3[y].to_numpy(float), M3, n3, cl).round(3).to_string())
    tag = "_carry" if CARRY else ""
    A.to_parquet(OUT / f"absence_starters{tag}.parquet")
    T.to_csv(OUT / f"absence_effects{tag}.csv")


if __name__ == "__main__":
    main()
