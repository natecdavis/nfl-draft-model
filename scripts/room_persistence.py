"""How persistent are team "rooms" (a position group's total snap share in a season)?

The simulator's room check divides each team's occupied snaps by one league-wide median room per position
(2014-25). This tests, out of sample, whether a team's own recent totals predict next season's room better.
Each target season Y (2016-2025) is predicted with coefficients fit only on seasons before Y, using information
known on draft day of Y: the team's totals in Y-1 and Y-2 (as deviations from that season's league median) and
whether the head coach changed between Y-1 and Y (first regular-season game's coach).

Candidates (prediction of the team's total starter-equivalents in group g, season Y):
  pooled   league median over 2014-25 (the current simulator; uses future seasons, so slightly flattered)
  median   league median of seasons < Y
  lag1     median + b * (team's Y-1 deviation)
  lag12    median + b1 * dev(Y-1) + b2 * dev(Y-2)
  lag1_hc  lag1 with a separate b for teams that changed head coach
Offensive line: also scored as one room (T + G + C), since PFF labels move players between the three.

Output: out-of-sample RMSE by group and candidate, paired team-clustered bootstrap CI for lag1 minus median,
and the fitted coefficients on all seasons. Usage: python scripts/room_persistence.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import roster_need as rn  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PFF = {"QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "T", "G": "G", "C": "C", "ED": "ED",
       "DI": "IDL", "LB": "LB", "CB": "CB", "S": "S"}


def totals() -> pd.DataFrame:
    """Team-season-group total starter-equivalents, positions as in the simulator's room check."""
    groups = rn.GROUPS + ["QB"] if "QB" not in rn.GROUPS else rn.GROUPS
    s, _, games, dp = rn.load()
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    ppos = {i: PFF.get(q) for i, q in zip(pl.pfr_id, pl.pff_position) if PFF.get(q)}
    dpos = dict(zip(dp.pfr_player_id, dp.position.map(rn.DRAFT_POS)))
    s["grp"] = [ppos.get(p) or rn.SNAP_POS.get(q) or dpos.get(p) for p, q in zip(s.pfr_player_id, s.position)]
    p = (s.groupby(["season", "team", "pfr_player_id"])
         .agg(pct=("pct", "sum"), grp=("grp", lambda x: x.mode().iloc[0] if x.notna().any() else None)).reset_index())
    p["se"] = p.pct / p.set_index(["season", "team"]).index.map(games)
    t = p[p.grp.isin(groups)].groupby(["season", "team", "grp"]).se.sum().unstack(fill_value=0.0)
    t["OL"] = t[["T", "G", "C"]].sum(axis=1)
    return t.stack().rename("tot").reset_index()


def hc_change() -> dict:
    """(season, team) -> True if the week-1 head coach differs from last season's week-1 head coach."""
    sc = pd.read_parquet(RAW / "schedules.parquet")
    sc = sc[sc.game_type == "REG"].sort_values(["season", "week"])
    rows = pd.concat([sc[["season", "home_team", "home_coach"]].set_axis(["season", "team", "coach"], axis=1),
                      sc[["season", "away_team", "away_coach"]].set_axis(["season", "team", "coach"], axis=1)])
    rows["team"] = rows.team.map(rn.team)
    first = rows.groupby(["season", "team"]).coach.first()
    prev = first.groupby(level="team").shift(1)
    return {k: bool(prev[k] != first[k]) for k in first.index if pd.notna(prev[k])}


def features(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values(["team", "grp", "season"]).copy()
    t["lgmed"] = t.groupby(["season", "grp"]).tot.transform("median")
    t["dev"] = t.tot - t.lgmed
    t["dev1"] = t.groupby(["team", "grp"]).dev.shift(1)
    t["dev2"] = t.groupby(["team", "grp"]).dev.shift(2)
    hc = hc_change()
    t["hc"] = [hc.get((y, tm), False) for y, tm in zip(t.season, t.team)]
    return t


def fit(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Least squares of dev on cols, no intercept (deviations from the season median)."""
    X = d[cols].to_numpy(float)
    return np.linalg.lstsq(X, d.dev.to_numpy(float), rcond=None)[0]


def predict(t: pd.DataFrame, years=range(2016, 2026)) -> pd.DataFrame:
    pooled = t[t.season.between(2014, 2025)].groupby("grp").tot.median()
    out = []
    for g, tg in t.groupby("grp"):
        tg = tg.assign(dev1hc=tg.dev1 * tg.hc, dev1st=tg.dev1 * ~tg.hc)
        for y in years:
            tr = tg[(tg.season < y) & tg.dev1.notna()]
            te = tg[(tg.season == y) & tg.dev1.notna()].copy()
            if te.empty or len(tr) < 30:
                continue
            med = tg[tg.season < y].groupby("season").tot.median().tail(4).mean()  # recent seasons' medians
            te["pooled"] = pooled[g]
            te["median"] = med
            te["lag1"] = med + fit(tr, ["dev1"])[0] * te.dev1
            tr2, b12 = tr[tr.dev2.notna()], None
            b12 = fit(tr2, ["dev1", "dev2"])
            te["lag12"] = med + b12[0] * te.dev1 + b12[1] * te.dev2.fillna(te.dev1)
            bh = fit(tr, ["dev1hc", "dev1st"])
            te["lag1_hc"] = med + bh[0] * te.dev1hc + bh[1] * te.dev1st
            out.append(te)
    return pd.concat(out)


CANDS = ["pooled", "median", "lag1", "lag12", "lag1_hc"]


def rmse_table(P: pd.DataFrame) -> pd.DataFrame:
    r = {c: P.groupby("grp").apply(lambda d, c=c: np.sqrt(((d.tot - d[c]) ** 2).mean())) for c in CANDS}
    T = pd.DataFrame(r)
    T["sd_room"] = P.groupby("grp").tot.std()
    T["cut_lag1_vs_median_%"] = 100 * (1 - T.lag1 / T["median"])
    return T.round(3)


def boot_ci(P: pd.DataFrame, a="lag1", b="median", n=2000, seed=0) -> pd.DataFrame:
    """Team-clustered bootstrap of the difference in mean squared error (a - b), per group."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, d in P.groupby("grp"):
        dd = ((d.tot - d[a]) ** 2 - (d.tot - d[b]) ** 2).groupby(d.team).agg(["sum", "count"])
        teams = dd.index.to_numpy()
        est = dd["sum"].sum() / dd["count"].sum()
        bs = []
        for _ in range(n):
            k = rng.choice(len(teams), len(teams))
            bs.append(dd["sum"].to_numpy()[k].sum() / dd["count"].to_numpy()[k].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append([g, est, lo, hi])
    return pd.DataFrame(rows, columns=["grp", "dMSE", "lo", "hi"]).set_index("grp").round(4)


def main():
    t = features(totals())
    P = predict(t)
    pd.set_option("display.width", 200)
    print(f"out-of-sample team-seasons: {len(P[P.grp == 'WR'])} per group, seasons {P.season.min()}-{P.season.max()}")
    print("\nRMSE (starter-equivalents), predicting each season from earlier seasons only:")
    print(rmse_table(P).to_string())
    print("\nlag1 minus median, mean squared error (negative = lag1 better), team-clustered 95% CI:")
    print(boot_ci(P).to_string())
    d = t[t.season.between(2014, 2025) & t.dev1.notna()]
    rows = []
    for g, dg in d.groupby("grp"):
        b1 = fit(dg, ["dev1"])[0]
        d2 = dg[dg.dev2.notna()]
        b12 = fit(d2, ["dev1", "dev2"])
        bh = fit(dg.assign(a=dg.dev1 * dg.hc, b=dg.dev1 * ~dg.hc), ["a", "b"])
        rows.append([g, b1, b12[0], b12[1], bh[0], bh[1], dg.hc.mean()])
    print("\nfitted on all 2014-25 seasons: share of last season's deviation that carries over")
    print(pd.DataFrame(rows, columns=["grp", "b_lag1", "b1_of_2", "b2_of_2", "b_new_HC", "b_same_HC", "share_new_HC"])
          .set_index("grp").round(2).to_string())


if __name__ == "__main__":
    main()
