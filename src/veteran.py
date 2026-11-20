"""What a veteran on a given contract is worth in Super Bowl odds, on the draft model's yardstick.

Free agency / trades, step 1 (Nov 2026). Tests: scripts/vet_value.py (forecast + past signings),
scripts/vet_currency_check.py (veterans and rookie-deal players buy the same strength per 1% of curve value).

Season value: snap share x veteran pay curve(share, darko quality) x games fraction, this project's units (share
of cap) - the same pricing as a draft pick's seasons (scripts/vet_season_values.py).

1. Forecast (Comps): the K nearest past "anchors" (player entering a season) in the same pricing group, by age,
   value last season (v1), the season before (v2) and games fraction last season (g1). Each comp's next five
   seasons, shifted by a per-group linear slope times the feature gap (so comps drawn mostly from below a top
   player don't drag him to the middle), are the possible futures. Zeros (cut and unsigned, retired, hurt) count.
2. Contract (team_seasons): flat APY for `years` seasons, guarantees paid first. Before each later season the team
   cuts him if last season's value < CUT_THETA x the non-guaranteed part of next season's pay; once cut (or
   retired: no value) he adds nothing and the guarantees left are charged as dead money that season.
3. Points per season, as pick_value.pick_effect: rate x (value - share x VET_MIN) - COST_RATE x (cap charge -
   VET_MIN while on the roster), rate = bridge QB / non-QB. Titles: pick_value's baseline paths and title curve.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
LAST = 2025
K = 60
H = 5  # seasons forecast
AGE_SD = 2.0
RAW_FEATS = ["age", "v1", "v2f", "g1"]
VET_MIN = 0.005
CUT_THETA = 0.5  # fit to UFA signings' retention (scripts/vet_value.py part C)


def season_table():
    v = pd.read_parquet(OUT / "vet_season_values.parquet")
    return v.sort_values("value_ours").drop_duplicates(["pfr_id", "season"], keep="last")


def anchors(v):
    """One row per (player, season Y) with value in Y-1 or Y-2: features known before Y, values (y1..y5) and
    snap shares (s1..s5) in seasons Y..Y+4 (any team; NaN after LAST)."""
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    born = dict(zip(pl.pfr_id, pd.to_datetime(pl.birth_date, errors="coerce")))
    val = v.set_index(["pfr_id", "season"])
    V, G, S = val.value_ours.to_dict(), val.gfrac.to_dict(), (val.share * val.gfrac).to_dict()
    grp = v.sort_values("season").groupby("pfr_id").group.last().to_dict()  # latest pricing group
    first = v.groupby("pfr_id").season.min().to_dict()
    rows = []
    for p in v.pfr_id.unique():
        b = born.get(p)
        for Y in range(2014, LAST + 1):
            v1, v2 = V.get((p, Y - 1), 0.0), V.get((p, Y - 2), np.nan if Y - 2 < 2013 else 0.0)
            if not (v1 > 0 or (v2 == v2 and v2 > 0)):
                continue
            age = (pd.Timestamp(f"{Y}-09-01") - b).days / 365.25 if b is not None and pd.notna(b) else np.nan
            r = {"pfr_id": p, "Y": Y, "grp": grp[p], "age": age, "v1": v1, "v2": v2,
                 "g1": G.get((p, Y - 1), 0.0), "exp": Y - first[p]}
            for k in range(1, H + 1):
                y = Y + k - 1
                r[f"y{k}"] = V.get((p, y), 0.0) if y <= LAST else np.nan
                r[f"s{k}"] = S.get((p, y), 0.0) if y <= LAST else np.nan
            rows.append(r)
    A = pd.DataFrame(rows)
    A["age"] = A.age.fillna(A.groupby("grp").age.transform("median"))
    A["v2f"] = A.v2.fillna(A.v1)  # 2014 anchors have no Y-2 season
    return A


class Comps:
    """Nearest comparable anchors within a pricing group."""

    def __init__(self, A):
        self.A = A
        self.scale = A.groupby("grp").v1.std().to_dict()
        # local-linear correction: each comp's outcome is shifted by beta x (target features - comp features),
        # beta from a per-group, per-horizon linear fit, so comps drawn mostly from one side (few players are as
        # good as a top player) don't pull the forecast toward the middle
        self.beta = {}
        for g, G in A.groupby("grp"):
            for k in range(1, H + 1):
                d = G[G[f"y{k}"].notna()]
                X = np.column_stack([np.ones(len(d)), d[RAW_FEATS].to_numpy()])
                self.beta[(g, k)] = np.linalg.lstsq(X, d[f"y{k}"].to_numpy(), rcond=None)[0][1:]

    def feats(self, d, g):
        s = self.scale[g]
        return np.column_stack([d.age / AGE_SD, 1.5 * d.v1 / s, d.v2f / s, d.g1 / 0.5])

    def neighbors(self, g, x, exclude_player=None, exclude_Y=None, k=K, need=1):
        """Indices (into self.A) of the k nearest anchors in group g with horizon `need` observed."""
        P = self.A[(self.A.grp == g) & self.A[f"y{need}"].notna()]
        if exclude_player is not None:
            P = P[P.pfr_id != exclude_player]
        if exclude_Y is not None:
            P = P[P.Y != exclude_Y]
        F = self.feats(P, g)
        d = ((F - x) ** 2).sum(axis=1)
        idx = np.argpartition(d, min(k, len(d) - 1))[:k]
        return P.index[idx]

    def _x(self, row):
        row = {**row, "v2f": row.get("v2f", row["v2"] if row.get("v2") == row.get("v2") else row["v1"])}
        return row, self.feats(pd.DataFrame([row]), row["grp"])[0]

    def outcomes(self, row, k, exclude=True, need=None):
        """Comps' season-k values, shifted to the target's features (clipped at 0)."""
        row, x = self._x(row)
        g = row["grp"]
        nb = self.neighbors(g, x, row.get("pfr_id") if exclude else None, row.get("Y") if exclude else None,
                            need=need or k)
        P = self.A.loc[nb]
        shift = (np.array([row[f] for f in RAW_FEATS], float) - P[RAW_FEATS].to_numpy()) @ self.beta[(g, k)]
        return (P[f"y{k}"].to_numpy() + shift).clip(min=0), nb

    def forecast(self, row, exclude=True):
        return {f"f{k}": self.outcomes(row, k, exclude)[0].mean() for k in range(1, H + 1)}

    def paths(self, row, years, exclude=True):
        """(K, years) values and snap shares: whole futures of the comps with all `years` seasons observed,
        each season shifted to the target's features. Share is scaled with the value shift (same role, better
        or worse play keeps share; only used for the small replacement-level credit)."""
        row, x = self._x(row)
        g = row["grp"]
        nb = self.neighbors(g, x, row.get("pfr_id") if exclude else None, row.get("Y") if exclude else None,
                            need=years)
        P = self.A.loc[nb]
        dx = np.array([row[f] for f in RAW_FEATS], float) - P[RAW_FEATS].to_numpy()
        V = np.column_stack([(P[f"y{k}"].to_numpy() + dx @ self.beta[(g, k)]).clip(min=0) for k in range(1, years + 1)])
        S = np.column_stack([P[f"s{k}"].to_numpy() for k in range(1, years + 1)])
        S = np.where(V > 0, np.maximum(S, 0.05), 0.0).clip(max=1.0)
        return V, S


@lru_cache(maxsize=1)
def comps() -> Comps:
    return Comps(pd.read_parquet(OUT / "vet_anchors.parquet"))


def team_seasons(V, years, apy, gtd, theta=CUT_THETA):
    """Apply the contract to value paths V (n, years): returns value to the team, cap charged, on-roster flag."""
    n = V.shape[0]
    val, cost, on = np.zeros((n, years)), np.zeros((n, years)), np.zeros((n, years), bool)
    alive = np.ones(n, bool)
    for t in range(years):
        remaining_gtd = max(gtd - apy * t, 0.0)  # guarantees not yet paid at the start of season t+1
        if t > 0:
            nongtd = apy - min(remaining_gtd, apy)
            cut = alive & ((V[:, t - 1] <= 0) | (V[:, t - 1] < theta * nongtd))
            cost[cut, t] = remaining_gtd  # dead money
            alive &= ~cut
        val[alive, t] = V[alive, t]
        cost[alive, t] = apy
        on[:, t] = alive
    return val, cost, on


def points(val, share, cost, on, qb, rates=None, cost_rate=None):
    """(n, years) points of team strength per season, as pick_value.pick_effect."""
    from pick_value import COST_RATE
    rates = rates or json.load(open(OUT / "bridge.json"))["preseason"]
    rate = rates["qb"] if qb else rates["nonqb"]
    cr = COST_RATE if cost_rate is None else cost_rate
    repl = np.where(on, share, 0.0) * VET_MIN
    charge = np.where(on, cost - VET_MIN, cost)  # dead money after a cut has no roster spot to replace
    return rate * (val - repl) * 100 - cr * charge * 100


def evaluate(sit, player: dict, years: int, apy: float, gtd: float, n=20000, seed=0, exclude=False,
             cost_rate=None) -> dict:
    """Titles added over 5 seasons by signing `player` (grp, age, v1, v2, g1) for `years` at `apy` (share of
    cap per season) with `gtd` guaranteed (share of cap, paid first). Situation as pick_value.SITUATIONS."""
    import pick_value as pv
    from simulate import league_params, load_schedules
    sit = pv.SITUATIONS[sit] if isinstance(sit, str) else sit
    years = int(min(years, H))
    V, S = comps().paths(player, years, exclude=exclude)
    val, cost, on = team_seasons(V, years, apy, gtd)
    pts = np.zeros((len(V), 5))
    pts[:, :years] = points(val, S, cost, on, player["grp"] == "QB", cost_rate=cost_rate)
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    curve = _curve()
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)
    draw = pts[rng.integers(0, len(pts), n)]
    d = (sb(base + draw) - sb(base)).sum(axis=1)
    return {"titles_added": d.mean(), "se": d.std() / np.sqrt(n), "pts_per_season": pts[:, :years].mean(axis=0),
            "value": val.mean(axis=0), "cost": cost.mean(axis=0), "on_team": on.mean(axis=0)}


@lru_cache(maxsize=1)
def _curve():
    import pick_value as pv
    from simulate import league_params, load_schedules
    sched = load_schedules()
    return pv.strength_curve(sched, pd.read_parquet(RAW / "teams.parquet"), league_params(sched))
