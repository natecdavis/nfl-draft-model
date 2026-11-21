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
H = 6  # seasons forecast (5 contract seasons + 1 when the data ends a season before the signing)
AGE_SD = 2.0
RAW_FEATS = ["age", "v1", "v2f", "g1"]
# the season in progress (scripts/partial_season_test.py): snap share per team game and share of team games played in
# weeks 1-PARTIAL_W of the anchor season, added to the comps' linear shift when the player has them (planning a year
# ahead). Season-2 forecast RMSE 1.344 -> 1.264; players who missed half+ of those weeks no longer forecast at full value.
PARTIAL_W = 11
P_FEATS = ["p_share", "p_gp"]
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
        self.beta, self.beta_p = {}, {}
        has_p = all(c in A.columns for c in P_FEATS)
        for g, G in A.groupby("grp"):
            for k in range(1, H + 1):
                d = G[G[f"y{k}"].notna()]
                X = np.column_stack([np.ones(len(d)), d[RAW_FEATS].to_numpy()])
                self.beta[(g, k)] = np.linalg.lstsq(X, d[f"y{k}"].to_numpy(), rcond=None)[0][1:]
                if has_p:
                    X = np.column_stack([np.ones(len(d)), d[RAW_FEATS + P_FEATS].to_numpy()])
                    self.beta_p[(g, k)] = np.linalg.lstsq(X, d[f"y{k}"].to_numpy(), rcond=None)[0][1:]

    def _shift(self, row, P, g, k):
        """Comp-to-target shift for season k: with the season in progress when the target has it."""
        if self.beta_p and all(pd.notna(row.get(c)) for c in P_FEATS):
            f = RAW_FEATS + P_FEATS
            return (np.array([row[c] for c in f], float) - P[f].to_numpy()) @ self.beta_p[(g, k)]
        return (np.array([row[c] for c in RAW_FEATS], float) - P[RAW_FEATS].to_numpy()) @ self.beta[(g, k)]

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
        return (P[f"y{k}"].to_numpy() + self._shift(row, P, g, k)).clip(min=0), nb

    def comp_age(self, row, need=1):
        """Mean age of the comps used (when far below the player's age, he's older than almost anyone like him)."""
        row, x = self._x(row)
        return float(self.A.loc[self.neighbors(row["grp"], x, need=need), "age"].mean())

    def forecast(self, row, exclude=True):
        return {f"f{k}": self.outcomes(row, k, exclude)[0].mean() for k in range(1, H + 1)}

    def paths(self, row, years, exclude=True, skip=0):
        """(K, years) values and snap shares: whole futures of the comps with all seasons observed, each season
        shifted to the target's features. skip: seasons between the features and the contract's first season
        (1 when the data ends two seasons before it). Share only feeds the small replacement-level credit."""
        row, x = self._x(row)
        g = row["grp"]
        nb = self.neighbors(g, x, row.get("pfr_id") if exclude else None, row.get("Y") if exclude else None,
                            need=years + skip)
        P = self.A.loc[nb]
        ks = range(skip + 1, skip + years + 1)
        V = np.column_stack([(P[f"y{k}"].to_numpy() + self._shift(row, P, g, k)).clip(min=0) for k in ks])
        S = np.column_stack([P[f"s{k}"].to_numpy() for k in ks])
        S = np.where(V > 0, np.maximum(S, 0.05), 0.0).clip(max=1.0)
        return V, S


@lru_cache(maxsize=32)
def partial_snaps(season, weeks=PARTIAL_W):
    """pfr_id -> p_share (snap share per team game) and p_gp (share of team games played), weeks 1..weeks of
    `season`; empty if there are no snap counts for that season."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import roster_need as rn
    f = RAW / f"snap_counts_{season}.parquet"
    sc = pd.read_parquet(f if f.exists() else RAW / "snap_counts.parquet")
    sc = sc[(sc.game_type == "REG") & (sc.season == season) & (sc.week <= weeks)].copy()
    if sc.empty:
        return pd.DataFrame(columns=P_FEATS)
    sc["team"] = sc.team.map(rn.team)
    pct = sc[["offense_pct", "defense_pct"]].max(axis=1).astype(float)
    sc["pct"] = pct / (100 if pct.max() > 1.5 else 1)
    tg = sc.groupby("team").game_id.nunique()
    per = sc.groupby("pfr_player_id").agg(pct=("pct", "sum"), gp=("game_id", "nunique"),
                                          team=("team", lambda x: x.mode().iloc[0]))
    per["tg"] = per.team.map(tg)
    return pd.DataFrame({"p_share": per.pct / per.tg, "p_gp": (per.gp / per.tg).clip(upper=1), "tg": per.tg})


@lru_cache(maxsize=1)
def comps() -> Comps:
    A = pd.read_parquet(OUT / "vet_anchors.parquet")
    pp = []
    for Y in sorted(A.Y.unique()):
        Pt = partial_snaps(int(Y))
        if len(Pt):
            pp.append(Pt[P_FEATS].assign(Y=Y).reset_index(names="pfr_id"))
    if pp:
        A = A.merge(pd.concat(pp), on=["pfr_id", "Y"], how="left")
        A[P_FEATS] = A[P_FEATS].fillna(0.0)  # no snaps in weeks 1-11 of that season
    return Comps(A)


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


def team_seasons_sched(V, cost_y, gtd_y, extra_dead_y=None, theta=CUT_THETA, prorate_y=None):
    """team_seasons for a real contract: cap charge cost_y[t] and guaranteed salary gtd_y[t] per season (shares of cap),
    extra_dead_y[t] = money that turns into dead cap if he's cut before season t (the owner's remaining prorated bonus;
    0 for a team acquiring him by trade, whose old team keeps the bonus). Same cut rule as team_seasons.
    prorate_y[t]: bonus proration inside cost_y[t] (not saved by a cut, so not part of the non-guaranteed pay the cut
    rule weighs); default 0 (the trade and workbench callers)."""
    n, years = V.shape[0], len(cost_y)
    cost_y, gtd_y = np.asarray(cost_y, float), np.asarray(gtd_y, float)
    extra = np.zeros(years) if extra_dead_y is None else np.asarray(extra_dead_y, float)
    pr = np.zeros(years) if prorate_y is None else np.asarray(prorate_y, float)
    val, cost, on = np.zeros((n, years)), np.zeros((n, years)), np.zeros((n, years), bool)
    alive = np.ones(n, bool)
    for t in range(years):
        if t > 0:
            nongtd = max(cost_y[t] - pr[t] - gtd_y[t], 0.0)
            dead = gtd_y[t:].sum() + extra[t]
            # out of football -> gone; otherwise cut only when it saves cap this season (dead money below his cap hit):
            # a team doesn't cut a player whose guarantees (e.g. a fresh extension) would cost more than keeping him
            cut = alive & ((V[:, t - 1] <= 0) | ((V[:, t - 1] < theta * nongtd) & (dead < cost_y[t])))
            cost[cut, t] = dead  # dead money
            alive &= ~cut
        val[alive, t] = V[alive, t]
        cost[alive, t] = cost_y[t]
        on[:, t] = alive
    return val, cost, on


MAX_PRORATION = 5  # CBA: a signing bonus is prorated over at most 5 seasons (void years included)


def structure(base_m, gbase_m, bonus_m, void, cap_m, growth=1.07):
    """A structured contract in shares of each season's cap (cap_m in the first season, growing `growth` a year).
    base_m[t] base salary, gbase_m[t] its guaranteed part, bonus_m signing bonus prorated over the real seasons plus
    `void` void years (at most MAX_PRORATION). Returns the arrays veteran.evaluate(sched=) takes."""
    n = len(base_m)
    P = min(n + int(void), MAX_PRORATION)
    pr = bonus_m / P
    cap = cap_m * growth ** np.arange(n + 1)
    base, gb = np.asarray(base_m, float), np.minimum(np.asarray(gbase_m, float), np.asarray(base_m, float))
    return {"cost_y": (base + pr) / cap[:n], "gtd_y": gb / cap[:n], "prorate_y": np.full(n, pr) / cap[:n],
            # if cut before season t: the proration not yet charged (t..P-1, void years too), in season t's cap
            "extra_y": np.array([pr * (P - t) for t in range(n)]) / cap[:n],
            # proration on void years: dead money the season after the real contract ends
            "void_dead": pr * (P - n) / cap[n]}


def points(val, share, cost, on, qb, rates=None, cost_rate=None):
    """(n, years) points of team strength per season, as pick_value.pick_effect."""
    from pick_value import COST_RATE
    rates = rates or json.load(open(OUT / "bridge.json"))["preseason"]
    rate = rates["qb"] if qb else rates["nonqb"]
    cr = COST_RATE if cost_rate is None else cost_rate
    repl = np.where(on, share, 0.0) * VET_MIN
    charge = np.where(on, cost - VET_MIN, cost)  # dead money after a cut has no roster spot to replace
    return rate * (val - repl) * 100 - cr * charge * 100


ROLLOVER_YIELD = 0.12  # points per 1% of cap that market free agency buys (scripts/vet_value.py, UFA 2015-23)


def evaluate(sit, player: dict, years: int, apy: float, gtd: float, n=20000, seed=0, exclude=False,
             cost_rate=None, skip=0, cap_price="model", paths=None, adj=None, sched=None) -> dict:
    """Titles added over 5 seasons by signing `player` (grp, age, v1, v2, g1) for `years` at `apy` (share of
    cap per season) with `gtd` guaranteed (share of cap, paid first). Situation as pick_value.SITUATIONS.
    cap_price "model": cap charged at COST_RATE points in the season it's used (the draft model's charge);
    "rollover": charged at ROLLOVER_YIELD x the best titles-per-point from that season on (unused cap rolls over and
    is spent at market rates later; scripts/cap_rollover_test.py found real teams don't, so it's a scenario).
    sched: a structured contract (veteran.structure): cap hits, guaranteed salary and bonus proration by season; the
    cut rule of team_seasons_sched; void-year proration charged the season after it ends (inside the 5 seasons)."""
    import pick_value as pv
    sit = pv.SITUATIONS[sit] if isinstance(sit, str) else sit
    years = int(min(len(sched["cost_y"]) if sched is not None else years, 5))
    V, S = paths if paths is not None else comps().paths(player, years, exclude=exclude, skip=skip)
    if sched is None:
        val, cost, on = team_seasons(V, years, apy, gtd)
    else:
        val, cost, on = team_seasons_sched(V, sched["cost_y"], sched["gtd_y"], sched["extra_y"],
                                           prorate_y=sched["prorate_y"])
        if years < 5 and sched.get("void_dead", 0) > 0:  # one more season: void proration as dead money
            k = len(V)
            val, S = np.column_stack([val, np.zeros(k)]), np.column_stack([S[:, :years], np.zeros(k)])
            cost = np.column_stack([cost, np.where(on[:, -1], sched["void_dead"], 0.0)])
            on = np.column_stack([on, np.zeros(k, bool)])
            years += 1
    qb = player["grp"] == "QB"
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    if adj is not None:  # the team's own pending free agents leaving (fa_market.own_fa_adjust)
        base = base - np.asarray(adj)
    curve = _curve()
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)
    idx = rng.integers(0, len(V), n)
    pts = np.zeros((len(V), 5))
    if cap_price == "rollover":
        pts[:, :years] = points(val, S, cost, on, qb, cost_rate=0.0)
        charge = (points(np.zeros_like(val), np.zeros_like(S), cost, on, qb, cost_rate=1.0) * -1).mean(axis=0)
        m = (sb(base + 0.5) - sb(base - 0.5)).mean(axis=0)
        M = np.maximum.accumulate(m[::-1])[::-1]
        d = (sb(base + pts[idx]) - sb(base)).sum(axis=1) - (ROLLOVER_YIELD * charge * M[:years]).sum()
    else:
        pts[:, :years] = points(val, S, cost, on, qb, cost_rate=cost_rate)
        d = (sb(base + pts[idx]) - sb(base)).sum(axis=1)
    return {"titles_added": d.mean(), "se": d.std() / np.sqrt(n), "pts_per_season": pts[:, :years].mean(axis=0),
            "value": val.mean(axis=0), "cost": cost.mean(axis=0), "on_team": on.mean(axis=0),
            "value_pct": np.percentile(val, [10, 50, 90], axis=0)}


def break_even(sit, player, years, gtd_share, skip=0, cap_price="model", lo=0.0, hi=0.30, tol=0.0005, paths=None,
               n=8000, adj=None, cost_rate=None):
    """Highest APY (share of cap) with titles added >= 0, guarantees kept at gtd_share of total value."""
    P = paths if paths is not None else comps().paths(player, int(min(years, 5)), exclude=False, skip=skip)
    f = lambda a: evaluate(sit, player, years, a, gtd_share * a * years, skip=skip, cap_price=cap_price, paths=P,
                           n=n, adj=adj, cost_rate=cost_rate)["titles_added"]
    if f(lo + 0.005) < 0:
        return 0.0
    if f(hi) > 0:
        return hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def current_players(asof=LAST):
    """Players with value in the last two seasons of data: features entering season asof + 1."""
    v = season_table()
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    V = v.set_index(["pfr_id", "season"])
    grp = v.sort_values("season").groupby("pfr_id").group.last()
    rec = v[v.season.between(asof - 1, asof)].pfr_id.unique()
    d = pl[pl.pfr_id.isin(rec)][["pfr_id", "display_name", "birth_date", "latest_team", "status"]].copy()
    d["grp"] = d.pfr_id.map(grp)
    get = lambda p, y, c: float(V[c].get((p, y), 0.0))
    d["v1"] = [get(p, asof, "value_ours") for p in d.pfr_id]
    d["v2"] = [get(p, asof - 1, "value_ours") for p in d.pfr_id]
    d["v3"] = [get(p, asof - 2, "value_ours") for p in d.pfr_id]
    d["g1"] = [get(p, asof, "gfrac") for p in d.pfr_id]
    d["share1"] = [get(p, asof, "share") for p in d.pfr_id]
    d["born"] = pd.to_datetime(d.birth_date, errors="coerce")
    Pt = partial_snaps(asof + 1)  # the season in progress, if there are snap counts for it
    if len(Pt):
        d = d.join(Pt[P_FEATS], on="pfr_id")
        d[P_FEATS] = d[P_FEATS].fillna(0.0)  # no snaps so far this season
    else:
        d[P_FEATS] = np.nan
    return d[d.grp.notna()].sort_values("v1", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=1)
def _curve():
    import pick_value as pv
    from simulate import league_params, load_schedules
    sched = load_schedules()
    return pv.strength_curve(sched, pd.read_parquet(RAW / "teams.parquet"), league_params(sched))


# ---------------- why a player missed games (scripts/availability_split.py) ----------------
INJ_CODES = {"R01", "R48", "R04", "R05"}  # injured reserve (incl. designated to return), PUP, non-football injury


@lru_cache(maxsize=64)
def availability(season, weeks=None):
    """pfr_id -> a_play, a_inj, a_healthy, a_other: shares of his team's games (weeks 1..weeks) he played (any snaps),
    missed injured (IR / PUP / NFI, or out / doubtful on the injury report), missed healthy (on the roster or practice
    squad, no injury designation, no snaps), or other (suspended, other reserve lists, off the roster).
    Weekly rosters and injury reports: nflverse, 2013-2026."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import roster_need as rn
    ro = pd.read_parquet(RAW / "rosters_weekly.parquet")
    ro = ro[(ro.season == season) & ro.pfr_id.notna()]
    if weeks is not None:
        ro = ro[ro.week <= weeks]
    if ro.empty:
        return pd.DataFrame(columns=["a_play", "a_inj", "a_healthy", "a_other"])
    ro = ro.assign(team=ro.team.map(rn.team))
    f = RAW / f"snap_counts_{season}.parquet"
    sc = pd.read_parquet(f if f.exists() else RAW / "snap_counts.parquet")
    sc = sc[(sc.game_type == "REG") & (sc.season == season)]
    if weeks is not None:
        sc = sc[sc.week <= weeks]
    sc = sc.assign(team=sc.team.map(rn.team))
    games = sc.groupby("team").week.agg(lambda w: set(w)).to_dict()  # weeks each team played
    pct = sc[["offense_pct", "defense_pct", "st_pct"]].max(axis=1).astype(float)
    played = set(zip(sc.pfr_player_id[pct > 0], sc.week[pct > 0]))
    inj = pd.read_parquet(RAW / "injuries.parquet")
    inj = inj[(inj.season == season) & inj.report_status.isin(["Out", "Doubtful"])]
    g2p = dict(zip(ro.gsis_id, ro.pfr_id))
    reported = {(g2p.get(g), int(w)) for g, w in zip(inj.gsis_id, inj.week)}
    # one row per player-week: an active-roster row wins over others
    rank = {"ACT": 0, "INA": 1, "RES": 2, "DEV": 3}
    ro = ro.assign(r=ro.status.map(rank).fillna(9)).sort_values("r").drop_duplicates(["pfr_id", "week"])
    main = ro.groupby("pfr_id").team.agg(lambda x: x.mode().iloc[0])
    rows = []
    for p, d in ro.groupby("pfr_id"):
        wk = games.get(main[p], set())
        if not wk:
            continue
        st = dict(zip(d.week, zip(d.status, d.status_description_abbr)))
        n = {"play": 0, "inj": 0, "healthy": 0, "other": 0}
        for w in wk:
            s, code = st.get(w, (None, None))
            if (p, w) in played:
                n["play"] += 1
            elif s == "RES" and code in INJ_CODES or (s in ("ACT", "INA") and (p, w) in reported):
                n["inj"] += 1
            elif s in ("ACT", "INA", "DEV"):
                n["healthy"] += 1
            else:
                n["other"] += 1
        tot = len(wk)
        rows.append((p, n["play"] / tot, n["inj"] / tot, n["healthy"] / tot, n["other"] / tot))
    return pd.DataFrame(rows, columns=["pfr_id", "a_play", "a_inj", "a_healthy", "a_other"]).set_index("pfr_id")
