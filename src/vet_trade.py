"""Offseason trades with players and picks, in Super Bowl odds (free agency / trades, step 3).

A trade = players and picks we give, players and picks we get. Everything goes onto the same 5-season strength path
(pick_value.baseline_paths) at once, as trade.evaluate does for picks, because gains combine non-linearly on the title
curve.

Players: forecast as in src/veteran.py (comps; data through 2025, so the 2027 contract seasons are the comps' seasons
2-6). Contract from the player's ACTIVE OTC row, season by season from SIGN_SEASON (playing seasons = base salary > 0;
void years only carry bonus proration):
  acquiring team  charged his cap number minus his prorated signing bonus (the old team keeps the proration), with
                  his guaranteed salary as dead money if cut (veteran.team_seasons_sched cut rule)
  trading team    keeping him: full cap number, cut -> remaining guarantees + remaining proration as dead money;
                  trading him: all remaining proration (incl. void years) accelerates into SIGN_SEASON (pre-June 1)
Cap $ -> share of cap: the SIGN_SEASON cap, growing CAP_GROWTH a year (2023-26 averaged ~10%; 7% is a cautious guess).
Picks: trade.package_points (this draft = SIGN_SEASON's draft; future picks by round).
Checked (scripts/traded_player_check.py): offseason-traded veterans 2014-24 delivered 0.94x forecast in season 1
(0.86 to 1.03; starters 0.90, 0.80 to 1.01): no haircut, shown as a caution.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

import fa_market as fm
import veteran as vt

SIGN_SEASON = vt.LAST + 2
SKIP = SIGN_SEASON - vt.LAST - 1
CAP_GROWTH = 1.07
WINDOW = 5


@lru_cache(maxsize=1)
def _active_rows():
    raw = pd.read_parquet(vt.RAW / "contracts.parquet", columns=["player", "gsis_id", "is_active", "year_signed",
                                                                 "season_history", "position", "team"])
    act = raw[raw.is_active == True].sort_values("year_signed").drop_duplicates("gsis_id", keep="last")  # noqa: E712
    pl = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    act["pfr_id"] = act.gsis_id.map(g2p)
    return act.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")


def contract(pfr_id, cap0, season=SIGN_SEASON):
    """Remaining contract from `season`, shares of that season's cap. None if no active contract row."""
    A = _active_rows()
    if pfr_id not in A.index:
        return None
    sh = A.at[pfr_id, "season_history"]
    rows = sorted([x for x in (sh if sh is not None else []) if str(x.get("year", "")).isdigit() and int(x["year"]) >= season],
                  key=lambda x: int(x["year"]))
    f = lambda x, k: float(x.get(k) or 0.0)  # noqa: E731
    play = [x for x in rows if fm.real_season(x)][:WINDOW]  # skip void / placeholder years
    cap = lambda y: cap0 * CAP_GROWTH ** (y - season)  # noqa: E731
    years = [int(x["year"]) for x in play]
    pro = np.array([f(x, "prorated_bonus") / cap(int(x["year"])) for x in play])
    capn = np.array([f(x, "cap_number") / cap(int(x["year"])) for x in play])
    gtd = np.array([f(x, "guaranteed_salary") / cap(int(x["year"])) for x in play])
    played = {int(x["year"]) for x in play}
    void_pro = sum(f(x, "prorated_bonus") / cap(season) for x in rows if int(x["year"]) not in played)
    keep_extra = np.array([pro[t:].sum() + void_pro for t in range(len(play))])  # proration left if cut before t
    base = np.array([f(x, "base_salary") / cap(int(x["year"])) for x in play])
    return {"years": years, "keep_cost": capn, "acq_cost": np.maximum(capn - pro, 0.0), "gtd": gtd, "base": base,
            "pro": pro, "void_pro": float(void_pro),
            "keep_extra": keep_extra, "dead_if_traded": float(pro.sum() + void_pro),
            "team": A.at[pfr_id, "team"], "cap_m": [float(f(x, "cap_number")) for x in play]}


def player_features(pfr_id):
    P = _players()
    if pfr_id not in P.index:
        return None
    r = P.loc[pfr_id]
    return {"grp": r.grp, "age": r.age - SKIP, "v1": r.v1, "v2": r.v2, "g1": r.g1}


@lru_cache(maxsize=1)
def _players():
    d = vt.current_players()
    d["age"] = (pd.Timestamp(f"{SIGN_SEASON}-09-01") - d.born).dt.days / 365.25
    return d.set_index("pfr_id")


def player_points(pfr_id, cap0, side):
    """(K, 5) points for the comps' futures: side 'get' (we acquire him) or 'keep' (we hold him); plus the dead money
    (share of cap) we'd take by trading him away, and the contract."""
    pl, c = player_features(pfr_id), contract(pfr_id, cap0)
    if pl is None or c is None or not c["years"]:
        return None, c
    yrs = len(c["years"])
    V, S = vt.comps().paths(pl, yrs, exclude=False, skip=SKIP)
    if side == "get":
        val, cost, on = vt.team_seasons_sched(V, c["acq_cost"], c["gtd"])
    else:
        val, cost, on = vt.team_seasons_sched(V, c["keep_cost"], c["gtd"], c["keep_extra"])
    pts = np.zeros((len(V), WINDOW))
    pts[:, :yrs] = vt.points(val, S, cost, on, pl["grp"] == "QB")
    return pts, c


def evaluate(sit, give_players=(), get_players=(), give_picks=(), get_picks=(), cap0=299.0, ctx=None, n=20000, seed=0):
    """Change in expected Super Bowls over 5 seasons (share of one title) from the trade, plus each asset alone."""
    import pick_value as pv
    import trade as tr
    from pick_value import COST_RATE
    ctx = ctx or tr.Context()
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    sb = ctx.sb
    b0 = sb(base)
    keep, get = np.zeros((n, WINDOW)), np.zeros((n, WINDOW))
    assets, contracts = [], {}
    for side, ids in (("keep", give_players), ("get", get_players)):
        for p in ids:
            pts, c = player_points(p, cap0, side)
            contracts[(side, p)] = c
            if pts is None:
                assets.append({"asset": p, "side": "give" if side == "keep" else "get", "titles": np.nan,
                               "note": "no contract for this season or no recent value"})
                continue
            draw = pts[rng.integers(0, len(pts), n)]
            if side == "keep":
                keep += draw
                dead = np.zeros((n, WINDOW)); dead[:, 0] = -COST_RATE * c["dead_if_traded"] * 100
                get += dead  # trading him away accelerates his remaining bonus proration
                alone = (sb(base + dead) - sb(base + draw)).sum(axis=1).mean()
            else:
                get += draw
                alone = (sb(base + draw) - b0).sum(axis=1).mean()
            assets.append({"asset": p, "side": "give" if side == "keep" else "get", "titles": alone,
                           "note": ""})
    for side, picks in (("give", give_picks), ("get", get_picks)):
        for pk in picks:
            pts = tr.package_points(ctx, sit, [pk], base, rng)
            (keep if side == "give" else get)[:] += pts
            alone = (sb(base + pts) - b0).sum(axis=1).mean() * (-1 if side == "give" else 1)
            assets.append({"asset": pk, "side": side, "titles": alone, "note": ""})
    t_keep, t_get = sb(base + keep).sum(axis=1), sb(base + get).sum(axis=1)
    d = t_get - t_keep
    return {"change": d.mean(), "se": d.std() / np.sqrt(n), "assets": assets, "contracts": contracts}


def roster(season=SIGN_SEASON):
    """Players under contract for `season` with recent value, for pickers: pfr_id, name, team, group, label."""
    uc = fm.under_contract(season)
    uc = uc[uc.signed & uc.pfr_id.notna()].drop_duplicates("pfr_id")
    P = _players()
    uc = uc[uc.pfr_id.isin(P.index)].copy()
    uc["v1"] = uc.pfr_id.map(P.v1)
    uc["grp"] = uc.pfr_id.map(P.grp)
    uc["label"] = uc.player + " (" + uc.grp + ", " + uc.team.fillna("?") + ")"
    return uc.sort_values("v1", ascending=False).reset_index(drop=True)
