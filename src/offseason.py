"""Offseason planner (free agency / trades, step 4): what to fill in free agency, what to leave to the draft.

Pieces (all tested elsewhere; this module only composes them):
  budget      free-agency budget = cap + rollover - 2027 commitments - dead money already on the books (not in the data;
              an input) - rookie pool for our picks - filling the roster to 51 at about the minimum - reserve.
              Commitments: every active OTC contract with a 2027 row for the team (cap number; void years included,
              they count on the cap), plus players whose active row has no season history yet (mostly 2026 rookies):
              APY if their contract runs through 2027.
  free agents fa_market.rank: each free agent at his predicted contract, titles added for this team.
  plan        greedy: highest titles per $ first, within the budget, at most MAX_PER_GROUP signings per position group
              (each signing is valued as if he alone fills a minimum-salary slot, so stacking a group overstates it),
              skipping likely-restricted and likely-extended players by default. Then the whole plan is scored together
              on one strength path (titles combine on the curve; veteran.team_seasons for each contract).
  draft side  by position group, a typical player at our first pick (trade.package_points with the pick used at
              that position) next to the best free-agent option: where the draft is likely to fill a need cheaper.
Not modeled: re-signing our own free agents (they're in the ranking like anyone else), trades, cap moves
(restructures, cuts), June 1 designations, the market reacting to us.
"""
import numpy as np
import pandas as pd

import fa_market as fm
import veteran as vt

SIGN_SEASON = vt.LAST + 2
SKIP = SIGN_SEASON - vt.LAST - 1
MAX_PER_GROUP = {"OL": 2, "DB": 2, "WR": 2, "DL": 2, "EDGE": 1, "LB": 1, "RB": 1, "TE": 1, "QB": 1}
ROSTER = 51
FILL_SHARE = 0.0035  # about the 2027 minimum salary, share of cap
# draft position for each veteran pricing group (the best of these at our pick is shown)
DRAFT_POS = {"QB": ["QB"], "RB": ["RB"], "WR": ["WR"], "TE": ["TE"], "OL": ["T", "G", "C"], "DL": ["IDL"],
             "EDGE": ["ED"], "LB": ["LB"], "DB": ["CB", "S"]}


def commitments(team, cap0, season=SIGN_SEASON):
    """Cap committed for `season` ($M) and players counted."""
    import sys
    sys.path.insert(0, str(vt.ROOT / "scripts"))
    import roster_need as rn
    raw = pd.read_parquet(vt.RAW / "contracts.parquet", columns=["player", "gsis_id", "is_active", "year_signed", "years",
                                                                 "apy", "season_history", "team"])
    act = raw[raw.is_active == True].sort_values("year_signed").drop_duplicates(["player", "gsis_id"], keep="last")  # noqa: E712
    nick = dict(zip(*pd.read_parquet(vt.RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS"})
    total, n, rows = 0.0, 0, []
    for nm, sh, ys, yrs, apy, tm in zip(act.player, act.season_history, act.year_signed, act.years, act.apy, act.team):
        hist = [x for x in (sh if sh is not None else []) if str(x.get("year", "")).isdigit()]
        if hist:
            for x in hist:
                if int(x["year"]) == season and x.get("team") and rn.team(nick.get(x["team"], x["team"])) == team:
                    capn = float(x.get("cap_number") or 0.0)
                    total += capn
                    n += (x.get("base_salary") or 0) > 0
                    rows.append((nm, capn))
        elif pd.notna(ys) and pd.notna(yrs) and ys + yrs - 1 >= season and \
                rn.team(nick.get(str(tm).split("/")[-1], str(tm).split("/")[-1])) == team:
            total += float(apy or 0.0)
            n += 1
            rows.append((nm, float(apy or 0.0)))
    return total, n, pd.DataFrame(rows, columns=["player", "cap_m"]).sort_values("cap_m", ascending=False)


def rookie_pool(picks, cap0):
    """Cap ($M) of rookie contracts for our picks (value.rookie_cost_curve, year 1)."""
    import polars as pl
    from value import rookie_cost_curve
    c = pl.read_parquet(vt.RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    cost = rookie_cost_curve(c)
    return float(sum(cost.get(int(min(p, 262)), cost.iloc[-1]) for p in picks) * cap0)


def budget(team, cap0, picks, rollover=0.0, dead=0.0, reserve_share=0.03):
    com, n, lst = commitments(team, cap0)
    rook = rookie_pool(picks, cap0)
    fill = max(ROSTER - n - len(picks), 0) * FILL_SHARE * cap0
    b = cap0 + rollover - com - dead - rook - fill - reserve_share * cap0
    return {"cap": cap0, "rollover": rollover, "commitments": com, "n_signed": n, "dead": dead, "rookies": rook,
            "fill": fill, "reserve": reserve_share * cap0, "budget": b, "committed_players": lst}


def plan(R, budget_m, cap0, max_signings=8, qb_need=False, include_flagged=False, max_per_group=None):
    """Greedy plan from a ranked free-agent table (fa_market.rank output with notes): titles per $ first."""
    mpg = dict(MAX_PER_GROUP if max_per_group is None else max_per_group)
    C = R[(R.titles > 0) & ~R.restricted].copy()
    if not qb_need:
        C = C[C.grp != "QB"]
    if not include_flagged:
        C = C[~C.note.str.contains("likely extended")]
    C["cost_m"] = C.apy_pred * cap0
    C["per_10m"] = C.titles * 100 / C.cost_m * 10  # % of a title per $10M of APY
    C = C.sort_values("per_10m", ascending=False)
    chosen, left, used = [], budget_m, {}
    for r in C.itertuples():
        if len(chosen) >= max_signings:
            break
        if r.cost_m <= left and used.get(r.grp, 0) < mpg.get(r.grp, 1):
            chosen.append(r.Index)
            left -= r.cost_m
            used[r.grp] = used.get(r.grp, 0) + 1
    return C.loc[chosen]


def joint_titles(sit, P, n=20000, seed=0):
    """Titles added by signing every player in P together (one strength path)."""
    import pick_value as pv
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    sb = vt._curve()
    f = lambda x: np.interp(x, sb.strength, sb.super_bowl)  # noqa: E731
    total = np.zeros((n, 5))
    for r in P.itertuples():
        yrs = int(r.yrs_pred)
        pl = {"grp": r.grp, "age": r.age - SKIP, "v1": r.v1, "v2": r.v2, "g1": r.g1}
        V, S = vt.comps().paths(pl, yrs, exclude=False, skip=SKIP)
        val, cost, on = vt.team_seasons(V, yrs, r.apy_pred, r.gshare_pred * r.apy_pred * yrs)
        pts = np.zeros((len(V), 5))
        pts[:, :yrs] = vt.points(val, S, cost, on, r.grp == "QB")
        total += pts[rng.integers(0, len(V), n)]
    return float((f(base + total) - f(base)).sum(axis=1).mean())


def draft_by_group(sit, pick, ctx, qb_need=False, n=8000, seed=0):
    """Titles from a typical player at `pick`, by veteran pricing group (best of its draft positions)."""
    import pick_value as pv
    import trade as tr
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    b0 = ctx.sb(base)
    s = dict(sit)
    out = {}
    for g, poss in DRAFT_POS.items():
        if g == "QB" and not qb_need:
            continue
        best = -np.inf
        for pos in poss:
            pts = tr.package_points(ctx, s, [tr.Pick(int(pick), pos=pos)], base, rng)
            best = max(best, float((ctx.sb(base + pts) - b0).sum(axis=1).mean()))
        out[g] = best
    return out
