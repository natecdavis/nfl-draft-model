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
import zlib

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
                                                                 "apy", "season_history", "team", "contract_history"])
    act = raw[raw.is_active == True].sort_values("year_signed").drop_duplicates(["player", "gsis_id"], keep="last")  # noqa: E712
    nick = dict(zip(*pd.read_parquet(vt.RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS"})
    total, n, rows = 0.0, 0, []
    for nm, sh, ys, yrs, apy, tm, ch in zip(act.player, act.season_history, act.year_signed, act.years, act.apy,
                                            act.team, act.contract_history):
        hist = [x for x in (sh if sh is not None else []) if str(x.get("year", "")).isdigit()]
        end = fm.fresh_deal_end(ch)
        if hist:
            for x in hist:
                if int(x["year"]) == season and x.get("team") and rn.team(nick.get(x["team"], x["team"])) == team:
                    if fm.real_season(x, end):
                        capn = float(x.get("cap_number") or 0.0)
                        n += 1
                        rows.append((nm, capn))
                    else:  # the deal voids: all remaining proration accelerates into this season (dead money)
                        capn = sum(float(y.get("prorated_bonus") or 0.0) for y in hist
                                   if int(y["year"]) >= season and not fm.real_season(y, end))
                        rows.append((nm + " (void years: dead money)", capn))
                    total += capn
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
        pl = {"grp": r.grp, "age": r.age - SKIP, "v1": r.v1, "v2": r.v2, "g1": r.g1,
              "p_share": getattr(r, "p_share", np.nan), "p_gp": getattr(r, "p_gp", np.nan)}
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


# ---------------- cap moves: cuts and restructures, and scoring any mix of moves together ----------------
SUGGEST_CUT = 0.0002  # suggest a cut when it adds at least 0.02% of a title on its own (he's paid well over his value)


def roster_moves(team, sit, cap0, n=4000, seed=0):
    """Cut and restructure options for every player under contract with `team` in SIGN_SEASON with recent value.
    Returns (table, arrays): arrays[pfr] = {"keep", "cut", "restr"} -> (K, 5) points per comp future.
      cut (before June 1): dead = remaining guaranteed salary + all remaining proration (void years too), all in
        SIGN_SEASON; saved in SIGN_SEASON = his cap number - dead; later seasons save his whole cap number.
      restructure: his SIGN_SEASON base salary above the minimum becomes a bonus prorated over his remaining playing
        seasons (n >= 2; no void years added): saves amount x (1 - 1/n) now, adds amount / n to each later season
        and to the dead money if he's cut later."""
    import pick_value as pv
    import vet_trade as vtr
    from pick_value import COST_RATE
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    curve = vt._curve()
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)  # noqa: E731
    R = vtr.roster()
    ours = R[R.team == team]
    # CBA: a drafted player's rookie contract can't be renegotiated (extended or restructured) until after his third
    # regular season - for SIGN_SEASON, draft classes up to SIGN_SEASON - 3 (2024 for 2027). Undrafted players aren't
    # restricted here (the rule is written for drafted rookie contracts; not verified for undrafted deals).
    dy = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").draft_year
    too_early = {p for p in ours.pfr_id if pd.notna(dy.get(p)) and dy.get(p) > SIGN_SEASON - 3}
    EXT = fm.predict_extension(ours.pfr_id) if len(ours) else pd.DataFrame()
    rows, arrays = [], {}
    for r in ours.itertuples():
        pl, c = vtr.player_features(r.pfr_id), vtr.contract(r.pfr_id, cap0)
        if pl is None or c is None or not c["years"]:
            continue
        yrs = len(c["years"])
        ext = EXT.loc[r.pfr_id] if r.pfr_id in EXT.index and r.pfr_id not in too_early else None
        n_new = int(np.clip(ext.yrs_pred, 1, 4)) if ext is not None else 0
        n_tot = min(yrs + n_new, 5)  # seasons inside the 5-season window
        # one set of comp futures, long enough for the extension, so keep / cut / restructure / extend are paired
        V, S = vt.comps().paths(pl, n_tot, exclude=False, skip=SKIP)
        qb = pl["grp"] == "QB"

        def pts_for(cost_y, extra_y, gtd_y=None):
            m = len(cost_y)
            val, cost, on = vt.team_seasons_sched(V[:, :m], cost_y, c["gtd"] if gtd_y is None else gtd_y, extra_y)
            out = np.zeros((len(V), 5))
            out[:, :m] = vt.points(val, S[:, :m], cost, on, qb)
            return out
        keep = pts_for(c["keep_cost"], c["keep_extra"])
        dead = float(c["gtd"].sum() + c["pro"].sum() + c["void_pro"])
        cut = np.zeros_like(keep)
        cut[:, 0] = -COST_RATE * dead * 100
        restr, amount, per = None, 0.0, 0.0
        if yrs >= 2 and r.pfr_id not in too_early:
            amount = max(float(c["base"][0]) - FILL_SHARE, 0.0)
            per = amount / yrs
            cost_y = c["keep_cost"].copy()
            cost_y[0] -= amount - per
            cost_y[1:] += per
            pro_new = c["pro"] + per
            extra = np.array([pro_new[t:].sum() + c["void_pro"] for t in range(yrs)])
            restr = pts_for(cost_y, extra)
        # extension: n_new seasons after his deal at the predicted APY (guarantees on the first new seasons), and his
        # SIGN_SEASON base above the minimum turned into a bonus spread over all seasons in the window (the usual way
        # an extension lowers the current cap hit)
        extp, e_apy, e_gtd, e_saved, e_later = None, 0.0, 0.0, 0.0, 0.0
        if n_new and n_tot > yrs:
            e_apy = float(ext.apy_pred)
            m_new = n_tot - yrs
            g_total = float(ext.gshare_pred) * e_apy * n_new
            gnew = np.array([min(e_apy, max(g_total - e_apy * i, 0.0)) for i in range(m_new)])
            conv = max(float(c["base"][0]) - FILL_SHARE, 0.0)
            per_c = conv / n_tot
            cost_y = np.concatenate([c["keep_cost"], np.full(m_new, e_apy)])
            cost_y[0] -= conv - per_c
            cost_y[1:] += per_c
            gtd_y = np.concatenate([c["gtd"], gnew])
            pro_new = np.concatenate([c["pro"], np.zeros(m_new)]) + per_c
            extra = np.array([pro_new[t:].sum() + c["void_pro"] for t in range(n_tot)])
            extp = pts_for(cost_y, extra, gtd_y)
            e_gtd, e_saved, e_later = g_total, conv - per_c, per_c
        idx = rng.integers(0, len(keep), n)
        # the team's baseline strength already includes him as he is: a move adds only its change (new - keep)
        alone = lambda new: float((sb(base + new[idx] - keep[idx]) - sb(base)).sum(axis=1).mean())  # noqa: E731
        arrays[r.pfr_id] = {"keep": keep, "cut": cut, "restr": restr, "ext": extp}
        rows.append({"pfr_id": r.pfr_id, "player": r.player, "grp": pl["grp"], "age": pl["age"] + SKIP,
                     "years_left": yrs, "cap_now": c["keep_cost"][0] * cap0, "dead_cut": dead * cap0,
                     "saved_cut": (c["keep_cost"][0] - dead) * cap0, "titles_cut": alone(cut), "renegotiable": r.pfr_id not in too_early,
                     "restr_amount": amount * cap0, "saved_restr": (amount - per) * cap0, "later_restr": per * cap0,
                     "titles_restr": alone(restr) if restr is not None else np.nan,
                     "ext_years": n_new if extp is not None else 0, "ext_apy": e_apy * cap0, "ext_gtd": e_gtd * cap0,
                     "saved_ext": e_saved * cap0, "later_ext": e_later * cap0,
                     "titles_ext": alone(extp) if extp is not None else np.nan})
    return pd.DataFrame(rows), arrays


def suggest(T, base_budget):
    """Suggested cuts (cutting adds titles on its own and saves cap) and, only if the budget is still negative,
    restructures (largest savings first, age < 32, 2+ seasons left) until it isn't."""
    cuts = set(T[(T.titles_cut > SUGGEST_CUT) & (T.saved_cut > 0)].pfr_id)
    b = base_budget + T[T.pfr_id.isin(cuts)].saved_cut.sum()
    restr = set()
    if b < 0:
        for r in T[~T.pfr_id.isin(cuts) & (T.saved_restr > 0) & (T.age < 32)].sort_values("saved_restr", ascending=False).itertuples():
            if b >= 0:
                break
            restr.add(r.pfr_id)
            b += r.saved_restr
    return cuts, restr


def evaluate_moves(sit, arrays, cuts, restrs, P, n=20000, seed=1, exts=()):
    """Titles added (share of one) by doing all the moves together, and by each kind alone."""
    import pick_value as pv
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    curve = vt._curve()
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)  # noqa: E731
    before, after = np.zeros((n, 5)), np.zeros((n, 5))
    parts = {"cuts": np.zeros((n, 5)), "restructures": np.zeros((n, 5)), "extensions": np.zeros((n, 5)),
             "signings": np.zeros((n, 5))}
    for p in sorted(set(cuts) | set(restrs) | set(exts)):
        a = arrays[p]
        # same draws for a player whatever else is toggled (seeded by player), so options compare like for like
        idx = np.random.default_rng(zlib.crc32(str(p).encode()) + seed).integers(0, len(a["keep"]), n)
        k = a["keep"][idx]
        kind = "cuts" if p in cuts else ("extensions" if p in exts else "restructures")
        new = a["cut"][idx] if kind == "cuts" else (a["ext"][idx] if kind == "extensions" else a["restr"][idx])
        after += new - k  # baseline strength already includes him as he is
        parts[kind] += new - k
    for r in P.itertuples():
        yrs = int(r.yrs_pred)
        pl = {"grp": r.grp, "age": r.age - SKIP, "v1": r.v1, "v2": r.v2, "g1": r.g1,
              "p_share": getattr(r, "p_share", np.nan), "p_gp": getattr(r, "p_gp", np.nan)}
        V, S = vt.comps().paths(pl, yrs, exclude=False, skip=SKIP)
        val, cost, on = vt.team_seasons(V, yrs, r.apy_pred, r.gshare_pred * r.apy_pred * yrs)
        pts = np.zeros((len(V), 5))
        pts[:, :yrs] = vt.points(val, S, cost, on, r.grp == "QB")
        d = pts[np.random.default_rng(zlib.crc32(str(r.pfr_id).encode()) + seed + 7).integers(0, len(V), n)]
        after += d
        parts["signings"] += d
    total = float((sb(base + after) - sb(base + before)).sum(axis=1).mean())
    alone = {k: float((sb(base + v) - sb(base)).sum(axis=1).mean()) for k, v in parts.items()}
    return total, alone


# ---------------- optimizer: best combination of cuts, restructures, extensions and signings ----------------
def signing_points(r):
    """(K, 5) points for signing free agent row r at his predicted contract."""
    yrs = int(r.yrs_pred)
    pl = {"grp": r.grp, "age": r.age - SKIP, "v1": r.v1, "v2": r.v2, "g1": r.g1,
          "p_share": getattr(r, "p_share", np.nan), "p_gp": getattr(r, "p_gp", np.nan)}
    V, S = vt.comps().paths(pl, yrs, exclude=False, skip=SKIP)
    val, cost, on = vt.team_seasons(V, yrs, r.apy_pred, r.gshare_pred * r.apy_pred * yrs)
    pts = np.zeros((len(V), 5))
    pts[:, :yrs] = vt.points(val, S, cost, on, r.grp == "QB")
    return pts


def optimize(sit, T, arrays, Cand, B, n_picks, cap0, max_signings=8, start=None, n=4000, seed=11,
             max_per_group=None, max_iter=80):
    """Local search for the combination with the most titles within the budget.
    State: one action per player under contract (keep / cut / restructure / extend, where available) and a set of
    signings from Cand (fa_market.rank rows with cost_m). Moves: change one player's action; add, drop or swap a
    signing; change an action and add the best signing that then fits. Best improvement each step; stops when nothing
    improves. All options are drawn once (n futures, seeded per player) so every combination is scored on the same
    draws. Returns the chosen sets and the score on those draws (optimistic: re-score on fresh draws to report)."""
    import pick_value as pv
    mpg = dict(MAX_PER_GROUP if max_per_group is None else max_per_group)
    rng = np.random.default_rng(seed)
    base = pv.baseline_paths(sit, rng, n)
    curve = vt._curve()
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)  # noqa: E731
    b0 = sb(base).sum(axis=1).mean()
    Tm = T.set_index("pfr_id")
    D = {}  # (player, action) -> (n, 5) change vs keeping him as is
    saved = {}
    for p, a in arrays.items():
        idx = np.random.default_rng(zlib.crc32(str(p).encode()) + seed).integers(0, len(a["keep"]), n)
        k = a["keep"][idx]
        for act, key, sv in (("cut", "cut", "saved_cut"), ("restr", "restr", "saved_restr"), ("ext", "ext", "saved_ext")):
            if a.get(key) is not None and p in Tm.index:
                D[(p, act)] = a[key][idx] - k
                saved[(p, act)] = float(Tm.at[p, sv])
    Cand = Cand.drop_duplicates("pfr_id").set_index("pfr_id")
    G = {}
    for p, r in Cand.iterrows():
        pts = signing_points(r)
        G[p] = pts[np.random.default_rng(zlib.crc32(str(p).encode()) + seed + 7).integers(0, len(pts), n)]
    cost = Cand.cost_m.to_dict()
    grp = Cand.grp.to_dict()
    fill = lambda n_cut, n_sign: max(ROSTER - (B["n_signed"] - n_cut + n_sign) - n_picks, 0) * FILL_SHARE * cap0  # noqa: E731

    def budget_left(act, signs):
        n_cut = sum(1 for v in act.values() if v == "cut")
        return (B["budget"] + B["fill"] - fill(n_cut, len(signs)) + sum(saved[(p, v)] for p, v in act.items())
                - sum(cost[p] for p in signs))

    def feasible(act, signs):
        if len(signs) > max_signings or budget_left(act, signs) < -1e-9:
            return False
        cnt = {}
        for p in signs:
            cnt[grp[p]] = cnt.get(grp[p], 0) + 1
        return all(v <= mpg.get(g, 1) for g, v in cnt.items())

    def score(act, signs):
        tot = np.zeros((n, 5))
        for p, v in act.items():
            tot += D[(p, v)]
        for p in signs:
            tot += G[p]
        return float(sb(base + tot).sum(axis=1).mean() - b0)

    act = {p: v for p, v in (start or {}).get("act", {}).items() if (p, v) in D}
    signs = set(p for p in (start or {}).get("signs", []) if p in G)
    if not feasible(act, signs):
        act, signs = {}, set()
    cur = score(act, signs)
    players = sorted({p for p, _ in D})
    for _ in range(max_iter):
        best = (cur, None)
        cands = []
        for p in players:
            for v in ("keep", "cut", "restr", "ext"):
                if v == act.get(p, "keep") or (v != "keep" and (p, v) not in D):
                    continue
                a2 = dict(act)
                if v == "keep":
                    a2.pop(p, None)
                else:
                    a2[p] = v
                cands.append((a2, signs))
        free = [p for p in G if p not in signs]
        for p in free:
            cands.append((act, signs | {p}))
        for p in signs:
            cands.append((act, signs - {p}))
            for q in free:
                cands.append((act, (signs - {p}) | {q}))
        for a2, s2 in cands:
            if feasible(a2, s2):
                sc = score(a2, s2)
                if sc > best[0] + 1e-7:
                    best = (sc, (a2, s2))
            elif s2 is signs and budget_left(a2, signs) > budget_left(act, signs):  # frees money: try adding a signing
                pass
        # compound: a cap move that frees money, plus the best signing that then fits
        for a2, s2 in cands:
            if s2 is not signs or budget_left(a2, signs) <= budget_left(act, signs):
                continue
            for q in free:
                s3 = signs | {q}
                if feasible(a2, s3):
                    sc = score(a2, s3)
                    if sc > best[0] + 1e-7:
                        best = (sc, (a2, s3))
        if best[1] is None:
            break
        cur, (act, signs) = best[0], best[1]
    return {"act": act, "signs": sorted(signs), "titles": cur, "budget_left": budget_left(act, signs)}


# ---------------- dead money already on the books (entered by hand; not in the contract data) ----------------
DEAD_FILE = vt.ROOT / "inputs" / f"dead_money_{SIGN_SEASON}.csv"


def load_dead(path=DEAD_FILE):
    """team -> (dead money $M, date checked, source) for SIGN_SEASON, from inputs/dead_money_<season>.csv.
    Only June 1 cuts and post-June 1 trades from the season before need entering: void-year acceleration is already
    in the contract data's commitments (README 'Offseason workbench'). OTC's terms bar automated collection."""
    if not path.exists():
        return {}
    d = pd.read_csv(path, dtype={"team": str, "checked": str, "source": str})
    d = d[pd.to_numeric(d.dead_money_m, errors="coerce").notna()]
    return {r.team: (float(r.dead_money_m), r.checked if isinstance(r.checked, str) else "",
                     r.source if isinstance(r.source, str) else "") for r in d.itertuples()}


def save_dead(team, amount, source, checked=None, path=DEAD_FILE):
    """Write one team's figure (others unchanged)."""
    checked = checked or pd.Timestamp.today().strftime("%Y-%m-%d")
    d = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["team", "dead_money_m", "checked", "source"])
    row = {"team": team, "dead_money_m": f"{amount:.2f}", "checked": checked, "source": source}
    if (d.team == team).any():
        for k, v in row.items():
            d.loc[d.team == team, k] = v
    else:
        d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.sort_values("team").to_csv(path, index=False)
