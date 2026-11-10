"""Whole-draft choice test with a teammate (stacking) discount for WR, and WR + OL, drafts 2014-2022.

From scripts/wr_stacking.py test D: the betting market prices a missing WR lower when the WRs still playing
are well paid (-0.048 pts/game per 1% of cap of teammates' pre-season price, on an average of 0.73 for the
absent starter: -6.6% per 1%); OL -0.030 on 0.37 (-8.1% per 1%). CB / DL ~0.

Versions (greedy, single units, QB need probability, talent-model label; tables in data/derived/stack_tables):
  B   today's rule: each earlier pick at the group this draft cuts years 1-2 playing time x0.6 / x0.3
  W   WR: playing time in ALL seasons x m, m = 1 - 0.066 x (X - mean X), clipped 0.4-1.5, replacing the
      same-draft rule for WRs. X = summed pre-season price (% of cap) of the team's top 2 WRs on draft day:
      last season's WRs who didn't leave before the draft, veterans added before it, and WRs the model
      already took this draft (at their slot's expected price). Mean X = average over WRs actually drafted
      2014-22, at their real team at the moment of their pick, so the average drafted WR keeps m = 1.
  WO  W plus the same for OL (T / G / C together; top 4; 8.1% per 1%).
Penalty-only versions (m capped at 1) W- and WO- check that the result isn't coming from boosting WR-thin teams.

Scoring per team-draft, version minus B: realized titles (review.RealizedV2, single units) as the player
played; the same rescaled by m at this team / m at the team that really drafted him ("adjusted": his real
career happened beside his real teammates); PFR AV, seasons started, 2nd-contract pay.

Run:  python scripts/stack_choice.py build [workers]   then   python scripts/stack_choice.py choose [workers]
      python scripts/stack_choice.py report
"""
import os
import sys

os.environ["OUTCOME_UNITS"] = "single"
os.environ["POS_PAY"] = "off"
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import GROUPS, OUT, RAW  # noqa: E402

TDIR = OUT / "stack_tables"
YEARS = list(range(2014, 2023))
MGRID = np.round(np.arange(0.4, 1.51, 0.1), 2)
STACK = {"WR": ("WR", 2, 0.066), "T": ("OL", 4, 0.081), "G": ("OL", 4, 0.081), "C": ("OL", 4, 0.081)}
OL = {"T", "G", "C"}
FAM = {"WR": "WR", "T": "OL", "G": "OL", "C": "OL"}
VERSIONS = {"B": (), "W": ("WR",), "WO": ("WR", "OL"), "W-": ("WR",), "WO-": ("WR", "OL")}
FW = 0.85


def _key(sit):
    sh = "_".join(f"{k}:{v:g}" for k, v in sorted(sit["shift"].items())) or "none"
    return f"start{sit['start']:g}_qb{sit['qb_from']}_shift{sh}"


def build_table(sit):
    f = TDIR / f"{_key(sit)}.npz"
    if f.exists():
        return
    from draftday import DraftEngine
    from pick_value import pick_effect
    from review import K_GRID, T_GRID
    from trade import Context
    eng = DraftEngine(sit, ctx=Context(), fall_weight=FW)
    out = {}
    for g in GROUPS:
        if g == "QB":
            arr = np.array([[[eng.titles(eng.player_points(g, t, k, qb_filled=f_)) for k in K_GRID] for t in T_GRID]
                            for f_ in (False, True)])
        else:
            levels = [(1.0, 1.0), (0.6, 1.0), (0.3, 1.0)] + ([(1.0, m) for m in MGRID] if g in STACK else [])
            arr = np.zeros((len(levels), len(T_GRID), len(K_GRID)))
            for a, t in enumerate(T_GRID):
                sh0, price = eng._outcomes(g, int(t))
                for b, k in enumerate(K_GRID):
                    for li, (me, ma) in enumerate(levels):
                        sh = sh0 * ma
                        sh[:, :2] *= me
                        pts = pick_effect(sh, price, int(k), g, {**sit, "qb_from": eng._qb_from()},
                                          eng.ctx.cost, eng.ctx.bar, eng.ctx.rates)
                        arr[li, a, b] = eng.titles(pts)
        out[g] = arr
    TDIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(f)


def sits():
    from simulator import situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(YEARS)]
    return {_key(situation(r)): situation(r) for _, r in S.iterrows()}


def build(workers):
    from concurrent.futures import ProcessPoolExecutor
    todo = [s for k, s in sits().items() if not (TDIR / f"{k}.npz").exists()]
    print(f"building {len(todo)} tables in {TDIR.name}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))


# ---------- teammates on draft day ----------
_PRICES = None


def prices():
    global _PRICES
    if _PRICES is None:
        import absence_test as at
        _PRICES = at.prices() * 100
    return _PRICES


def slot_price():
    c = pd.read_parquet(OUT / "outlook_curves.parquet")
    lib = pd.read_parquet(OUT / "value_library.parquet")
    tp = lib.groupby(["grp", "tier"], observed=True)["price"].mean().unstack()
    c["exp"] = sum(c[t] * c.grp.map(tp[t]) for t in ["bust", "starter", "star"]) * 100
    return {(g, int(k)): v for g, k, v in zip(c.grp, c.pick, c.exp)}


def departures_and_adds(y):
    """roster_need.departures_and_adds, but a contract's team string ("BUF/DAL/CLE") counts every team in it
    (roster_need keeps only the last one, so a player who re-signed and was later traded looks like he left)."""
    from review import DRAFT_START
    from roster_need import team
    pl_ = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl_.gsis_id, pl_.pfr_id))
    c = pd.read_parquet(RAW / "contracts.parquet")
    nick = dict(zip(*pd.read_parquet(RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    c["tms"] = c.team.map(lambda x: {team(nick.get(t, t)) for t in str(x).split("/")})
    c["pfr"] = c.gsis_id.map(g2p)
    cy = c[(c.year_signed == y) & c.pfr.notna() & (c.draft_year != y)]
    signed = {}
    for p, ts in zip(cy.pfr, cy.tms):
        signed.setdefault(p, set()).update(ts)
    t = pd.read_parquet(RAW / "trades.parquet")
    t["date"] = pd.to_datetime(t.trade_date)
    t = t[(t.date > pd.Timestamp(f"{y - 1}-08-01")) & (t.date < pd.Timestamp(DRAFT_START[y])) & t.pfr_id.notna()]
    return signed, {(p, team(g)) for p, g in zip(t.pfr_id, t.gave)}, {(p, team(r)) for p, r in zip(t.pfr_id, t.received)}


def incumbents(y):
    """{(team, fam): [prices of last season's players who stayed + veterans added before the draft]}
    A multi-team contract string can't say which team signed him, so adds count only single-team strings."""
    ps = pd.read_parquet(OUT / "wr_stacking_ps.parquet")
    ps = ps[ps.g.isin(["WR", "OL"])]
    prev = ps[ps.season == y - 1].groupby(["pfr", "team"]).g.first().reset_index()
    signed, away, tin = departures_and_adds(y)
    pr = prices()
    gof = prev.groupby("pfr").g.first()
    out = {}
    for p, tm, g in zip(prev.pfr, prev.team, prev.g):
        if (p, tm) in away or (p in signed and tm not in signed[p]):
            continue
        out.setdefault((tm, g), {})[p] = float(pr.get((p, y), 0.5))
    adds = [(p, next(iter(ts))) for p, ts in signed.items() if len(ts) == 1] + list(tin)
    for p, t in adds:
        if p in gof.index:
            out.setdefault((t, gof[p]), {})[p] = float(pr.get((p, y), 0.5))
    return {k: list(v.values()) for k, v in out.items()}


def topk(vals, k):
    return float(np.sum(sorted(vals, reverse=True)[:k]))


def mult(fam, X, mean, cap):
    b = 0.066 if fam == "WR" else 0.081
    m = float(np.clip(1 - b * (X - mean), MGRID[0], MGRID[-1]))
    return min(m, 1.0) if cap else m


def real_X(years):
    """X each actually drafted WR / OL faced at his real team when picked: incumbents + the team's earlier
    picks at the family this draft. Returns {(year, pick): X} and the mean by family."""
    sp = slot_price()
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    from review import draft_night
    out, by = {}, {"WR": [], "OL": []}
    for y in years:
        d, _ = draft_night(y)
        inc = incumbents(y)
        d = d[d.grp.isin(list(FAM))].sort_values("pick")
        rook = {}
        for k, tm, g in zip(d.pick, d.team, d.grp):
            fam = FAM[g]
            n = STACK[g][1]
            X = topk(inc.get((tm, fam), []) + rook.get((tm, fam), []), n)
            out[(y, int(k))] = X
            by[fam].append(X)
            rook.setdefault((tm, fam), []).append(sp.get((g, int(k)), 1.0))
    return out, {f: float(np.mean(v)) for f, v in by.items()}


# ---------- choices ----------
def one_year(args):
    y, means = args
    import time
    from plan_choice import p_need
    from review import K_GRID, OFF_BOARD_RANK, T_GRID, board, draft_night
    from simulator import situation
    t0 = time.time()
    sp = slot_price()
    picks_df, _ = draft_night(y)
    b = board(y, picks_df)
    B = b[b.grp.isin(GROUPS)].reset_index(drop=True)
    B["rank"] = B["rank"].clip(upper=OFF_BOARD_RANK).astype(float)
    inc = incumbents(y)
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year == y]
    x, yk = np.log(T_GRID + 10), np.log(K_GRID + 10)

    def bilin(A, rank, pick):
        t = rank if pick <= rank else np.exp(np.log(rank + 10) + FW * (np.log(pick + 10) - np.log(rank + 10))) - 10
        lt, lk = np.log(t + 10), np.log(pick + 10)
        i = int(np.clip(np.searchsorted(x, lt) - 1, 0, len(x) - 2)); j = int(np.clip(np.searchsorted(yk, lk) - 1, 0, len(yk) - 2))
        fx = np.clip((lt - x[i]) / (x[i + 1] - x[i]), 0, 1); fy = np.clip((lk - yk[j]) / (yk[j + 1] - yk[j]), 0, 1)
        return (1 - fx) * (1 - fy) * A[..., i, j] + fx * (1 - fy) * A[..., i + 1, j] + (1 - fx) * fy * A[..., i, j + 1] + fx * fy * A[..., i + 1, j + 1]

    rows = []
    grp, rank, dat = B.grp.to_numpy(), B["rank"].to_numpy(float), B.drafted_at.to_numpy(float)
    for _, srow in S.iterrows():
        tm, sit = srow.team, situation(srow)
        picks = sorted(int(k) for k in picks_df[picks_df.team == tm].pick)
        if not picks:
            continue
        z = np.load(TDIR / f"{_key(sit)}.npz")
        V = {g: z[g] for g in z.files}
        p = p_need(srow.status, "prob") if sit["qb_from"] is not None else 1.0
        for v, fams in VERSIONS.items():
            cap = v.endswith("-")
            taken = np.zeros(len(B), bool)
            count = {}
            rook = {"WR": [], "OL": []}
            for k in picks:
                avail = ~taken & ~(np.nan_to_num(dat, nan=1e9) < k)
                if not avail.any():
                    continue
                Xs = {f: topk(inc.get((tm, f), []) + rook[f], 2 if f == "WR" else 4) for f in ("WR", "OL")}
                best, bi, bm = -np.inf, None, 1.0
                for i in np.where(avail)[0]:
                    g = grp[i]
                    if g == "QB":
                        val = bilin(V["QB"][1], rank[i], k) if count.get("QB") else (
                            p * bilin(V["QB"][0], rank[i], k) + (1 - p) * bilin(V["QB"][1], rank[i], k))
                        m = 1.0
                    elif g in STACK and FAM[g] in fams:
                        m = mult(FAM[g], Xs[FAM[g]], means[FAM[g]], cap)
                        val = float(np.interp(m, MGRID, bilin(V[g][3:], rank[i], k)))
                    else:
                        val = bilin(V[g][min(count.get(g, 0), 2)], rank[i], k)
                        m = mult(FAM[g], Xs[FAM[g]], means[FAM[g]], cap) if g in STACK else 1.0
                    if val > best:
                        best, bi, bm = val, i, m
                g = grp[bi]
                taken[bi] = True
                count[g] = count.get(g, 0) + 1
                if g in STACK:
                    rook[FAM[g]].append(sp.get((g, int(k)), 1.0))
                rows.append({"version": v, "year": y, "team": tm, "pick": k, "player": B.loc[bi, "name"], "grp": g,
                             "rank": rank[bi], "drafted_at": dat[bi], "m_here": bm, "value": float(best)})
    print(f"{y}: {len(rows)} choices, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def choose(workers):
    from concurrent.futures import ProcessPoolExecutor
    rx, means = real_X(YEARS)
    print("mean X of drafted players (pre-season price, % of cap):", {k: round(v, 2) for k, v in means.items()})
    with ProcessPoolExecutor(workers) as ex:
        P = pd.concat(list(ex.map(one_year, [(y, means) for y in YEARS])), ignore_index=True)
    import plan_choice as pc
    P = pc.score(P)
    P = P.drop(columns=["t_family"])
    # m at the team that really drafted him (same formula, uncapped; 1 if undrafted or not WR / OL)
    P["m_real"] = [mult(FAM[g], rx[(y, int(a))], means[FAM[g]], False) if g in STACK and pd.notna(a)
                   and (y, int(a)) in rx else 1.0 for g, y, a in zip(P.grp, P.year, P.drafted_at)]
    # m here under the uncapped rule for every version (B's chosen WRs scored the same way)
    from review import RealizedV2
    from simulator import situation
    from trade import Context
    St = pd.read_parquet(OUT / "review_states.parquet")
    sit = {(r.year, r.team): situation(r) for _, r in St.iterrows()}
    R = RealizedV2(Context(), units="single")
    P = add_m_uncapped(P, means)
    P["t_adj"] = [R.titles(sit[(y, tm)], g, y, at, k, nm, scale=mh / mr) for y, tm, g, at, k, nm, mh, mr in
                  zip(P.year, P.team, P.grp, P.drafted_at, P.pick, P.player, P.m_eval, P.m_real)]
    P.to_parquet(OUT / "stack_choice.parquet")
    report(P)


def add_m_uncapped(P, means):
    """m_eval: the uncapped WR + OL teammate multiplier for each chosen player given that version's own
    earlier picks (the yardstick is the same for every version)."""
    sp = slot_price()
    out = []
    for (v, y, tm), d in P.sort_values("pick").groupby(["version", "year", "team"], sort=False):
        inc = incumbents_cached(y)
        rook = {"WR": [], "OL": []}
        for idx, g, k in zip(d.index, d.grp, d.pick):
            if g in STACK:
                f = FAM[g]
                X = topk(inc.get((tm, f), []) + rook[f], 2 if f == "WR" else 4)
                out.append((idx, mult(f, X, means[f], False)))
                rook[f].append(sp.get((g, int(k)), 1.0))
            else:
                out.append((idx, 1.0))
    P["m_eval"] = pd.Series(dict(out))
    return P


_INC = {}


def incumbents_cached(y):
    if y not in _INC:
        _INC[y] = incumbents(y)
    return _INC[y]


def report(P):
    U = pd.read_parquet(RAW / "draft_picks.parquet")
    rnd = dict(zip(zip(U.season, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    M = [("t_single", "titles, as played"), ("t_adj", "titles, teammate-adjusted"), ("av", "PFR AV"),
         ("gs", "seasons started"), ("pay", "2nd-contract pay, % cap")]
    for m, _ in M:
        P[m] = P[m].fillna(0.0)
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"

    G = [("All", lambda r: r > 0), ("R1", lambda r: r == 1), ("R2-4", lambda r: (r >= 2) & (r <= 4)),
         ("R5-7", lambda r: r >= 5)]
    base = P[P.version == "B"].set_index(["year", "team", "pick"])
    for v in [v for v in VERSIONS if v != "B"]:
        print(f"\n{v} - B per team-draft ({P.groupby(['year', 'team']).ngroups}):")
        for m, lab in M:
            parts = []
            for gl, f in G:
                x = {w: P[(P.version == w) & f(P["round"])].groupby(["year", "team"])[m].sum() for w in (v, "B")}
                idx = x[v].index.union(x["B"].index)
                parts.append(f"{gl} {boot(x[v].reindex(idx, fill_value=0) - x['B'].reindex(idx, fill_value=0))}")
            print(f"  {lab:26s} " + " | ".join(parts))
        a = P[P.version == v].set_index(["year", "team", "pick"])
        print(f"  same choice as B: {(a.player == base.player.reindex(a.index)).mean():.0%}")
    P["rg"] = pd.cut(P["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    for v in VERSIONS:
        Q = P[P.version == v]
        print(f"-- {v} position mix (% of picks)\n" + (pd.crosstab(Q.rg, Q.grp, normalize="index") * 100).round(0).to_string())
    for v in VERSIONS:
        Q = P[P.version == v].sort_values("pick")
        first3 = Q.groupby(["year", "team"]).head(3)
        n = first3.groupby(["year", "team"]).grp.apply(lambda s: (s == "WR").sum())
        print(f"{v}: WRs among each team's first 3 picks: mean {n.mean():.2f}, 2+ in {np.mean(n >= 2):.0%} of drafts")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "report"
    w = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 6
    if step == "build":
        build(w)
    elif step == "choose":
        choose(w)
    else:
        report(pd.read_parquet(OUT / "stack_choice.parquet"))
