"""Whole-draft choice test, drafts 2014-2022: greedy vs looking ahead to the team's later picks,
with the units and QB-need fixes. The model drafts each team's whole draft at the picks the team
actually made (other teams draft as they really did; players the model took are gone for it later),
like the app's draft simulator. Compared with the team's real draft.

Versions
  A  greedy, family units, QB need certain            (today's model, fall shading applied once)
  B  greedy, single units, QB need probability
  C  position drop-off, single, probability:  take the player with the most value now minus the
     expected best value at his position at the team's next pick
  D  rollout, single, probability:  for each serious candidate, value now + expected value of the
     team's later picks this draft if we take him (later picks greedy), over simulated drafts
  E  rollout, family units, QB need certain             (isolates the rollout)

Decisions use only what was known at the pick: the board still available at that moment (players
really taken earlier are gone) and simulated other teams for the future: they take board players
in an order scattered around consensus rank (draftday.drift / spread, measured on Hasan boards),
at the historical rate of board players per pick (picks 1-100 ~all, 150-200 ~76%, 200+ ~55%).
Common random numbers across candidates.

QB need probability (scripts/need_choice diagnostics, 2014-2023 team-years): share of flagged teams
whose main QB two seasons later was someone else: no starter 1.0, stopgap veteran 0.79, later-round
rookie 0.62, paid veteran after a below-median season 0.54, age 36+ 0.52. A first QB's value =
p x value(fills need) + (1 - p) x value(need already filled). Unflagged teams: unchanged (no QB need).
These rates come from the same years as the test (5 numbers; small risk of overfitting).

Scoring per team-draft: sum over the team's picks of each player's realized value at that pick, model
minus team: titles in single and in family units (review.RealizedV2), PFR AV, seasons started, best
second-contract pay (% of cap). Undrafted players count 0 on AV / starts / pay (titles use their real
seasons if they played). "Shared": a player on several teams' model drafts that year counts 1/n on
each, so one hit isn't credited many times.

Run:  python scripts/plan_choice.py [workers=6]      report only:  ... report
"""
import sys
import time
import zlib

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from draftday import drift, spread  # noqa: E402
from need_choice import FW, NTable, _key  # noqa: E402
from outlook import GROUPS, OUT, RAW  # noqa: E402
from review import OFF_BOARD_RANK, board, draft_night  # noqa: E402

YEARS = list(range(2014, 2023))
VERSIONS = {"A": ("greedy", "family", "certain"), "B": ("greedy", "single", "prob"),
            "C": ("dropoff", "single", "prob"), "D": ("rollout", "single", "prob"),
            "E": ("rollout", "family", "certain")}
TABLE_DIR = {"family": OUT / "need_tables", "single": OUT / "units_tables"}
SIMS, TOP = 200, 12
ON_K = [16, 48, 82, 125, 175, 240]  # board players' share of picks, by pick (review 2014-22)
ON_P = [1.0, 1.0, 0.97, 0.94, 0.76, 0.55]


def p_need(status: str, mode: str) -> float:
    if mode == "certain":
        return 1.0
    s = str(status)
    if s.startswith("no starter"):
        return 1.0
    if s.startswith("stopgap"):
        return 0.79
    if s.startswith("rookie deal (later"):
        return 0.62
    if "below-average" in s:
        return 0.54
    if s.startswith("veteran, "):
        return 0.52
    return 1.0  # unflagged: no QB need in the tables, so p doesn't matter


class Tab(NTable):
    def __init__(self, sit, units):
        z = np.load(TABLE_DIR[units] / f"{_key(sit)}.npz")
        self.v = {g: z[g] for g in z.files}
        from review import K_GRID, T_GRID
        self.x, self.y = np.log(T_GRID + 10), np.log(K_GRID + 10)


def value_matrix(tab, B, picks, p):
    """(players, picks, 3 need levels) titles in % of one."""
    V = np.zeros((len(B), len(picks), 3))
    for i, (g, r) in enumerate(zip(B.grp, B["rank"])):
        for j, k in enumerate(picks):
            if g == "QB":
                nf, f = (float(tab._bilin(tab.v["QB"][q], r, k)) for q in (0, 1))
                V[i, j] = [p * nf + (1 - p) * f, f, f]
            else:
                V[i, j] = [tab.current(g, lv, r, k) for lv in range(3)]
    return V * 100


def on_share(a, b):
    return float(np.interp((a + b) / 2, ON_K, ON_P))


def future(V, gidx, picks, j, avail0, count0, rng):
    """Simulated value of the team's picks after index j, picking greedily, for each row of
    avail0 (rows = candidate x sims). Returns (rows,) total and the next-pick best by group."""
    R, N = avail0.shape
    avail, count = avail0.copy(), count0.copy()
    total = np.zeros(R)
    rank = RANKS
    z = rng.standard_normal((SIMS, N))
    order_key = np.log(rank + 10) + drift(rank) + spread(rank) * z  # (SIMS, N); lower = taken sooner
    order_key = np.tile(order_key, (R // SIMS, 1))
    best_next = None
    for jj in range(j + 1, len(picks)):
        n = int(round((picks[jj] - picks[jj - 1] - 1) * on_share(picks[jj - 1], picks[jj])))
        if n > 0:
            key = np.where(avail, order_key, np.inf)
            n_ = min(n, N - 1)
            gone = np.argpartition(key, n_, axis=1)[:, :n_]
            np.put_along_axis(avail, gone, False, axis=1)
        L = np.minimum(count[:, gidx], 2)
        vals = np.where(avail, V[np.arange(N)[None, :], jj, L], -np.inf)
        if jj == j + 1:
            best_next = vals
        i = np.argmax(vals, axis=1)
        v = vals[np.arange(R), i]
        ok = np.isfinite(v)
        total[ok] += v[ok]
        avail[np.arange(R)[ok], i[ok]] = False
        np.add.at(count, (np.arange(R)[ok], gidx[i[ok]]), 1)
    return total, best_next


RANKS = None


def team_draft(B, V, gidx, picks, drafted_before, method, seed):
    """The model's choices at each of the team's picks. drafted_before[j] = bool mask of board
    players really taken before picks[j]."""
    global RANKS
    RANKS = B["rank"].to_numpy(float)
    N, G = len(B), gidx.max() + 1
    taken = np.zeros(N, bool)
    count = np.zeros(G, int)
    out = []
    for j, k in enumerate(picks):
        avail = ~drafted_before[j] & ~taken
        if not avail.any():
            out.append(None); continue
        now = np.where(avail, V[np.arange(N), j, np.minimum(count[gidx], 2)], -np.inf)
        rng = np.random.default_rng(seed + j)
        last = j == len(picks) - 1
        if method == "greedy" or last:
            i = int(np.argmax(now))
        elif method == "dropoff":
            # expected best value at each position at the next pick, if we take no one now
            # (future() with one row block of sims)
            _, bn = future(V, gidx, picks, j, np.tile(avail, (SIMS, 1)), np.tile(count, (SIMS, 1)), rng)
            exp_best = np.array([np.mean(np.max(np.where(gidx[None, :] == g, bn, -np.inf), axis=1)
                                         .clip(min=0)) if (gidx == g).any() else 0.0 for g in range(G)])
            # players at a position nobody is left at next pick: nothing to subtract
            i = int(np.argmax(np.where(avail, now - exp_best[gidx], -np.inf)))
        else:  # rollout
            top = np.argsort(-now)[:TOP]
            cand = set(int(c) for c in top[np.isfinite(now[top])])
            for g in range(G):
                m = np.where(gidx == g, now, -np.inf)
                if np.isfinite(m.max()):
                    cand.add(int(np.argmax(m)))
            cand = sorted(cand)
            av = np.repeat(avail[None, :], len(cand) * SIMS, axis=0)
            ct = np.repeat(count[None, :], len(cand) * SIMS, axis=0)
            for c_i, c in enumerate(cand):
                rows = slice(c_i * SIMS, (c_i + 1) * SIMS)
                av[rows, c] = False
                ct[rows, gidx[c]] += 1
            tot, _ = future(V, gidx, picks, j, av, ct, rng)
            score = [now[c] + tot[c_i * SIMS:(c_i + 1) * SIMS].mean() for c_i, c in enumerate(cand)]
            i = cand[int(np.argmax(score))]
        out.append(i)
        taken[i] = True
        count[gidx[i]] += 1
    return out


def one_year(y):
    from simulator import situation
    t0 = time.time()
    picks_df, _ = draft_night(y)
    b = board(y, picks_df)
    B = b[b.grp.isin(GROUPS)].reset_index(drop=True)  # no specialists
    B["rank"] = B["rank"].clip(upper=OFF_BOARD_RANK).astype(float)
    gnames = sorted(B.grp.unique())
    gidx = B.grp.map({g: i for i, g in enumerate(gnames)}).to_numpy()
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year == y]
    rows = []
    for _, srow in S.iterrows():
        tm, sit = srow.team, situation(srow)
        picks = sorted(int(k) for k in picks_df[picks_df.team == tm].pick)
        if not picks:
            continue
        before = [(B.drafted_at.notna() & (B.drafted_at < k)).to_numpy() for k in picks]
        Vs = {}
        for v, (method, units, need) in VERSIONS.items():
            if (units, need) not in Vs:
                p = p_need(srow.status, need) if sit["qb_from"] is not None else 1.0
                Vs[(units, need)] = value_matrix(Tab(sit, units), B, picks, p)
            V = Vs[(units, need)]
            ch = team_draft(B, V, gidx, picks, before, method, seed=y * 1000 + zlib.crc32(tm.encode()) % 997)
            for k, i in zip(picks, ch):
                if i is None:
                    continue
                r = B.iloc[i]
                rows.append({"version": v, "year": y, "team": tm, "pick": k, "player": r["name"], "grp": r.grp,
                             "rank": r["rank"], "drafted_at": r.drafted_at})
    print(f"{y}: {len(rows)} choices, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def score(P):
    from review import RealizedV2
    from simulator import situation
    from trade import Context
    S = pd.read_parquet(OUT / "review_states.parquet")
    sit = {(r.year, r.team): situation(r) for _, r in S.iterrows()}
    ctx = Context()
    for units in ("single", "family"):
        R = RealizedV2(ctx, units=units)
        P[f"t_{units}"] = [R.titles(sit[(y, tm)], g, y, at, k, nm) for y, tm, g, at, k, nm in
                           zip(P.year, P.team, P.grp, P.drafted_at, P.pick, P.player)]
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    pay = {(y, p): a for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}
    get = lambda d, y, a: (0.0 if pd.isna(a) else (d.get((y, int(a))) if pd.notna(d.get((y, int(a)))) else 0.0))
    P["av"] = [get(av, y, a) for y, a in zip(P.year, P.drafted_at)]
    P["gs"] = [get(gs, y, a) for y, a in zip(P.year, P.drafted_at)]
    P["pay"] = [get(pay, y, a) * 100 for y, a in zip(P.year, P.drafted_at)]
    return P


METRICS = [("t_single", "titles, single units"), ("t_family", "titles, family units"), ("av", "PFR AV"),
           ("gs", "seasons started"), ("pay", "2nd-contract pay, % cap")]


def report(P):
    U = pd.read_parquet(OUT / "units_choice.parquet")  # the team's own picks, same yardsticks
    T = U[["year", "team", "pick", "round", "grp", "t_single_team", "t_family_team", "av_team", "gs_team", "pay_team"]] \
        .rename(columns={"t_single_team": "t_single", "t_family_team": "t_family", "av_team": "av", "gs_team": "gs",
                         "pay_team": "pay"})
    rnd = dict(zip(zip(U.year, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    for m, _ in METRICS:
        T[m] = T[m].fillna(0.0); P[m] = P[m].fillna(0.0)
    P["n_paths"] = P.groupby(["version", "year", "player"]).team.transform("nunique")
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(m, 2.5):+.2f} to {np.percentile(m, 97.5):+.2f})"

    groups = [("All", lambda r: r > 0), ("Round 1", lambda r: r == 1), ("Rounds 2-4", lambda r: (r >= 2) & (r <= 4)),
              ("Rounds 5-7", lambda r: r >= 5)]
    print("Per team-draft: model's draft minus the team's real draft (sum over the team's picks), 288 team-drafts:")
    for shared in (False, True):
        print(f"\n=== {'shared (a player on n teams model drafts counts 1/n)' if shared else 'raw'}")
        for m, lab in METRICS:
            print(f"-- {lab}")
            for v in VERSIONS:
                Q = P[P.version == v].copy()
                if shared:
                    Q[m] = Q[m] / Q.n_paths
                parts = []
                for gl, f in groups:
                    a = Q[f(Q["round"])].groupby(["year", "team"])[m].sum()
                    t = T[f(T["round"])].groupby(["year", "team"])[m].sum()
                    d = (a.reindex(t.index, fill_value=0) - t)
                    parts.append(f"{gl}: {boot(d)}")
                print(f"   {v} {'/'.join(VERSIONS[v])[:26]:26s} " + " | ".join(parts))
    print("\nPaired versions (shared scoring), per team-draft, all rounds:")
    for a_, b_ in [("B", "A"), ("C", "B"), ("D", "B"), ("E", "A"), ("D", "A")]:
        parts = []
        for m, lab in METRICS:
            x = {}
            for v in (a_, b_):
                Q = P[P.version == v]
                x[v] = (Q[m] / Q.n_paths).groupby([Q.year, Q.team]).sum()
            parts.append(f"{lab}: {boot(x[a_] - x[b_].reindex(x[a_].index, fill_value=0))}")
        print(f"  {a_} - {b_}: " + " | ".join(parts))
    print("\nPosition mix by version (% of model picks) and the team:")
    P["rg"] = pd.cut(P["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    for v in VERSIONS:
        Q = P[P.version == v]
        print(f"-- {v}\n" + (pd.crosstab(Q.rg, Q.grp, normalize="index") * 100).round(0).to_string())
    print("\nSame player in the most teams' model drafts (version D):")
    print(P[P.version == "D"].drop_duplicates(["year", "player"]).nlargest(8, "n_paths")[["year", "player", "grp", "n_paths"]]
          .to_string(index=False))
    print("\nQB picks per team-draft:", {v: round((P[P.version == v].grp == "QB").sum() / 288, 2) for v in VERSIONS},
          "team:", round((T.grp == "QB").sum() / 288, 2))


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        P = pd.concat(list(ex.map(one_year, YEARS)), ignore_index=True)
    P.to_parquet(OUT / "plan_choice_raw.parquet")
    P = score(P)
    P.to_parquet(OUT / "plan_choice.parquet")
    report(P)


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "plan_choice.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
