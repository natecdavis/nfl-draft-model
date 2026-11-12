"""Recheck of the whole-draft lookahead test (scripts/plan_choice.py) with the availability fix.

plan_choice.py simulated other teams' picks from scratch at every decision, so a faller still on the board
was drawn mostly before the current pick and treated as gone at once (README 'Availability: needs, runs and
fallers'). Here the same versions run with each player's draft spot drawn given he's still on the board at
the current pick (truncated, as in draftday.simulate now):
  B  greedy (unchanged by the fix)
  C  position drop-off, old draws      Ct  position drop-off, conditioned draws
  D  rollout, old draws                Dt  rollout, conditioned draws
Same value tables, boards, teams and seeds as plan_choice.py; drafts 2014-2022; scoring = plan_choice.score.

Run:  python scripts/lookahead_recheck.py [workers=6]      report only:  ... report
"""
import sys
import time
import zlib

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402

import plan_choice as pc  # noqa: E402
from draftday import drift, spread  # noqa: E402
from outlook import GROUPS, OUT  # noqa: E402
from review import OFF_BOARD_RANK, board, draft_night  # noqa: E402

VERSIONS = {"B": ("greedy", False), "C": ("dropoff", False), "Ct": ("dropoff", True),
            "D": ("rollout", False), "Dt": ("rollout", True)}
TRUNC = False
_orig_future = pc.future


def future_trunc(V, gidx, picks, j, avail0, count0, rng):
    """pc.future with each player's draft spot drawn given he's still there at picks[j]."""
    R, N = avail0.shape
    avail, count = avail0.copy(), count0.copy()
    total = np.zeros(R)
    rank = pc.RANKS
    mu, sd = np.log(rank + 10) + drift(rank), spread(rank)
    lo = ndtr((np.log(picks[j] + 10) - mu) / sd)
    z = ndtri(np.clip(lo + (1 - lo) * rng.random((pc.SIMS, N)), 1e-12, 1 - 1e-12))
    order_key = np.tile(mu + sd * z, (R // pc.SIMS, 1))
    best_next = None
    for jj in range(j + 1, len(picks)):
        n = int(round((picks[jj] - picks[jj - 1] - 1) * pc.on_share(picks[jj - 1], picks[jj])))
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


def one_year(y):
    from simulator import situation
    t0 = time.time()
    picks_df, _ = draft_night(y)
    b = board(y, picks_df)
    B = b[b.grp.isin(GROUPS)].reset_index(drop=True)
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
        p = pc.p_need(srow.status, "prob") if sit["qb_from"] is not None else 1.0
        V = pc.value_matrix(pc.Tab(sit, "single"), B, picks, p)
        for v, (method, trunc) in VERSIONS.items():
            pc.future = future_trunc if trunc else _orig_future
            ch = pc.team_draft(B, V, gidx, picks, before, method, seed=y * 1000 + zlib.crc32(tm.encode()) % 997)
            for k, i in zip(picks, ch):
                if i is None:
                    continue
                r = B.iloc[i]
                rows.append({"version": v, "year": y, "team": tm, "pick": k, "player": r["name"], "grp": r.grp,
                             "rank": r["rank"], "drafted_at": r.drafted_at})
    print(f"{y}: {len(rows)} choices, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def report(P):
    U = pd.read_parquet(OUT / "units_choice.parquet")
    rnd = dict(zip(zip(U.year, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    for m, _ in pc.METRICS:
        P[m] = P[m].fillna(0.0)
    P["n_paths"] = P.groupby(["version", "year", "player"]).team.transform("nunique")
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(m, 2.5):+.2f} to {np.percentile(m, 97.5):+.2f})"

    print("How often each version's choice differs from greedy (B), by round:")
    base = P[P.version == "B"].set_index(["year", "team", "pick"]).player
    for v in VERSIONS:
        if v == "B":
            continue
        q = P[P.version == v].set_index(["year", "team", "pick"])
        diff = (q.player != base.reindex(q.index))
        print(f"  {v:3s} " + "  ".join(f"R{r}: {diff[q['round'] == r].mean():.0%}" for r in range(1, 8)))
    for scoring in ("raw", "shared"):
        print(f"\nPaired versions ({scoring} scoring), per team-draft, all rounds (95% bootstrap):")
        for a_, b_ in [("D", "B"), ("Dt", "B"), ("Dt", "D"), ("C", "B"), ("Ct", "B"), ("Ct", "C")]:
            parts = []
            for m, lab in pc.METRICS:
                x = {}
                for v in (a_, b_):
                    Q = P[P.version == v]
                    val = Q[m] / Q.n_paths if scoring == "shared" else Q[m]
                    x[v] = val.groupby([Q.year, Q.team]).sum()
                parts.append(f"{lab}: {boot(x[a_] - x[b_].reindex(x[a_].index, fill_value=0))}")
            print(f"  {a_:2s} - {b_:2s}: " + " | ".join(parts))
    print("\nDt - B by round group, titles (single, shared):")
    for gl, f in [("Round 1", lambda r: r == 1), ("Rounds 2-4", lambda r: (r >= 2) & (r <= 4)), ("Rounds 5-7", lambda r: r >= 5)]:
        x = {}
        for v in ("Dt", "B"):
            Q = P[(P.version == v) & f(P["round"])]
            x[v] = (Q.t_single / Q.n_paths).groupby([Q.year, Q.team]).sum()
        idx = x["Dt"].index.union(x["B"].index)
        print(f"  {gl}: {boot(x['Dt'].reindex(idx, fill_value=0) - x['B'].reindex(idx, fill_value=0))}")
    print("\nPosition mix (% of picks), rounds 1-4:")
    Q = P[P["round"] <= 4]
    print((pd.crosstab(Q.version, Q.grp, normalize="index") * 100).round(0).to_string())


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        P = pd.concat(list(ex.map(one_year, pc.YEARS)), ignore_index=True)
    P.to_parquet(OUT / "lookahead_recheck_raw.parquet")
    P = pc.score(P)
    P.to_parquet(OUT / "lookahead_recheck.parquet")
    report(P)


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "lookahead_recheck.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
