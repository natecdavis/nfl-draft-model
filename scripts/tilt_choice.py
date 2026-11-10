"""Whole-draft choice test: tilt each position's value by board rank so its spread matches second-contract pay.

scripts/grade_noise_test.py part 1: the label's elasticity of 4-year value with log(rank + 10) vs that of
second-contract pay (Poisson GLM, drafted board players 2012-20). The label is flatter at DB / LB / RB / EDGE
(its pay curves under-price quality where quality persists poorly). Tilt: a player's playing time (so his value,
not his cost) in every season x f_g(t) = ((t + 10) / (r0_g + 10)) ^ (s x d_g), d_g = b_pay - b_label,
t = effective talent rank (after fall shading). r0_g is set so the average f over the group's drafted board
players 2012-20 is 1 (no change in the position's overall level, only its spread). QB unchanged.

  B   today's model (data/derived/stack_tables, levels 0-2)
  T   full tilt (s = 1), T/2 half tilt (s = 0.5)          tables: data/derived/tilt_tables_s{1,0.5}
Greedy, single units, QB need probability, current same-draft need rule; drafts 2014-22 (288 team-drafts).
Scored like scripts/stack_choice.py: realized titles (label currency, so the yardstick shares the flat curves
and leans toward B), PFR AV, seasons started (the fairest outside yardsticks here), 2nd-contract pay (the tilt's
own target, partly circular).

Run:  python scripts/tilt_choice.py build [workers]; python scripts/tilt_choice.py choose [workers]; ... report
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

YEARS = list(range(2014, 2023))
FW = 0.85
# elasticities from grade_noise_test.py part 1 (label, pay)
B_LABEL = {"WR": -0.90, "TE": -0.56, "OL": -0.62, "DL": -0.72, "EDGE": -0.64, "RB": -0.78, "LB": -0.48, "DB": -0.58}
B_PAY = {"WR": -0.86, "TE": -0.57, "OL": -0.60, "DL": -0.74, "EDGE": -0.69, "RB": -0.84, "LB": -0.59, "DB": -0.68}
FAMILY = {"WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "IDL": "DL", "ED": "EDGE", "RB": "RB", "LB": "LB",
          "CB": "DB", "S": "DB"}
SCALES = [1.0, 0.5]


def r0():
    """Per family: rank r0 with mean over drafted board players of ((r+10)/(r0+10))^d = 1."""
    import prospect_model as pm
    d = pm.load()
    d = d[d.drafted & d.hasan_rank.notna() & d.cls.between(2012, 2020)]
    out = {}
    for f in B_LABEL:
        r = d[d.grp == f].hasan_rank.to_numpy(float)
        dd = B_PAY[f] - B_LABEL[f]
        if abs(dd) < 1e-9:
            out[f] = 50.0
            continue
        # mean((r+10)^d) = (r0+10)^d
        out[f] = float(np.mean((r + 10) ** dd) ** (1 / dd) - 10)
    return out


R0 = None


def tilt(g, t, s):
    f = FAMILY.get(g)
    if f is None:
        return 1.0
    return ((t + 10) / (R0[f] + 10)) ** (s * (B_PAY[f] - B_LABEL[f]))


def tdir(s):
    return OUT / f"tilt_tables_s{s:g}"


def build_table(args):
    sit, s, r0_ = args
    global R0
    R0 = r0_
    from stack_choice import _key
    f = tdir(s) / f"{_key(sit)}.npz"
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
            arr = np.zeros((3, len(T_GRID), len(K_GRID)))
            for a, t in enumerate(T_GRID):
                sh0, price = eng._outcomes(g, int(t))
                ma = tilt(g, float(t), s)
                for b, k in enumerate(K_GRID):
                    for li, me in enumerate((1.0, 0.6, 0.3)):
                        sh = sh0 * ma
                        sh[:, :2] *= me
                        pts = pick_effect(sh, price, int(k), g, {**sit, "qb_from": eng._qb_from()},
                                          eng.ctx.cost, eng.ctx.bar, eng.ctx.rates)
                        arr[li, a, b] = eng.titles(pts)
        out[g] = arr
    tdir(s).mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(f)


def build(workers):
    from concurrent.futures import ProcessPoolExecutor
    import stack_choice as sc
    sc.YEARS = YEARS
    r = r0()
    print("r0 by family:", {k: round(v, 1) for k, v in r.items()})
    print("tilt at rank 5 / 50 / 150 (s=1):", {f: tuple(round(((t + 10) / (r[f] + 10)) ** (B_PAY[f] - B_LABEL[f]), 2)
                                                       for t in (5, 50, 150)) for f in B_LABEL})
    todo = [(st, s, r) for s in SCALES for k, st in sc.sits().items() if not (tdir(s) / f"{k}.npz").exists()]
    print(f"building {len(todo)} tables", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))


def one_year(y):
    import time
    from plan_choice import p_need
    from review import K_GRID, OFF_BOARD_RANK, T_GRID, board, draft_night
    from simulator import situation
    from stack_choice import TDIR, _key
    t0 = time.time()
    picks_df, _ = draft_night(y)
    b = board(y, picks_df)
    B = b[b.grp.isin(GROUPS)].reset_index(drop=True)
    B["rank"] = B["rank"].clip(upper=OFF_BOARD_RANK).astype(float)
    x, yk = np.log(T_GRID + 10), np.log(K_GRID + 10)

    def bilin(A, rank, pick):
        t = rank if pick <= rank else np.exp(np.log(rank + 10) + FW * (np.log(pick + 10) - np.log(rank + 10))) - 10
        lt, lk = np.log(t + 10), np.log(pick + 10)
        i = int(np.clip(np.searchsorted(x, lt) - 1, 0, len(x) - 2)); j = int(np.clip(np.searchsorted(yk, lk) - 1, 0, len(yk) - 2))
        fx = np.clip((lt - x[i]) / (x[i + 1] - x[i]), 0, 1); fy = np.clip((lk - yk[j]) / (yk[j + 1] - yk[j]), 0, 1)
        return (1 - fx) * (1 - fy) * A[..., i, j] + fx * (1 - fy) * A[..., i + 1, j] + (1 - fx) * fy * A[..., i, j + 1] + fx * fy * A[..., i + 1, j + 1]

    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year == y]
    grp, rank, dat = B.grp.to_numpy(), B["rank"].to_numpy(float), B.drafted_at.to_numpy(float)
    dirs = {"B": TDIR, **{("T" if s == 1 else f"T/{int(1 / s)}"): tdir(s) for s in SCALES}}
    rows = []
    for _, srow in S.iterrows():
        tm, sit = srow.team, situation(srow)
        picks = sorted(int(k) for k in picks_df[picks_df.team == tm].pick)
        if not picks:
            continue
        p = p_need(srow.status, "prob") if sit["qb_from"] is not None else 1.0
        for v, dr in dirs.items():
            z = np.load(dr / f"{_key(sit)}.npz")
            V = {g: z[g][:3] for g in z.files}
            taken = np.zeros(len(B), bool)
            count = {}
            for k in picks:
                avail = np.where(~taken & ~(np.nan_to_num(dat, nan=1e9) < k))[0]
                if not len(avail):
                    continue
                vals = []
                for i in avail:
                    g = grp[i]
                    if g == "QB":
                        vals.append(bilin(V["QB"][1], rank[i], k) if count.get("QB") else
                                    p * bilin(V["QB"][0], rank[i], k) + (1 - p) * bilin(V["QB"][1], rank[i], k))
                    else:
                        vals.append(bilin(V[g][min(count.get(g, 0), 2)], rank[i], k))
                bi = avail[int(np.argmax(vals))]
                taken[bi] = True
                count[grp[bi]] = count.get(grp[bi], 0) + 1
                rows.append({"version": v, "year": y, "team": tm, "pick": k, "player": B.loc[bi, "name"],
                             "grp": grp[bi], "rank": rank[bi], "drafted_at": dat[bi]})
    print(f"{y}: {len(rows)} choices, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def choose(workers):
    from concurrent.futures import ProcessPoolExecutor
    import plan_choice as pc
    with ProcessPoolExecutor(workers) as ex:
        P = pd.concat(list(ex.map(one_year, YEARS)), ignore_index=True)
    P = pc.score(P).drop(columns=["t_family"])
    P.to_parquet(OUT / "tilt_choice.parquet")
    report(P)


def report(P):
    U = pd.read_parquet(RAW / "draft_picks.parquet")
    rnd = dict(zip(zip(U.season, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    M = [("t_single", "titles (label currency)"), ("av", "PFR AV"), ("gs", "seasons started"),
         ("pay", "2nd-contract pay, % cap")]
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
    for v in [v for v in P.version.unique() if v != "B"]:
        print(f"\n{v} - B per team-draft ({P.groupby(['year', 'team']).ngroups}):")
        for m, lab in M:
            parts = []
            for gl, f in G:
                x = {w: P[(P.version == w) & f(P["round"])].groupby(["year", "team"])[m].sum() for w in (v, "B")}
                idx = x[v].index.union(x["B"].index)
                parts.append(f"{gl} {boot(x[v].reindex(idx, fill_value=0) - x['B'].reindex(idx, fill_value=0))}")
            print(f"  {lab:26s} " + " | ".join(parts))
        a = P[P.version == v].set_index(["year", "team", "pick"])
        same = a.player == base.player.reindex(a.index)
        print(f"  same choice as B: {same.mean():.0%} (R1 {same[a['round'] == 1].mean():.0%})")
    P["rg"] = pd.cut(P["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    for v in P.version.unique():
        Q = P[P.version == v]
        print(f"-- {v} position mix (% of picks)\n" + (pd.crosstab(Q.rg, Q.grp, normalize="index") * 100).round(0).to_string())
        first3 = Q.sort_values("pick").groupby(["year", "team"]).head(3)
        n = first3.groupby(["year", "team"]).grp.apply(lambda s: (s == "WR").sum())
        print(f"   WRs among first 3 picks: mean {n.mean():.2f}, 2+ in {np.mean(n >= 2):.0%} of drafts")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "report"
    w = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 6
    if step == "build":
        build(w)
    elif step == "choose":
        choose(w)
    else:
        report(pd.read_parquet(OUT / "tilt_choice.parquet"))
