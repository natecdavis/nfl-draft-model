"""Within-family position pay (OT / OG / C, CB / S) in the whole-draft choice test, drafts 2014-2022.

  B  today's model: greedy, single units, QB need probability, fall shading once (tables:
     data/derived/units_tables)
  P  the same with POS_PAY on: a season's value scaled by the position's pay relative to its family
     (OT 1.27, OG 0.86, C 0.77, CB 1.11, S 0.89; outcomes_v2.POS_PAY_FACTOR) (tables: paypos_tables)

Same setup and yardsticks as scripts/plan_choice.py (the model drafts each team's whole draft at its
real picks, other teams as they really drafted), plus realized titles with the pay factors.

Run:  python scripts/paypos_choice.py [workers=6]      report only: ... report
"""
import os
import sys

os.environ["OUTCOME_UNITS"] = "single"
MODE = os.environ.get("PAYPOS_MODE", "pay")  # "pay": version P; "on": version Q (pay x turnout)
os.environ["POS_PAY"] = MODE
os.environ["NEED_TABLE_DIR"] = "paypos_tables" if MODE == "pay" else "paypos2_tables"
os.environ["NEED_LEVELS"] = "current"
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plan_choice as pc  # noqa: E402
from need_choice import TDIR, _key, build_table  # noqa: E402
from outlook import OUT  # noqa: E402

VERSIONS = {"B": ("greedy", "single", "prob"), "P": ("greedy", "single_pay", "prob"),
            "Q": ("greedy", "single_paytn", "prob")}


def _init():
    pc.VERSIONS = VERSIONS
    pc.TABLE_DIR["single_pay"] = OUT / "paypos_tables"
    pc.TABLE_DIR["single_paytn"] = OUT / "paypos2_tables"


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    from simulator import situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(pc.YEARS)]
    sits = {_key(situation(r)): situation(r) for _, r in S.iterrows()}
    todo = [s for k, s in sits.items() if not (TDIR / f"{k}.npz").exists()]
    print(f"building {len(todo)} tables in {TDIR.name}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        P = pd.concat(list(ex.map(pc.one_year, pc.YEARS)), ignore_index=True)
    _init()
    import outcomes_v2
    outcomes_v2.POS_PAY = "off"  # t_single / t_family scored without pay factors; t_pay with
    P = pc.score(P)  # t_single / t_family without pay factors, AV, starts, 2nd-contract pay
    from review import RealizedV2
    from simulator import situation as sit_of
    from trade import Context
    St = pd.read_parquet(OUT / "review_states.parquet")
    sit = {(r.year, r.team): sit_of(r) for _, r in St.iterrows()}
    R = RealizedV2(Context(), units="single", pos_pay=True)
    P["t_pay"] = [R.titles(sit[(y, tm)], g, y, at, k, nm) for y, tm, g, at, k, nm in
                  zip(P.year, P.team, P.grp, P.drafted_at, P.pick, P.player)]
    P.to_parquet(OUT / "paypos_choice.parquet")  # versions B, P, Q
    report(P)


def report(P):
    U = pd.read_parquet(OUT / "units_choice.parquet")
    rnd = dict(zip(zip(U.year, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    M = [("t_pay", "titles, with position pay"), ("t_single", "titles, without"), ("av", "PFR AV"),
         ("gs", "seasons started"), ("pay", "2nd-contract pay, % cap")]
    for m, _ in M:
        P[m] = P[m].fillna(0.0)
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"

    G = [("All", lambda r: r > 0), ("R1", lambda r: r == 1), ("R2-4", lambda r: (r >= 2) & (r <= 4)), ("R5-7", lambda r: r >= 5)]
    for v1 in ("P", "Q"):
        print(f"\n{v1} - B per team-draft (288), raw scoring:")
        for m, lab in M:
            parts = []
            for gl, f in G:
                x = {v: P[(P.version == v) & f(P["round"])].groupby(["year", "team"])[m].sum() for v in (v1, "B")}
                idx = x[v1].index.union(x["B"].index)
                parts.append(f"{gl} {boot(x[v1].reindex(idx, fill_value=0) - x['B'].reindex(idx, fill_value=0))}")
            print(f"  {lab:28s} " + " | ".join(parts))
        a, b = P[P.version == v1].set_index(["year", "team", "pick"]), P[P.version == "B"].set_index(["year", "team", "pick"])
        print(f"  same choice as B: {(a.player == b.player.reindex(a.index)).mean():.0%}")
    P["rg"] = pd.cut(P["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    for v in VERSIONS:
        Q = P[P.version == v]
        print(f"-- {v} position mix (% of picks)\n" + (pd.crosstab(Q.rg, Q.grp, normalize="index") * 100).round(0).to_string())


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "paypos_choice.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
