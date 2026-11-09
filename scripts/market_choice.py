"""Market-calibrated position factors in the whole-draft choice test, drafts 2014-2022.

  B  today's model (greedy, single units, QB need probability, fall shading once), tables rebuilt on the
     narrow label (data/derived/units_tables_narrow)
  M  the same with outcomes_v2.POS_PAY="market": each position's pay-curve value scaled so the average
     absent starter matches the betting market's starter-minus-backup estimate (QB 0.89, RB 0.51, WR 0.95,
     TE 1.06, OL 0.94, IDL 0.65, ED 0.42, LB 1.21, CB 0.82, S 0.65; scripts/absence_curve.py factors)
     (data/derived/market_tables)

Same setup and yardsticks as scripts/plan_choice.py, plus realized titles with the market factors.

Run:  python scripts/market_choice.py build base|market [workers]   (each in its own process)
      python scripts/market_choice.py choose [workers]
      python scripts/market_choice.py report
"""
import os
import sys

STEP = sys.argv[1] if len(sys.argv) > 1 else "report"
MODE = sys.argv[2] if STEP == "build" else "base"
os.environ["OUTCOME_UNITS"] = "single"
os.environ["POS_PAY"] = "market" if MODE == "market" else "off"
if MODE == "repl":
    os.environ["LABEL"] = "repl"
SHRINK = os.environ.get("MARKET_SHRINK", "")
MKT_DIR = "market_tables" + (f"_s{SHRINK}" if SHRINK else "")
VARIANT = os.environ.get("CHOICE_VARIANT", "market")  # which alternative the choose/report steps compare
ALT_DIR = MKT_DIR if VARIANT == "market" else "repl_tables"
os.environ["NEED_TABLE_DIR"] = {"market": MKT_DIR, "repl": "repl_tables"}.get(MODE, "units_tables_narrow")
os.environ["NEED_LEVELS"] = "current"
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plan_choice as pc  # noqa: E402
from need_choice import TDIR, _key, build_table  # noqa: E402
from outlook import OUT  # noqa: E402

VERSIONS = {"B": ("greedy", "single_narrow", "prob"), "M": ("greedy", "single_market", "prob")}


def _init():
    pc.VERSIONS = VERSIONS
    pc.TABLE_DIR["single_narrow"] = OUT / "units_tables_narrow"
    pc.TABLE_DIR["single_market"] = OUT / ALT_DIR


def sits():
    from simulator import situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(pc.YEARS)]
    return {_key(situation(r)): situation(r) for _, r in S.iterrows()}


def build(workers):
    from concurrent.futures import ProcessPoolExecutor
    todo = [s for k, s in sits().items() if not (TDIR / f"{k}.npz").exists()]
    print(f"building {len(todo)} tables in {TDIR.name} (POS_PAY={os.environ['POS_PAY']})", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))


def choose(workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        P = pd.concat(list(ex.map(pc.one_year, pc.YEARS)), ignore_index=True)
    _init()
    import outcomes_v2
    outcomes_v2.POS_PAY = "off"
    P = pc.score(P)  # t_single / t_family without factors, AV, starts, 2nd-contract pay
    from review import RealizedV2
    import prospect_model as pm
    if VARIANT == "repl":
        pm.LABEL_FILE = OUT / "labels_pickadj_narrow_repl.parquet"
    else:
        outcomes_v2.POS_PAY = "market"
    from simulator import situation
    from trade import Context
    St = pd.read_parquet(OUT / "review_states.parquet")
    sit = {(r.year, r.team): situation(r) for _, r in St.iterrows()}
    R = RealizedV2(Context(), units="single", pos_pay=VARIANT == "market")
    P["t_market"] = [R.titles(sit[(y, tm)], g, y, at, k, nm) for y, tm, g, at, k, nm in
                     zip(P.year, P.team, P.grp, P.drafted_at, P.pick, P.player)]
    P.to_parquet(OUT / f"{VARIANT}_choice{'_s' + SHRINK if SHRINK else ''}.parquet")
    report(P)


def report(P):
    U = pd.read_parquet(OUT / "units_choice.parquet")
    rnd = dict(zip(zip(U.year, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    M = [("t_market", f"titles, {VARIANT} scoring"), ("t_single", "titles, current"), ("av", "PFR AV"),
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
    print("\nM - B per team-draft (288):")
    for m, lab in M:
        parts = []
        for gl, f in G:
            x = {v: P[(P.version == v) & f(P["round"])].groupby(["year", "team"])[m].sum() for v in ("M", "B")}
            idx = x["M"].index.union(x["B"].index)
            parts.append(f"{gl} {boot(x['M'].reindex(idx, fill_value=0) - x['B'].reindex(idx, fill_value=0))}")
        print(f"  {lab:26s} " + " | ".join(parts))
    a = P[P.version == "M"].set_index(["year", "team", "pick"])
    b = P[P.version == "B"].set_index(["year", "team", "pick"])
    print(f"  same choice as B: {(a.player == b.player.reindex(a.index)).mean():.0%}")
    P["rg"] = pd.cut(P["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    for v in VERSIONS:
        Q = P[P.version == v]
        print(f"-- {v} position mix (% of picks)\n" + (pd.crosstab(Q.rg, Q.grp, normalize="index") * 100).round(0).to_string())


if __name__ == "__main__":
    w = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 6
    if STEP == "build":
        build(w)
    elif STEP == "choose":
        choose(w)
    else:
        report(pd.read_parquet(OUT / f"{VARIANT}_choice{'_s' + SHRINK if SHRINK else ''}.parquet"))
