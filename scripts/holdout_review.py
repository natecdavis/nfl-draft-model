"""Past drafts review with the outcome engine held out from the draft being valued.

The live engine (outcomes_v2) is fit on all classes through 2025, so when the review values a 2017
pick, 2017's own careers (Watson, Mahomes) are among the comparison careers. Two honest versions:

  lodo    leave-one-draft-out: the engine for draft Y is fit without class Y (2014-2022)
  expand  expanding window: only what was known on draft day Y, i.e. classes before Y and seasons
          through Y-1; comparison careers need 4 seasons, so classes <= Y-4 (2019-2022 only: earlier
          drafts have almost no comparison careers)

Everything else as the review (src/review.py). Not held out (small, single-number settings): fall
shading 0.85, QB need rates, the 1.10 units factor; position blend weights are leave-one-draft-out
in both versions. Realized value doesn't use the engine. Outputs go to data/derived/holdout/, the
production review files are not touched.

Run:  python scripts/holdout_review.py [workers=7]      report only: ... report
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT, RAW  # noqa: E402

import os  # noqa: E402

HD = OUT / "holdout"
FW = os.environ.get("REVIEW_FALL_WEIGHT")  # set: fall-shading test, leave-one-draft-out only
CR = os.environ.get("COST_RATE")  # set: cap-cost test (pick_value.COST_RATE)
WF = os.environ.get("WR_FACTOR")  # set: WR value test (pick_value.WR_FACTOR)
SUF = (f"_fw{FW}" if FW else "") + (f"_cost{CR}" if CR else "") + (f"_wr{WF}" if WF else "")
FW = FW or CR or WF  # either test: leave-one-draft-out only
JOBS = [("lodo", y) for y in range(2014, 2023)] + ([] if FW else [("expand", y) for y in range(2019, 2023)])


def qb_ev():
    import outcomes_v2 as o
    m = o.model()
    return {r: float((lambda pw: (pw[1] * pw[0].sum(1)).sum())(m._near("QB", r))) for r in (5, 15, 30, 60)}


def one_job(job):
    import outcomes_v2 as o
    import review as R
    from trade import Context
    t0 = time.time()
    o.HOLDOUT, o._MODEL = job, None
    ctx, states, tables = Context(), R.States(), {}
    df, st, _ = R.review_year(job[1], states, ctx, tables)
    return job, df.assign(mode=job[0]), st, qb_ev(), time.time() - t0


def main(workers=7):
    HD.mkdir(exist_ok=True)
    t0 = time.time()
    res, ev = {}, []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(one_job, j) for j in sorted(JOBS, key=lambda j: j[0] != "expand")]
        for f in as_completed(futs):
            job, df, st, q, secs = f.result()
            res[job] = (df, st)
            ev.append({"mode": job[0], "year": job[1], **{f"qb_ev_{r}": v for r, v in q.items()}})
            print(f"{job}: {len(df)} picks, {secs:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    import outcomes_v2 as o
    o.HOLDOUT, o._MODEL = None, None
    ev.append({"mode": "full", "year": 0, **{f"qb_ev_{r}": v for r, v in qb_ev().items()}})
    if not FW:
        pd.DataFrame(ev).to_parquet(HD / "qb_ev.parquet")
    import review as R
    from trade import Context
    for mode in sorted({k[0] for k in res}):
        ks = [k for k in res if k[0] == mode]
        df = pd.concat([res[k][0] for k in ks], ignore_index=True)
        st = pd.concat([res[k][1] for k in ks], ignore_index=True)
        df = R.add_realized_v2(df, st, Context())
        df.to_parquet(HD / f"review_picks_{mode}{SUF}.parquet")
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)
    report()


def report():
    full = pd.read_parquet(OUT / "review_picks.parquet").assign(mode="full")
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    get = lambda d, y, a: 0.0 if pd.isna(a) else float(np.nan_to_num(d.get((y, int(a)), 0.0)))
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        if len(x) == 0:
            return "n/a"
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f}) n={len(x)}"

    def card(d, label):
        d = d[d.real_team.notna()].copy()
        once = d[d.player != d.rec_player].drop_duplicates(["year", "rec_player"]).copy()
        once["d_t"] = once.real_model - once.real_team
        once["d_gs"] = [get(gs, y, a) - get(gs, y, k) for y, a, k in zip(once.year, once.rec_drafted_at, once.pick)]
        once["d_av"] = [get(av, y, a) - get(av, y, k) for y, a, k in zip(once.year, once.rec_drafted_at, once.pick)]
        print(f"  {label}")
        for gl, f in [("R1", once["round"] == 1), ("R2-4", once["round"].between(2, 4)), ("R5-7", once["round"] >= 5)]:
            o = once[f]
            print(f"    {gl:5s} titles {boot(o.d_t)} | starts {o.d_gs.mean():+.2f} | AV {o.d_av.mean():+.1f}"
                  f" | QB recs {(o.rec_grp == 'QB').mean():.0%}")

    for mode, years in [("lodo", range(2014, 2023)), ("expand", range(2019, 2023))]:
        f = HD / f"review_picks_{mode}.parquet"
        if not f.exists():
            continue
        h = pd.read_parquet(f)
        print(f"\n=== drafts {years[0]}-{years[-1]}: model minus team where they disagree, each rec once ===")
        F = full[full.year.isin(years)]
        card(F, "full engine (in-sample)")
        if mode == "expand":
            card(pd.read_parquet(HD / "review_picks_lodo.parquet").query("year >= 2019"), "leave-one-draft-out")
        card(h[h.year.isin(years)], f"{mode}")
        m = F.merge(h, on=["year", "pick"], suffixes=("", "_h"))
        print(f"  same recommendation as full engine: {(m.rec_player == m.rec_player_h).mean():.0%}"
              f" (R1 {(m[m['round'] == 1].rec_player == m[m['round'] == 1].rec_player_h).mean():.0%})")
    print("\nQB expected value (cap share, 4 seasons) at consensus rank 5 / 15 / 30 / 60:")
    q = pd.read_parquet(HD / "qb_ev.parquet").sort_values(["mode", "year"])
    print(q.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
