"""Run the past-drafts review with one draft per worker process, then combine.

Same results as `python src/review.py` (each draft's value tables are built from scratch in
its own process instead of shared, so it uses more CPU in total but finishes sooner).
Usage: python scripts/review_parallel.py [workers=6] [first last]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "src")
import pandas as pd


def one_year(y):
    import review as R
    from trade import Context
    t0 = time.time()
    ctx, states, tables = Context(), R.States(), {}
    df, st, night = R.review_year(y, states, ctx, tables)
    tr = R.score_trades(y, night, st, ctx, R.draft_night(y)[0])
    return y, df, st, tr, time.time() - t0


def one_year_trades(y):
    import review as R
    from trade import Context
    picks = pd.read_parquet(R.OUT / "review_picks.parquet")
    states = pd.read_parquet(R.OUT / "review_states.parquet")
    return y, R.trade_decisions(y, picks, states, Context(), {})


def main(workers=6, first=None, last=None):
    import review as R
    from trade import Context
    years = [y for y in R.YEARS if (first is None or y >= first) and (last is None or y <= last)]
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one_year, y) for y in years]):
            y, df, st, tr, secs = f.result()
            res[y] = (df, st, tr)
            print(f"{y}: {len(df)} picks, {len(tr) // 2} pick trades, {secs:.0f}s  (elapsed {time.time() - t0:.0f}s)", flush=True)
    allp = [res[y][0] for y in years]; alls = [res[y][1] for y in years]; allt = [res[y][2] for y in years]
    picks = R.add_outcomes(pd.concat(allp, ignore_index=True))
    picks = R.add_realized_v2(picks, pd.concat(alls, ignore_index=True), Context())
    st = pd.concat(alls, ignore_index=True); st["shift"] = st["shift"].astype(str)
    tr = pd.concat(allt, ignore_index=True)
    for name, new in [("review_picks", picks), ("review_states", st), ("review_trades", tr)]:
        f = R.OUT / f"{name}.parquet"
        old = pd.read_parquet(f) if f.exists() else new.iloc[:0]
        out = pd.concat([old[~old.year.isin(years)], new], ignore_index=True).sort_values(
            [c for c in ["year", "pick", "trade_id", "team"] if c in new.columns])
        out.reset_index(drop=True).to_parquet(f)
    print(f"review saved ({time.time() - t0:.0f}s)", flush=True)
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one_year_trades, y) for y in years]):
            y, d = f.result(); out.append(d)
            print(f"  trade rows {y}: {len(d)}", flush=True)
    df = pd.concat(out, ignore_index=True)
    f = R.OUT / "review_trade_decisions.parquet"
    if f.exists():
        old = pd.read_parquet(f)
        df = pd.concat([old[~old.year.isin(years)], df], ignore_index=True)
    df.sort_values(["year", "pick", "trade_id"]).reset_index(drop=True).to_parquet(f)
    print(f"all saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
