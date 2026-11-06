"""Compare leave-one-draft-out reviews run with WR value at 1.0 and 0.75 of other non-QBs (pick_value.WR_FACTOR).

Versions: 1.0 (data/derived/review_picks.parquet, cap charge 0.10) and 0.75 (holdout/
review_picks_lodo_wr0.75.parquet, from WR_FACTOR=0.75 scripts/holdout_review.py). Each version's
recommendations are scored on common yardsticks: realized titles with WR value at 1.0 and at
0.75 (RealizedV2 recomputed in this process for both), seasons started, PFR AV and second-contract
pay (% of cap), model minus team where they disagree, each recommendation once per draft, 2014-2022.

Run:  python scripts/wr_compare.py
"""
import importlib
import os
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT, RAW  # noqa: E402

VERSIONS = {"1.0": OUT / "review_picks.parquet", "0.75": OUT / "holdout/review_picks_lodo_wr0.75.parquet"}


def realized(df, rate):
    """real_team / real_model with WR factor `rate`."""
    import pick_value
    os.environ["WR_FACTOR"] = str(rate)
    importlib.reload(pick_value)
    import review
    importlib.reload(review)
    from trade import Context
    st = pd.read_parquet(OUT / "review_states.parquet")
    d = review.add_realized_v2(df.copy(), st[st.year <= 2022], Context())
    return d.real_team.to_numpy(), d.real_model.to_numpy()


def main():
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    look = {"gs": {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)},
            "av": {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)},
            "pay": {(y, p): a * 100 for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}}
    get = lambda d, y, a: 0.0 if pd.isna(a) else float(np.nan_to_num(d.get((y, int(a)), 0.0)))
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"

    keep = ["year", "pick", "round", "team", "player", "grp", "rec_player", "rec_grp", "rec_drafted_at",
            "best_available", "best_available_grp", "best_available_drafted_at"]
    V = {}
    for v, f in VERSIONS.items():
        d = pd.read_parquet(f)
        d = d[d.year <= 2022][keep + ["real_team"]].reset_index(drop=True)
        V[v] = d
    for rate, lab in [(1.0, "t19"), (0.75, "t0")]:
        for v, d in V.items():
            d[f"{lab}_team"], d[f"{lab}_model"] = realized(d[keep], rate)
    rows = []
    for v, d in V.items():
        d = d[d.real_team.notna()]
        o = d[d.player != d.rec_player].drop_duplicates(["year", "rec_player"]).copy()
        for m, dd in look.items():
            o[m] = [get(dd, y, a) - get(dd, y, k) for y, a, k in zip(o.year, o.rec_drafted_at, o.pick)]
        o["t19"] = o.t19_model - o.t19_team
        o["t0"] = o.t0_model - o.t0_team
        print(f"\n=== WR factor {v}: model minus team where they disagree, each rec once ===")
        for g, f in [("R1", o["round"] == 1), ("R2-4", o["round"].between(2, 4)), ("R5-7", o["round"] >= 5)]:
            x = o[f]
            print(f"  {g:5s} n={len(x):3d} | titles (WR at 1.0) {boot(x.t19)} | titles (WR at 0.75) {boot(x.t0)} | "
                  f"starts {boot(x.gs)} | AV {x.av.mean():+.1f} | 2nd-contract pay {boot(x.pay)}")
        r1 = d[d["round"] == 1]
        print(f"  round-1 recs: QB {(r1.rec_grp == 'QB').mean():.0%}; recs by position (all rounds): "
              + str(d.rec_grp.value_counts(normalize=True).head(6).round(2).to_dict()))
        rows.append(o.assign(version=v))
    if "1.0" in V:
        b = V["1.0"].set_index(["year", "pick"]).rec_player
        for v in ("0.75",):
            a = V[v].set_index(["year", "pick"]).rec_player
            print(f"same recommendation as 1.0: {v} {(a == b.reindex(a.index)).mean():.0%}")
    pd.concat(rows).to_parquet(OUT / "holdout/wr_compare.parquet")


if __name__ == "__main__":
    main()
