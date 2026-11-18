"""Compare leave-one-draft-out reviews run with different cap-cost charges (pick_value.COST_RATE).

Versions: 0.19 (the charge before Nov 2026; review_picks.parquet saved as holdout/
review_picks_lodo_cost0.19.parquet), 0.10 and 0 (holdout/review_picks_lodo_cost*.parquet, from
COST_RATE=... scripts/holdout_review.py). 0.10 is now the default (pick_value.COST_RATE). Each version's
recommendations are scored on common yardsticks: realized titles with the old 0.19 charge and with no
charge (RealizedV2 recomputed in this process for both), seasons started, PFR AV and second-contract
pay (% of cap), model minus team where they disagree, each recommendation once per draft, 2014-2022.

Run:  python scripts/cost_compare.py
"""
import importlib
import os
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT, RAW  # noqa: E402

VERSIONS = {"0.19": OUT / "holdout/review_picks_lodo_cost0.19.parquet", "0.10": OUT / "holdout/review_picks_lodo_cost0.1.parquet",
            "0": OUT / "holdout/review_picks_lodo_cost0.parquet"}


def realized(df, rate):
    """real_team / real_model with cap cost `rate`."""
    import pick_value
    os.environ["COST_RATE"] = str(rate)
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
    for rate, lab in [(0.19, "t19"), (0.0, "t0")]:
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
        print(f"\n=== cost charge {v}: model minus team where they disagree, each rec once ===")
        for g, f in [("R1", o["round"] == 1), ("R2-4", o["round"].between(2, 4)), ("R5-7", o["round"] >= 5)]:
            x = o[f]
            print(f"  {g:5s} n={len(x):3d} | titles (0.19 charge) {boot(x.t19)} | titles (no charge) {boot(x.t0)} | "
                  f"starts {boot(x.gs)} | AV {x.av.mean():+.1f} | 2nd-contract pay {boot(x.pay)}")
        r1 = d[d["round"] == 1]
        print(f"  round-1 recs: QB {(r1.rec_grp == 'QB').mean():.0%}; recs by position (all rounds): "
              + str(d.rec_grp.value_counts(normalize=True).head(6).round(2).to_dict()))
        rows.append(o.assign(version=v))
    if "0.19" in V:
        b = V["0.19"].set_index(["year", "pick"]).rec_player
        for v in ("0.10", "0"):
            a = V[v].set_index(["year", "pick"]).rec_player
            print(f"same recommendation as 0.19: {v} {(a == b.reindex(a.index)).mean():.0%}")
    pd.concat(rows).to_parquet(OUT / "holdout/cost_compare.parquet")


if __name__ == "__main__":
    main()
