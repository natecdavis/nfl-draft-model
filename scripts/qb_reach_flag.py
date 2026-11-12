"""How QB reaches turned out: the lookup behind the app's reach flag (README 'QB valuation').

When the model's top choice is a QB ranked well below the pick on the consensus board, the app shows how such
QBs did. Source: data/derived/holdout/defer_candidates.parquet (leave-one-draft-out engine, 2014-2022): every
available player at every pick a team made, the model's titles for that team then, and what he delivered.

For the model's QB choices (top value at that pick, each player once per draft and bucket), by how far below
the pick he sat on the board (rank - pick) and by his consensus rank: delivered / predicted titles (sum over
players) and the share who became starters (3+ seasons as a starter to date).
Bootstrap over draft years for the ratio.

Run:  python scripts/qb_reach_flag.py      -> data/derived/qb_reach.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402

DIST = [(-1000, 0, "at or above the pick"), (1, 15, "1-15 spots below the pick"), (16, 1000, "16+ spots below the pick")]
RANK = [(1, 32, "ranked 1-32"), (33, 64, "ranked 33-64"), (65, 1000, "ranked 65+")]


def summarize(P, rng):
    """Delivered / predicted (95% over draft years), starter share and the biggest hits for one bucket."""
    yrs = P.year.unique()
    bs = []
    for _ in range(2000):
        s = pd.concat([P[P.year == y] for y in rng.choice(yrs, len(yrs))])
        bs.append(s.real.sum() / max(s.pred.sum(), 1e-9))
    return {"n": len(P), "ratio": float(P.real.sum() / P.pred.sum()), "ratio_lo": float(np.percentile(bs, 2.5)),
            "ratio_hi": float(np.percentile(bs, 97.5)), "starters": float((P.gs.fillna(0) >= 3).mean()),
            "hits": P[P.real > P.pred].sort_values("real", ascending=False).name.head(3).tolist()}


def main():
    C = pd.read_parquet(OUT / "holdout" / "defer_candidates.parquet")
    C = C[C.real.notna()].sort_values(["year", "team", "pick", "titles"], ascending=[True, True, True, False])
    top = C.groupby(["year", "team", "pick"]).head(1)
    q = top[top.grp == "QB"].copy()
    q["dist"] = q["rank"] - q.pick
    rng = np.random.default_rng(0)
    out = {"dist": [], "rank": []}
    for kind, buckets in [("dist", DIST), ("rank", RANK)]:
        for lo, hi, lab in buckets:
            g = q[q[kind].between(lo, hi)]
            P = g.groupby(["year", "name"]).agg(pred=("titles", "mean"), real=("real", "mean"),
                                                 gs=("gs", "first")).reset_index()
            row = {"bucket": lab, "lo": lo, "hi": hi} | summarize(P, rng)
            out[kind].append(row)
            print(f"{lab:26s} n {row['n']:3d}  delivered/predicted {row['ratio']:.2f} ({row['ratio_lo']:.2f}-"
                  f"{row['ratio_hi']:.2f})  starters {row['starters']:.0%}  hits {row['hits']}")
    json.dump(out, open(OUT / "qb_reach.json", "w"), indent=1)


if __name__ == "__main__":
    main()
