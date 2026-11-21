"""Realized check of free-agent verdicts the way the tool makes them (complaint #2, Nov 21 2026).

scripts/vet_value.py scores past UFA signings with a forecast from data through the season before signing (lag 1).
Since the season in progress was added (p_share / p_gp of the anchor's own season, veteran.comps), that lag-1 setup
leaks part of contract season 1 into the forecast. The tool plans a year ahead: data through Y-2, weeks 1-11 of Y-1,
contract seasons = comps' seasons 2.. (skip 1). Here each UFA contract signed in Y (2016-23, APY >= 1% of cap, matched in
scripts/vet_value.py part B) is forecast from the player's anchor row for Y-1 with skip 1, scored with
veteran.team_seasons + points (theta 0.5), and compared with what happened (value with the team, APY while there,
dead money proxy). Also "no partial": the same without the season-in-progress features.
By age at signing and by contract size (2b: big contracts read ~0.2 points optimistic in step 1).
Run: .venv/bin/python scripts/fa_realized_asused.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src", "scripts"]
import veteran as vt  # noqa: E402


def score(C, D, partial=True):
    Aidx = C.A.set_index(["pfr_id", "Y"])
    pred = []
    for r in D.itertuples():
        a = Aidx.loc[(r.pfr_id, r.Y - 1)]
        row = {"pfr_id": r.pfr_id, "Y": r.Y - 1, "grp": a.grp, "age": a.age, "v1": a.v1, "v2": a.v2, "v2f": a.v2f,
               "g1": a.g1, "p_share": a.p_share if partial else np.nan, "p_gp": a.p_gp if partial else np.nan}
        V, S = C.paths(row, int(r.n), exclude=True, skip=1)
        val, cost, on = vt.team_seasons(V, int(r.n), r.apy, r.gtd)
        pts = vt.points(val, S, cost, on, a.grp == "QB")
        pred.append((pts.sum(axis=1).mean(), V.mean(axis=0).sum()))
    return np.array(pred)


def main():
    pd.set_option("display.width", 220)
    C = vt.comps()
    F = pd.read_parquet(vt.OUT / "age_forecast_signed_points.parquet")  # finished UFA contracts with realized points
    keys = set(zip(C.A.pfr_id, C.A.Y))
    F = F[F.Y.between(2016, 2023) & np.array([(p, y - 1) in keys for p, y in zip(F.pfr_id, F.Y)])].copy()
    F["n"] = np.minimum(F.yrs, 5).astype(int)
    F = F[F.Y + F.n - 1 <= vt.LAST]
    F["fc_real"] = sum(np.where(F.n >= k, F[f"y{k}"], 0) for k in range(1, 6))  # value anywhere within contract
    for name, part in [("as used", True), ("no partial", False)]:
        P = score(C, F, part)
        F[f"pred_{name}"], F[f"fcv_{name}"] = P[:, 0], P[:, 1]
    F["ab"] = pd.cut(F.age, [0, 28, 31, 33, 99], labels=["<28", "28-30", "31-32", "33+"], right=False)
    F["size"] = pd.cut(F.apy, [0, 0.02, 0.05, 1], labels=["1-2%", "2-5%", "5%+"])
    F["qb"] = np.where(F.grp == "QB", "QB", "non-QB")
    rng = np.random.default_rng(0)

    def summ(d):
        out = {"n": len(d), "real": d.real_pts.mean()}
        for name in ["as used", "no partial"]:
            g = d.real_pts.to_numpy() - d[f"pred_{name}"].to_numpy()
            bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(1000)]
            out[f"pred {name}"] = d[f"pred_{name}"].mean()
            out[f"gap {name}"] = f"{g.mean():+.3f} ({np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f})"
            out[f"value a/f {name}"] = d.fc_real.sum() / d[f"fcv_{name}"].sum()
        return pd.Series(out)

    print(f"UFA contracts signed 2016-23, finished by {vt.LAST}, forecast a year ahead (skip 1): {len(F)}")
    print(summ(F).to_string())
    for col in ["ab", "size", "qb"]:
        print(f"\nby {col}:")
        print(F.groupby(col, observed=True).apply(summ, include_groups=False).round(3).to_string())
    print("\nnon-QB by size:")
    print(F[F.qb == "non-QB"].groupby("size", observed=True).apply(summ, include_groups=False).round(3).to_string())
    F.to_parquet(vt.OUT / "fa_realized_asused.parquet")


if __name__ == "__main__":
    main()
