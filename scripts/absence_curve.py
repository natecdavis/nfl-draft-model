"""Absence test in the rookie-value currency (follow-up to scripts/absence_test.py, carry-forward version).

For each absent starter, what each candidate rule credits him per game, using cfdarko's pay curve
price_g(share, q) / 1.10 with his PRIOR-season quality (neutral if none) and his recent snap share:
  curve_old   = rate x (curve(s, q) - 0.5% x s)            (current pick_effect rule on outcomes_v2 values)
  curve_qor   = 0.48 x (curve(s, q) - curve(s, 0.1))      (pooled quality over replacement, replacement_level.py)
  price_old   = rate x s x (pre-season price - 0.5%)       (absence_test.py, bridge currency)
rate = 0.34 QB, 0.19 otherwise. Calibration slope of the spread (and margin) on each: 1.0 = rule matches.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_units as ru  # noqa: E402
import absence_test as at  # noqa: E402


def main():
    C = ru.capture()
    cur, q, neutral, pos, p2g = C["curves"], C["q"], C["neutral"], C["pos"], C["pfr2gsis"]
    A = pd.read_parquet(at.OUT / "absence_starters_carry.parquet")
    A["gsis"] = A.pfr.map(p2g)
    A["g"] = A.gsis.map(pos)
    ab = A.absent & A.g.isin(cur)
    print(f"absent starters with a pricing group: {ab.sum()} of {A.absent.sum()}")

    def qprior(gs, g, y):
        return q.get((gs, y - 1), neutral.get((g, y), neutral[(g, None)]))
    rate = np.where(A.grp == "QB", 0.34, 0.19)
    cv = np.zeros(len(A)); base = np.zeros(len(A))
    for i in np.where(ab)[0]:
        r = A.iloc[i]
        s = min(r.avg_share, 1.0)
        cv[i] = cur[r.g](s, qprior(r.gsis, r.g, r.season)) / ru.UNITS * 100
        base[i] = cur[r.g](s, ru.Q_REF) / ru.UNITS * 100
    A["curve_old"] = np.where(ab, rate * (cv - 0.5 * A.avg_share), 0)
    A["curve_qor"] = np.where(ab, 0.48 * (cv - base), 0)
    A["price_old"] = np.where(ab, A.model_pts, 0)
    print("\nMean credit per absent starter by group (pts/game): market spread effect from absence_test carry run")
    print(A[ab].groupby("grp")[["price_old", "curve_old", "curve_qor"]].mean().round(2).to_string())

    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    for col in ["price_old", "curve_old", "curve_qor"]:
        M, n, g = at.design(s, A, [col])
        for y in ["spread_line", "result"]:
            f = at.fit(g[y].to_numpy(float), M, n, g.home_team.to_numpy())
            print(f"   {col:9s} {y:11s} slope {-f.b.iloc[0]:.2f} [{-f.hi.iloc[0]:.2f}, {-f.lo.iloc[0]:.2f}]")
    # by position: spread slope per rule within QB / non-QB
    A["qb"] = A.grp == "QB"
    for col in ["curve_old", "curve_qor"]:
        A[f"{col}_qb"] = np.where(A.qb, A[col], 0); A[f"{col}_non"] = np.where(~A.qb, A[col], 0)
        M, n, g = at.design(s, A, [f"{col}_qb", f"{col}_non"])
        f = at.fit(g.spread_line.to_numpy(float), M, n, g.home_team.to_numpy())
        print(f"   {col} spread slope: QB {-f.b.iloc[0]:.2f} [{-f.hi.iloc[0]:.2f},{-f.lo.iloc[0]:.2f}], "
              f"non-QB {-f.b.iloc[1]:.2f} [{-f.hi.iloc[1]:.2f},{-f.lo.iloc[1]:.2f}]")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def factors():
    """Market factor per draft group = market starter-minus-backup / mean current-rule credit (rookie
    currency), by cfdarko pricing group, DB split into CB / S by snap position."""
    C = ru.capture()
    cur, q, neutral, pos, p2g = C["curves"], C["q"], C["neutral"], C["pos"], C["pfr2gsis"]
    A = pd.read_parquet(at.OUT / "absence_starters_carry.parquet")
    A["g"] = A.pfr.map(p2g).map(pos)
    A["key"] = np.where(A.g == "DB", np.where(A.grp == "S", "S", "CB"), A.g)
    A = A[A.g.isin(cur) | ~A.absent]
    keys = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "CB", "S"]
    for k in keys:
        A[f"k_{k}"] = (A.absent & (A.key == k)).astype(float)
    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    M, n, g = at.design(s, A, [f"k_{k}" for k in keys])
    f = at.fit(g.spread_line.to_numpy(float), M, n, g.home_team.to_numpy())
    rate = lambda k: 0.34 if k == "QB" else 0.19
    rows = []
    for k in keys:
        X = A[A.absent & (A.key == k)]
        cr = np.mean([rate(k) * (cur[r.g](min(r.avg_share, 1), q.get((r.gsis, r.season - 1),
                      neutral.get((r.g, r.season), neutral[(r.g, None)]))) / ru.UNITS * 100 - 0.5 * r.avg_share)
                      for r in X.assign(gsis=X.pfr.map(p2g)).itertuples()])
        rows.append({"key": k, "n": len(X), "market": -f.loc[f"k_{k}", "b"], "lo": -f.loc[f"k_{k}", "hi"],
                     "hi": -f.loc[f"k_{k}", "lo"], "credit": cr})
    F = pd.DataFrame(rows).set_index("key")
    F["factor"] = F.market / F.credit
    print(F.round(2).to_string())
    return F


if __name__ == "__main__" and sys.argv[1:2] == ["factors"]:
    factors()
