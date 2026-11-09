"""Position-specific replacement levels fit to the betting market (follow-up to scripts/edge_check.py).

Each absent starter's credit under the current rule (rookie currency): c = rate x (curve_g(s, q_prior)/1.10
- 0.5% x s), rate 0.34 QB / 0.19 otherwise. With an extra replacement level r_g (% of cap per full season
of snaps), his worth over replacement is c - rate x r_g x s. Impose that on the spread:
  spread + sum(c_home) - sum(c_away) = FE + sum_g theta_g x (share_home_g - share_away_g),  r_g = theta_g / rate
(slope on the credit fixed at 1, so within-position differences stay as the curve prices them).
Also a free version (common slope on c) to show sensitivity. Writes data/derived/replacement_market.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_units as ru  # noqa: E402
import absence_test as at  # noqa: E402

GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


def main():
    C = ru.capture()
    cur, q, neutral, pos, p2g = C["curves"], C["q"], C["neutral"], C["pos"], C["pfr2gsis"]
    A = pd.read_parquet(at.OUT / "absence_starters_carry.parquet")
    A["gsis"] = A.pfr.map(p2g)
    A["g"] = A.gsis.map(pos)
    A = A[A.g.isin(cur) | ~A.absent].copy()
    rate = np.where(A.g == "QB", 0.34, 0.19)
    cr = np.zeros(len(A))
    for i in np.where(A.absent)[0]:
        r = A.iloc[i]
        s = min(r.avg_share, 1.0)
        qq = q.get((r.gsis, r.season - 1), neutral.get((r.g, r.season), neutral[(r.g, None)]))
        cr[i] = rate[i] * (cur[r.g](s, qq) / ru.UNITS * 100 - 0.5 * s)
    A["credit"] = cr
    for g in GROUPS:
        A[f"sh_{g}"] = np.where(A.absent & (A.g == g), A.avg_share, 0.0)
    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    cols = [f"sh_{g}" for g in GROUPS]
    M, n, g_ = at.design(s, A, cols + ["credit"])
    credit_diff = M[:, len(cols)]
    rows = []
    for y in ["spread_line", "result"]:
        yy = g_[y].to_numpy(float) + credit_diff
        Mf = np.delete(M, len(cols), axis=1)
        f = at.fit(yy, Mf, [c for c in n if c != "credit"], g_.home_team.to_numpy())
        free = at.fit(g_[y].to_numpy(float), M, n, g_.home_team.to_numpy())
        for gname in GROUPS:
            rt = 0.34 if gname == "QB" else 0.19
            t = f"sh_{gname}"
            rows.append({"outcome": y, "group": gname, "n_absent": int((A.absent & (A.g == gname)).sum()),
                         "r": f.loc[t, "b"] / rt, "lo": f.loc[t, "lo"] / rt, "hi": f.loc[t, "hi"] / rt})
        print(f"[{y}] free version: slope on credit {-free.loc['credit', 'b']:.2f} "
              f"[{-free.loc['credit', 'hi']:.2f},{-free.loc['credit', 'lo']:.2f}]")
    R = pd.DataFrame(rows)
    print("\nExtra replacement level r_g, % of cap per full season of snaps (slope on credit fixed at 1):")
    print(R.pivot(index="group", columns="outcome", values=["r", "lo", "hi"]).round(2).reindex(GROUPS).to_string())
    R.to_csv(at.OUT / "replacement_market.csv", index=False)


if __name__ == "__main__":
    main()
