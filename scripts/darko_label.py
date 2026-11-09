"""Narrow label with the season talent model's quality in place of one season's shrunk PFF grade.

Same harness as scripts/pickadj_label.py narrow (cfdarko's build_label, read only, plus the years 2-4
pick-attributable correction, re-estimated here because it controls for q). Only the quality table changes:
  q[(gsis, season)] = within-(pricing group x season) percentile of talent_win from scripts/darko_pff.py
  (Kalman-smoothed talent; rookie-deal seasons look ahead only to the end of year 4, veteran seasons use all
  seasons; no context adjustment by default, which tied in the movers test). Rows count toward a pricing group
  when they come from the same PFF grade table (as in cfdarko's quality_table_pff). The entrant prior is one
  group-wide mean: draft slot never enters the label's quality.
  neutral (seasons with snaps but no PFF row) = the PFF q's neutral, unchanged.
Veteran seasons get the new q too, so the pay curves are refit in the new currency.

Usage: python scripts/darko_label.py [COLUMN]   (default talent_win; e.g. talent_win_ctx)
Writes data/derived/labels_pickadj_narrow_darko{_ctx}.parquet (prospect_model LABEL="darko" / "darko_ctx").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COL = sys.argv[1] if len(sys.argv) > 1 else "talent_win"
sys.argv = [sys.argv[0], "narrow"]  # pickadj_label reads its spec from argv at import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pickadj_label as pa  # noqa: E402

go, rv = pa.go, pa.rv
PFF_QUALITY = go._quality  # cfdarko's original quality_table_pff, saved before the swap
ROOT = Path(__file__).resolve().parents[1]
TALENT = ROOT / "data/derived/darko_pff.parquet"
TAG = "" if COL == "talent_win" else "_" + COL.replace("talent_win_", "")
OUT = ROOT / "data/derived" / f"labels_pickadj_narrow_darko{TAG}.parquet"
TABLE = {"QB": "pass", "RB": "rush", "WR": "rec", "TE": "rec", "OL": "block",
         **{g: "def" for g in ["DL", "EDGE", "LB", "DB"]}}


def darko_quality(pos):
    t = pd.read_parquet(TALENT)
    pl = pd.read_csv(rv.CACHE / "nflverse_players.csv", low_memory=False, usecols=["gsis_id", "pff_id"]).dropna()
    pl = pl.assign(pff_id=pl.pff_id.astype(int)).drop_duplicates("pff_id")
    t = t.merge(pl, left_on="player_id", right_on="pff_id")
    t["pg"] = t.gsis_id.map(pos)
    t = t[t.pg.notna() & (t.g.map(TABLE) == t.pg.map(TABLE)) & t[COL].notna()]
    t = t[t.season.between(rv.FIRST_CLASS, rv.LAST_SEASON)]
    t = t.sort_values("snaps").drop_duplicates(["gsis_id", "season"], keep="last")
    q, neutral = {}, {}
    for (g, y), G in t.groupby(["pg", "season"]):
        ref = np.sort(G[COL].to_numpy())
        pct = lambda x: np.searchsorted(ref, x, side="right") / len(ref)
        q.update(zip(zip(G.gsis_id, G.season), pct(G[COL].to_numpy())))
        neutral[(g, y)] = float(pct(0.0))
    for g in rv.PAY_GROUPS:
        v = [x for (gg, _), x in neutral.items() if gg == g]
        neutral[(g, None)] = float(np.mean(v)) if v else 0.5
    # seasons with snaps but no PFF row (~4% of draftee seasons, mean share 0.10) keep the PFF q's neutral:
    # percentile of 0 here sits higher (0.68-0.83 vs 0.56-0.68) because low-snap talent shrinks toward a
    # below-average entrant mean, so the change stays confined to measured seasons
    q0, n0 = PFF_QUALITY(pos)
    n_new = {g: round(neutral[(g, None)], 3) for g in rv.PAY_GROUPS}
    neutral = n0
    common = [k for k in q if k in q0]
    a, b = np.array([q[k] for k in common]), np.array([q0[k] for k in common])
    print(f"\nDARKO QUALITY ({COL}): {len(q):,} player-seasons ({len(common):,} shared with the PFF q); "
          f"corr with PFF q {np.corrcoef(a, b)[0, 1]:.3f}; neutral kept from PFF q (talent-0 percentile would be {n_new})")
    return q, neutral


def main():
    go._quality = darko_quality  # go.quality_capture (used by pickadj) calls go._quality
    pa.OUT = OUT
    pa.main()


if __name__ == "__main__":
    main()
