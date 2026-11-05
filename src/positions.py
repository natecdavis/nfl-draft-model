"""Board position labels, valued as a blend over the NFL position families they historically moved to.

Before the draft, a prospect's position is the consensus board's label (Arif Hasan's, e.g. "DL5T",
"ER", "OB"). 4.9% of drafted board players were later priced in a different NFL family than that
label (5-techniques: ~40% became EDGE; edge rushers -> LB; off-ball LB -> EDGE; S -> LB). A board
player is valued as  sum over families f of  P(f | label) x value at f's position,  with P from drafted
board players 2014-2024 who played (cfdarko's NFL pricing group), leaving out the draft being valued,
plus 5 pseudo-players in the label's own family. Test: scripts/posproj_choice.py (README): changes ~17%
of whole-draft choices, at least as good as single labels, and removes the hindsight of using the
position a player's team listed when it drafted him.

The need level of a blended player is that of his board-label position.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from outlook import GROUPS, OUT, RAW

FAM = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "ED": "EDGE", "IDL": "DL",
       "LB": "LB", "CB": "DB", "S": "DB"}
TARGET = {"EDGE": "ED", "DL": "IDL", "LB": "LB", "DB": "S", "OL": "G", "RB": "RB", "WR": "WR", "TE": "TE", "QB": "QB"}
PRIOR = 5.0
MOVES = OUT / "position_moves.parquet"
YEARS = range(2014, 2025)


def moves() -> pd.DataFrame:
    """Drafted board players who played in the NFL: year, board label, NFL family priced at."""
    if MOVES.exists():
        return pd.read_parquet(MOVES)
    from review import board, draft_night
    L = pd.read_parquet(Path.home() / "cfdarko/data/processed/rookie_value_labels.parquet").dropna(subset=["pfr_id"])
    L = L.drop_duplicates("pfr_id").set_index("pfr_id")
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pid = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    rows = []
    for y in YEARS:
        pk, _ = draft_night(y)
        b = board(y, pk)
        for pos, at in zip(b.pos, b.drafted_at):
            p_ = pid.get((y, int(at))) if pd.notna(at) else None
            if p_ in L.index and (np.nan_to_num(L.at[p_, "value_y1"]) + np.nan_to_num(L.at[p_, "value_y2"])) > 0:
                rows.append({"year": y, "pos": pos, "nfl": L.at[p_, "group"]})
    M = pd.DataFrame(rows)
    M.to_parquet(MOVES)
    return M


@lru_cache(maxsize=None)
def weights(leave_out: int | None = None) -> dict:
    """{board label: {position group: weight}}, from drafts other than `leave_out`."""
    from review import HASAN_POS
    M = moves()
    d = M[M.year != leave_out]
    out = {}
    for pos, g in d.groupby("pos"):
        own_grp = HASAN_POS.get(pos)
        own = FAM.get(own_grp)
        if own is None:
            continue
        c = g.nfl.value_counts().to_dict()
        c[own] = c.get(own, 0) + PRIOR
        w = {}
        for f, n in c.items():
            tgt = own_grp if f == own else TARGET.get(f)
            if tgt in GROUPS:
                w[tgt] = w.get(tgt, 0) + n
        s = sum(w.values())
        out[pos] = {k: v / s for k, v in w.items()}
    return out


def blended(lookup, label, grp, level, rank, pick, p_need=1.0, W=None) -> float:
    """lookup(g, level, rank, pick, p_need) blended over the groups `label` moved to (else at grp)."""
    w = (W or {}).get(label)
    if not w or (len(w) == 1 and grp in w):
        return lookup(grp, level, rank, pick, p_need)
    return sum(v * lookup(g, level, rank, pick, p_need) for g, v in w.items())
