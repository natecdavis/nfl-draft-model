"""Draft simulator: replay one team's past draft, making your own choice at each of its picks.

Same currency and board as the Past drafts review (src/review.py): Arif Hasan's consensus board,
the team's state on draft day, titles over 5 seasons for that team's situation, fall weight 0.85.

Rules (model: outcome units outcomes_v2.UNITS, fall shading once, QB need as a probability):
  - The decision points are the picks the team actually made (including picks it traded for
    on draft night). Trades aren't simulated.
  - Other teams draft exactly as they really did. At pick k, every player actually taken
    before k is gone, including the team's own earlier choices, even if you passed on them.
  - Players you took are gone at your later picks (even if, in reality, they went later).
  - Your choices set your needs: each player you take at a position lowers the value of
    the next one there (after one QB, the QB need is filled), as in the review.
  - Candidates: every board player still available, plus drafted players who weren't on the
    board (valued as ranked just past it).
  - Kickers and punters (src/specialists.py): valued by where they were actually drafted, as the gain of a
    drafted specialist over an undrafted one; only the first at each position counts. Long snappers aren't
    valued.

Value tables (one per team situation, ~16s to build) are cached on disk in
data/derived/sim_tables/. Build them all ahead of time with:
    python src/simulator.py build [workers=6]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from draftday import DraftEngine  # noqa: E402
from outlook import GROUPS, OUT  # noqa: E402
from positions import blended  # noqa: E402
from review import K_GRID, OFF_BOARD_RANK, T_GRID, Table, board, draft_night  # noqa: E402
from specialists import POS_TO_GROUP, SPEC_GROUPS, get_tables as spec_tables, is_spec, signal  # noqa: E402

TABLE_DIR = OUT / "sim_tables"
FALL_WEIGHT = 0.85  # same as the review


def situation(state_row) -> dict:
    """Review state row -> situation dict."""
    sh = state_row["shift"]
    sh = sh if isinstance(sh, dict) else eval(sh)
    q = state_row["qb_from"]
    return {"start": float(state_row["start"]), "shift": {int(k): float(v) for k, v in sh.items()},
            "qb_from": None if pd.isna(q) else int(q)}


def _key(sit, year=None) -> str:
    sh = "_".join(f"{k}:{v:g}" for k, v in sorted(sit["shift"].items())) or "none"
    from outcomes_v2 import COMP_CORRECT, MONOTONE, NO_BOARD_CLASSES, UNITS
    # v2: fall shading applied once (Nov 2026); units = outcome units (outcomes_v2.UNITS)
    import prospect_model as pm
    from review import REVIEW_HOLDOUT
    # past drafts: engine fit without that draft class (review.engine_for), so one table per draft year
    ho = f"_lodo{int(year)}" if year is not None and REVIEW_HOLDOUT and int(year) <= pm.LAST_SEASON_PLAYED else ""
    cc = "" if COMP_CORRECT == "off" else f"_cc{COMP_CORRECT}"  # comparison-career correction (outcomes_v2)
    nb = "_nbdrop" if NO_BOARD_CLASSES == "drop" else ""  # board-less classes left out (outcomes_v2)
    nb += "_mono" + MONOTONE if MONOTONE != "off" else ""  # monotone value in rank (outcomes_v2)
    nb += "_lf" if pm.LINK_FIX == "on" else ""  # training-data link repair (prospect_model)
    nb += "" if pm.LABEL == "stored" else f"_lb{pm.LABEL}"  # outcome label (prospect_model.LABEL)
    from outcomes_v2 import QB_COMPS
    nb += "" if QB_COMPS == "ev" else f"_qbc{QB_COMPS}"  # QB comparison coordinate (outcomes_v2.QB_COMPS)
    return f"start{sit['start']:g}_qb{sit['qb_from']}_shift{sh}_fw{FALL_WEIGHT:g}_v2_{UNITS}{ho}{cc}{nb}"


class _Talent:
    """Stand-in for DraftEngine when a table is loaded from disk (only effective_talent is used)."""
    effective_talent = DraftEngine.effective_talent

    def __init__(self, fall_weight):
        self.fall_weight = fall_weight


def get_table(sit, ctx=None, year=None) -> Table:
    """The value table for this situation in draft `year`: from disk if built, else build (~16s) and save."""
    f = TABLE_DIR / f"{_key(sit, year)}.npz"
    if f.exists():
        z = np.load(f)
        t = Table.__new__(Table)
        t.eng, t.v = _Talent(FALL_WEIGHT), {g: z[g] for g in z.files}
        t.v.update(_spec(sit, year, ctx))
        return t
    from review import engine_for
    from trade import Context
    import outcomes_v2 as o
    prev = o.HOLDOUT
    if year is not None:
        engine_for(year)
    try:
        t = Table(sit, ctx or Context(), fall_weight=FALL_WEIGHT)
    finally:  # leave the live engine as it was (the draft-day app shares this process)
        if o.HOLDOUT != prev:
            o.HOLDOUT, o._MODEL = prev, None
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **t.v)
    tmp.replace(f)
    t.v.update(_spec(sit, year, ctx))
    return t


_CTX = []


def _spec(sit, year, ctx=None) -> dict:
    """Kicker / punter arrays for this situation (specialists.py; small, cached as spec_<key>.npz)."""
    def factory():
        if ctx is not None:
            return ctx
        if not _CTX:
            from trade import Context
            _CTX.append(Context())
        return _CTX[0]
    return spec_tables(TABLE_DIR, _key(sit, year), sit, factory, T_GRID, K_GRID)


def pool(year: int) -> pd.DataFrame:
    """Everyone who can be chosen in this draft: the consensus board plus drafted players
    who weren't on it. Columns: pid, name, pos (board label, or draft position if off board), grp
    (decisions: the board's position), grp_record (what he delivered: his team's listed position once
    drafted), label (board label; None off board), rank (NaN = off board), school, drafted_at (NaN =
    undrafted), drafted_by."""
    picks, _ = draft_night(year)
    b = board(year, picks)
    by = dict(zip(picks.pick, picks.team))
    on = pd.DataFrame({"name": b["name"].str.replace(r"\s*\(.*?\)", "", regex=True), "pos": b.pos, "grp": b.grp_label,
                       "grp_record": b.grp, "label": b.pos, "rank": b["rank"].astype(float), "school": b.school, "drafted_at": b.drafted_at})
    off = picks[~picks.pick.isin(b.drafted_at.dropna())]
    dp = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "raw" / "draft_picks.parquet")
    college = dict(zip(dp[dp.season == year].pick, dp[dp.season == year].college))
    off = pd.DataFrame({"name": off.pfr_player_name, "pos": off.position, "grp": off.grp,
                        "grp_record": off.grp, "label": None, "rank": np.nan,
                        "school": off.pick.map(college), "drafted_at": off.pick.astype(float)})
    p = pd.concat([on, off], ignore_index=True)
    sp = p.pos.astype(str).str.upper().map(POS_TO_GROUP)  # kickers / punters (long snappers stay SPEC)
    p.loc[sp.notna(), "grp"] = sp[sp.notna()]
    p.loc[sp.notna(), "grp_record"] = sp[sp.notna()]
    p["drafted_by"] = p.drafted_at.map(by)
    p = p.sort_values(["rank", "drafted_at"], na_position="last").reset_index(drop=True)
    p.insert(0, "pid", np.arange(len(p)))
    return p


def options(P: pd.DataFrame, table: Table, pick: int, taken: set, counts: dict, p_need: float = 1.0,
            W: dict | None = None, specialists: bool = False) -> pd.DataFrame:
    """Players available at `pick` with their titles for this team (in % of one title),
    best first. taken = pids you already chose; counts = your players per position group;
    p_need = chance the team's QB need is real (review.p_need); W = positions.weights(year)
    (board labels valued as a blend over the positions they historically moved to). specialists: include
    kickers and punters (the simulator pages; off by default so research scripts keep their results)."""
    ok = GROUPS + (SPEC_GROUPS if specialists else [])
    a = P[(P.drafted_at.isna() | (P.drafted_at >= pick)) & ~P.pid.isin(list(taken)) & P.grp.isin(ok)].copy()
    r = pd.Series([signal(at, rk, OFF_BOARD_RANK) if is_spec(g) else (OFF_BOARD_RANK if pd.isna(rk) else rk)
                   for g, at, rk in zip(a.grp, a.drafted_at, a["rank"])], index=a.index)
    a["titles"] = [blended(table.lookup, lab, g, min(counts.get(g, 0), 2), rk, pick, p_need, W) * 100
                   for lab, g, rk in zip(a.label, a.grp, r)]
    a = a.sort_values("titles", ascending=False).reset_index(drop=True)
    a.insert(0, "model_rank", np.arange(1, len(a) + 1))
    return a


def value_of(table, grp, rank, pick, counts, p_need=1.0, label=None, W=None, drafted_at=np.nan) -> float:
    if is_spec(grp):
        return table.lookup(grp, min(counts.get(grp, 0), 2), signal(drafted_at, rank, OFF_BOARD_RANK), pick) * 100
    if grp not in GROUPS:
        return np.nan
    r = OFF_BOARD_RANK if pd.isna(rank) else rank
    return blended(table.lookup, label, grp, min(counts.get(grp, 0), 2), r, pick, p_need, W) * 100


def _build_one(args):
    sit, year = args
    get_table(sit, year=year)
    return _key(sit, year)


def build_all(workers=6):
    """Build value tables for every team situation in the review (skips ones on disk)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    S = pd.read_parquet(OUT / "review_states.parquet")
    sits = {_key(situation(r), r.year): (situation(r), int(r.year)) for _, r in S.iterrows()}
    todo = sorted((a for k, a in sits.items() if not (TABLE_DIR / f"{k}.npz").exists()), key=lambda a: a[1])
    print(f"{len(sits)} situations, {len(todo)} to build", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_build_one, s) for s in todo]), 1):
            print(f"{i}/{len(todo)} {f.result()}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build_all(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
