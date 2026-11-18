"""Kickers and punters in the simulator (scripts/specialist_value.py, README 'Kickers and punters').

Specialists aren't in the outcome label and are rarely on the consensus board, so they get their own value:
  value of drafting one = the per-season gain (team points per game, years 1-4) of a drafted specialist over an
  undrafted one, by the slot where he was actually drafted (the market's pre-draft read; the board almost never
  ranks them), minus the rookie contract's cost over the minimum (as for every pick, pick_value.pick_effect).
  Undrafted specialists add nothing: they could be signed after the draft without a pick.
  Only the first one at each position counts (a second kicker or punter adds nothing).
  Long snappers aren't valued (no measure of snapping).
Realized ("delivered"): the player's points over replacement per season minus the undrafted average, minus cost.

Tables: one small (2 positions x 3 levels x T x K) array per team situation, cached next to the main simulator
tables as spec_<key>.npz, on the same grid (the "talent" axis = draft slot; off board = undrafted).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "data/derived/specialist_engine.json"
ENTRANTS = ROOT / "data/derived/specialist_entrants.parquet"
SPEC_GROUPS = ["K", "P"]
POS_TO_GROUP = {"K": "K", "PK": "K", "P": "P"}  # long snappers (LS) stay "SPEC": not valued


def is_spec(g) -> bool:
    return g in SPEC_GROUPS


def signal(drafted_at, rank, off_board):
    """The specialist's quality signal on the table's talent axis: where he was drafted (else board rank)."""
    if pd.notna(drafted_at):
        return float(drafted_at)
    return off_board if pd.isna(rank) else float(rank)


def _params():
    return json.loads(PARAMS.read_text())


def gain_per_season(g, slot, P=None) -> float:
    """Points per game per season (years 1-4) over an undrafted specialist, for one drafted at `slot`."""
    P = P or _params()
    if slot >= 263:  # undrafted
        return 0.0
    gains = P[g]["gain"]
    return gains[0] if slot <= P["slot_edges"][0] else gains[1]


def _cost_pts(pick, ctx):
    """Years 1-4 cost of the pick's rookie contract over an undrafted specialist's (the last pick's: the rookie
    minimum), in team points at pick_value's free-agency rate. The alternative is an undrafted signing, not a
    veteran at the minimum, so late picks (rookie minimum from about #110) cost nothing extra."""
    from pick_value import COST_RATE
    return COST_RATE * (ctx.cost[int(pick)] - ctx.cost[len(ctx.cost) - 1]) * 100


def tables(sit, ctx, T_GRID, K_GRID, n=3000):
    """{"K": (3, T, K), "P": (3, T, K)} titles (fraction of one) for this team situation."""
    from pick_value import baseline_paths
    base = baseline_paths(sit, np.random.default_rng(0), n)
    b0 = ctx.sb(base).sum(axis=1).mean()
    P = _params()
    out = {}
    for g in SPEC_GROUPS:
        arr = np.zeros((3, len(T_GRID), len(K_GRID)))
        for b, k in enumerate(K_GRID):
            c = _cost_pts(k, ctx)
            for a, t in enumerate(T_GRID):
                pts = np.zeros(5)
                pts[:4] = gain_per_season(g, t, P) - c
                arr[0, a, b] = ctx.sb(base + pts).sum(axis=1).mean() - b0
        out[g] = arr  # levels 1-2 (a second one at the position) stay 0
    return out


def table_file(table_dir, key):
    return Path(table_dir) / f"spec_{key}.npz"


def get_tables(table_dir, key, sit, ctx_factory, T_GRID, K_GRID):
    f = table_file(table_dir, key)
    stamp = PARAMS.stat().st_mtime
    if f.exists() and f.stat().st_mtime > stamp:
        z = np.load(f)
        return {g: z[g] for g in z.files}
    t = tables(sit, ctx_factory(), T_GRID, K_GRID)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **t)
    tmp.replace(f)
    return t


class Realized:
    """Delivered value for specialists: per season (points over replacement - undrafted average) / 17, years 1-4."""
    def __init__(self):
        from review import _norm
        self.P = _params()
        E = pd.read_parquet(ENTRANTS)
        self.by_pick = {(int(c), int(p)): r for c, p, r in zip(E.cls, E.pick, E.index) if pd.notna(p)}
        self.by_name = {(int(c), _norm(n)): r for c, n, r, d in zip(E.cls, E.name, E.index, E.drafted) if not d}
        self.E, self._norm = E, _norm

    def per_season(self, g, year, drafted_at, name, n_played):
        """(4,) team points per game added per season vs an undrafted specialist; seasons not played = 0."""
        i = self.by_pick.get((int(year), int(drafted_at))) if pd.notna(drafted_at) else \
            self.by_name.get((int(year), self._norm(name or "")))
        und = self.P[g]["undrafted_per_season"] * 4 * 17 / 4  # undrafted mean points over replacement per season
        y = np.zeros(4)
        if i is not None:
            r = self.E.loc[i]
            y = np.array([np.nan_to_num(r[f"por_y{t}"]) for t in range(1, 5)])
        out = (y - und) / 17
        out[n_played:] = 0.0
        return out


def realized_titles(R, g, sit_base, ctx, year, pick, drafted_at, name, n_played):
    """Titles (% of one) delivered by a specialist taken at `pick`."""
    b, b0 = sit_base
    pts = np.zeros(5)
    pts[:4] = R.per_season(g, year, drafted_at, name, n_played) - _cost_pts(pick, ctx)
    pts[n_played:] = 0.0
    return (ctx.sb(b + pts).sum(axis=1).mean() - b0) * 100
