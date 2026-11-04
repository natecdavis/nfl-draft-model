"""Draft-day engine: live decisions in Super Bowl odds, from the team's own board.

Inputs (the team's, not the model's):
  board  one row per prospect: name, pos, rank (the team's own ordering of talent),
         adp (where the market expects him to go; e.g. a consensus mock / big board)
  situation  from teams.py (real team) or a custom dict
  picks  the team's picks this draft, plus future picks

The model then supplies what it has measured:
  - a player ranked r on the team's board is treated as having the outcome
    chances of a typical r-th pick at his position (outlook.py), with real
    historical players' careers sampled at that level
  - his cost is the rookie contract of the pick where he's actually taken, so a
    player who falls to us is cheaper for the same talent
  - value -> team strength -> title odds for this team's situation (pick_value.py)
  - availability: other teams take players in an order scattered around adp; the
    spread was measured from ESPN's board vs actual picks 2012-2021 (about +-25%
    in log(pick+10) at the top of the draft, +-40% elsewhere)

Expensive simulation is done once per team-board (a few seconds); answers on
the clock are lookups plus a quick availability simulation.

Approximation: titles from several players are added up rather than simulated
jointly (fine for a few picks; slightly understates the extra value of two hits).
"""
import zlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from outlook import GROUPS
from pick_value import baseline_paths, pick_effect, sample_outcomes
from trade import Context, Pick, package_points

# board position labels -> model position groups
POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
       "OT": "T", "T": "T", "OL": "G", "OG": "G", "G": "G", "IOL": "G", "C": "C",
       "EDGE": "ED", "ED": "ED", "DE": "ED", "OLB": "ED", "DT": "IDL", "DL": "IDL", "IDL": "IDL", "NT": "IDL",
       "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S", "SAF": "S", "FS": "S", "SS": "S", "ATH": "WR"}
SPECIALISTS = {"K", "PK", "P", "LS"}  # not modeled: others can draft them, we never pick them
NEED_EARLY = {"starter": 1.0, "rotation": 0.6, "depth": 0.3}  # share of snaps years 1-2 vs normal
LEVELS = list(NEED_EARLY)


def spread(adp):
    """Spread of actual pick around expected pick, in log(pick+10) units."""
    return 0.25 + 0.15 * np.clip((np.asarray(adp, float) - 1) / 20, 0, 1)


@dataclass
class DraftEngine:
    situation: dict
    n: int = 3000
    seed: int = 0
    need: dict = field(default_factory=dict)  # grp -> "starter" | "rotation" | "depth"

    def __post_init__(self):
        self.ctx = Context()
        self.rng = np.random.default_rng(self.seed)
        self.base = baseline_paths(self.situation, self.rng, self.n)
        self.base_titles = self.ctx.sb(self.base).sum(axis=1).mean()
        self._outcomes = lru_cache(maxsize=None)(self._sample)
        self._value = lru_cache(maxsize=None)(self._player_value)

    def _sample(self, grp, talent):
        # stable seed (Python's hash() changes between runs, which made results drift)
        rng = np.random.default_rng(zlib.crc32(f"{grp}|{talent}|{self.seed}".encode()))
        return sample_outcomes(self.ctx.lib, self.ctx.curves, grp, int(np.clip(talent, 1, 262)), rng, n=self.n)

    def player_points(self, grp: str, talent: int, pick: int, level: str | None = None,
                      qb_filled: bool = False) -> np.ndarray:
        """(n, 5) points of team strength from taking a player of this talent at this pick.
        level: how much he plays early (default: the team's need at his position).
        qb_filled: we already drafted a QB this draft, so another one doesn't start."""
        shares, price = self._outcomes(grp, int(talent))
        shares = shares.copy()
        shares[:, :2] *= NEED_EARLY[level or self.need.get(grp, "starter")]
        sit = {**self.situation, "qb_from": 99 if qb_filled else self._qb_from()}
        return pick_effect(shares, price, int(pick), grp, sit, self.ctx.cost, self.ctx.bar, self.ctx.rates)

    def _qb_from(self):
        q = self.situation.get("qb_from")
        return 99 if q is None else q  # no QB need: a drafted QB doesn't play in the window

    def titles(self, pts: np.ndarray) -> float:
        """Super Bowls added over 5 seasons (fraction of one title)."""
        return self.ctx.sb(self.base + pts).sum(axis=1).mean() - self.base_titles

    def _player_value(self, grp, talent, pick, level=None, qb_filled=False) -> float:
        return self.titles(self.player_points(grp, talent, pick, level, qb_filled))

    def player_value(self, grp, talent, pick, level=None, qb_filled=False) -> float:
        return self._value(grp, int(talent), int(pick), level, qb_filled)

    def future_value(self, picks: list[Pick]) -> float:
        if not picks:
            return 0.0
        pts = package_points(self.ctx, self.situation, picks, self.base, np.random.default_rng(self.seed + 7))
        return self.titles(pts)

    # ---------- live draft ----------

    def prepare(self, board: pd.DataFrame) -> pd.DataFrame:
        b = board.copy()
        b["grp"] = b["pos"].str.upper().map(POS)
        b.loc[b["pos"].str.upper().isin(SPECIALISTS), "grp"] = "SPEC"
        if b.grp.isna().any():
            raise ValueError(f"Unknown positions: {sorted(b[b.grp.isna()].pos.unique())}")
        return b.reset_index(drop=True)

    def value_matrix(self, board: pd.DataFrame, picks: list[int]) -> np.ndarray:
        """Titles from taking each player at each of our picks, at each need level:
        (players, picks, 3). Level 0 = the team's need at that position; each
        player we've already taken at the position drops it one level (for QBs:
        after one QB, the need is filled)."""
        V = np.full((len(board), len(picks), 3), -1.0)
        for i, (g, r) in enumerate(zip(board.grp, board["rank"])):
            if g == "SPEC":
                continue
            for j, k in enumerate(picks):
                if g == "QB":
                    v0, v1 = self.player_value(g, r, k), self.player_value(g, r, k, qb_filled=True)
                    V[i, j] = [v0, v1, v1]
                else:
                    b = LEVELS.index(self.need.get(g, "starter"))
                    V[i, j] = [self.player_value(g, r, k, LEVELS[min(b + m, 2)]) for m in range(3)]
        return V

    def simulate(self, board: pd.DataFrame, picks: list[int], now: int, sims: int = 2000, seed: int = 1) -> dict:
        """Play the rest of the draft from pick `now` many times. Other teams take
        players in a random order scattered around adp; at our picks we take the
        best player by title value, given what we've already taken. Returns
        availability and expected value per pick."""
        picks = sorted(p for p in picks if p >= now)
        rng = np.random.default_rng(seed)
        V = self.value_matrix(board, picks)
        grp_codes, grp_idx = np.unique(board.grp.to_numpy(), return_inverse=True)
        adp = board.adp.to_numpy(float)
        z = rng.standard_normal((sims, len(board)))
        realized = np.exp(np.log(adp + 10) + spread(adp) * z) - 10
        order = np.argsort(realized, axis=1)
        avail_at = np.zeros((len(board), len(picks)))
        got_value = np.zeros((sims, len(picks)))
        took = np.full((sims, len(picks)), -1)
        rows = np.arange(len(board))
        for s in range(sims):
            gone = np.zeros(len(board), bool)
            count = np.zeros(len(grp_codes), int)  # players we've taken per position
            ptr, prev = 0, now
            for j, k in enumerate(picks):
                n_others = k - prev
                while n_others > 0 and ptr < len(board):
                    i = order[s, ptr]; ptr += 1
                    if not gone[i]:
                        gone[i] = True; n_others -= 1
                avail = ~gone
                avail_at[avail, j] += 1
                if avail.any():
                    v = V[rows, j, np.minimum(count[grp_idx], 2)]
                    i = np.flatnonzero(avail)[np.argmax(v[avail])]
                    gone[i] = True; took[s, j] = i; got_value[s, j] = v[i]
                    count[grp_idx[i]] += 1
                prev = k + 1
        return {"picks": picks, "V": V[:, :, 0], "p_available": avail_at / sims,
                "exp_value": got_value.mean(axis=0), "took": took}

    def on_the_clock(self, board: pd.DataFrame, picks: list[int], now: int, sims=2000) -> pd.DataFrame:
        """Rank available players for the pick we're on, with the chance each is
        still there at our next pick (if he likely is, we can wait)."""
        sim = self.simulate(board, picks, now, sims)
        out = board[["name", "pos", "rank", "adp"]].copy()
        out["titles_if_taken_now"] = sim["V"][:, 0] * 100
        if len(sim["picks"]) > 1:
            out[f"p_avail_at_{sim['picks'][1]}"] = sim["p_available"][:, 1]
        return out.sort_values("titles_if_taken_now", ascending=False)

    def draft_value(self, board, picks, now, future: list[Pick] = (), sims=2000) -> float:
        """Expected titles from a set of picks: this draft's (board-aware) + future (generic)."""
        this = sum(self.simulate(board, picks, now, sims)["exp_value"]) if picks else 0.0
        return this + self.future_value(list(future))

    def evaluate_trade(self, board, picks, now, give_now=(), get_now=(), give_future=(), get_future=(),
                       future_owned=(), sims=2000) -> dict:
        """Change in titles from a trade. picks = our current picks this draft."""
        before = self.draft_value(board, picks, now, future_owned, sims)
        after_picks = sorted(set(picks) - set(give_now) | set(get_now))
        after_future = [p for p in future_owned if p not in give_future] + list(get_future)
        after = self.draft_value(board, after_picks, now, after_future, sims)
        return {"before": before * 100, "after": after * 100, "change": (after - before) * 100}

    def trade_down_scan(self, board, picks, now, targets, sims=1500) -> pd.DataFrame:
        """For moving from `now` to each target pick: titles lost by moving down, and
        how that compares with the value of an extra pick at a few slots."""
        base = sum(self.simulate(board, picks, now, sims)["exp_value"])
        rows = []
        for t in targets:
            moved = sorted(set(picks) - {now} | {t})
            v = sum(self.simulate(board, moved, now, sims)["exp_value"])
            rows.append({"move_to": t, "titles_lost": (base - v) * 100})
        return pd.DataFrame(rows)
