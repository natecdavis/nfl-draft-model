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
    scatter and drift were measured from Arif Hasan's consensus boards vs actual
    picks, 2014-2024 (wide at the top: some top-10 prospects fall far)

Expensive simulation is done once per team-board (a few seconds); answers on
the clock are lookups plus a quick availability simulation.

Approximation: titles from several players are added up rather than simulated
jointly (fine for a few picks; slightly understates the extra value of two hits).
"""
import os
import zlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

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
# Rookies we already took at the position in this draft. "measured" (default since Nov 2026): the 2nd / 3rd+
# rookie at a group keeps x0.95 / x0.88 of his playing time in every season, as measured on 2014-22 rookies
# (scripts/roster_need.py regression with roster need, value years 1-4: -4.8%, 95% -14% to +5%; -12%, -27% to
# +3%; README 'Roster need test'). "early" (the old rule): each earlier rookie drops the need one level
# (NEED_EARLY, years 1-2 only), i.e. x0.6 then x0.3, several times the measured effect.
NEED_STEP = os.environ.get("NEED_STEP", "measured")
WITHIN_DRAFT = (1.0, 0.95, 0.88)


# How far actual picks land from the consensus rank, in log(pick+10) units, measured on
# Arif Hasan's consensus boards 2014-2024 (undrafted counted as pick 300). Top prospects
# scatter the most: some fall far (medicals, character, scheme).
_RANK_PTS = [5, 21, 48, 82, 125, 175, 260]
_SD = [0.67, 0.48, 0.46, 0.45, 0.41, 0.38, 0.29]
_DRIFT = [0.27, 0.13, 0.09, 0.11, 0.15, 0.07, -0.03]


def spread(adp):
    """Spread of actual pick around the consensus rank, in log(pick+10) units."""
    return np.interp(np.asarray(adp, float), _RANK_PTS, _SD)


def drift(adp):
    """Average offset: players tend to go a bit later than their consensus rank."""
    return np.interp(np.asarray(adp, float), _RANK_PTS, _DRIFT)


@dataclass
class DraftEngine:
    situation: dict
    n: int = 3000
    seed: int = 0
    need: dict = field(default_factory=dict)  # grp -> "starter" | "rotation" | "depth"
    # When a player falls past his board rank, other teams passing on him is information.
    # Players who fell (Hasan consensus 2014-21) turned out ~80-89% like the pick where they
    # went, not their board rank, whether or not undrafted fallers are included
    # (scripts/fall_weight.py, fall_weight_all.py; ~/cfdarko found the same).
    # 0 = trust the board fully, 1 = trust the draft slot fully.
    fall_weight: float = 0.85
    ctx: Context | None = None  # share one loaded Context across many engines

    def __post_init__(self):
        self.ctx = self.ctx or Context()
        self.rng = np.random.default_rng(self.seed)
        self.base = baseline_paths(self.situation, self.rng, self.n)
        self.base_titles = self.ctx.sb(self.base).sum(axis=1).mean()
        self._outcomes = lru_cache(maxsize=None)(self._sample)
        self._value = lru_cache(maxsize=None)(self._player_value)

    def _sample(self, grp, talent):
        # talent runs to prospect_model.OFF_BOARD (330 = off the board); the engine caps it there. It was
        # clipped to 262 here until Nov 17 2026, which valued off-board players as board rank 262.
        # stable seed (Python's hash() changes between runs, which made results drift)
        rng = np.random.default_rng(zlib.crc32(f"{grp}|{talent}|{self.seed}".encode()))
        return sample_outcomes(self.ctx.lib, self.ctx.curves, grp, max(1, int(talent)), rng, n=self.n)

    def player_points(self, grp: str, talent: int, pick: int, level: str | None = None,
                      qb_filled: bool = False, nth: int = 0) -> np.ndarray:
        """(n, 5) points of team strength from taking a player of this talent at this pick.
        level: how much he plays early (default: the team's need at his position).
        qb_filled: we already drafted a QB this draft, so another one doesn't start.
        nth: rookies already taken at his group this draft (NEED_STEP "measured" only; with "early"
        callers lower `level` instead)."""
        shares, price = self._outcomes(grp, int(talent))
        shares = shares.copy()
        shares[:, :2] *= NEED_EARLY[level or self.need.get(grp, "starter")]
        if nth:
            shares *= WITHIN_DRAFT[min(int(nth), 2)]
        sit = {**self.situation, "qb_from": 99 if qb_filled else self._qb_from()}
        return pick_effect(shares, price, int(pick), grp, sit, self.ctx.cost, self.ctx.bar, self.ctx.rates)

    def _qb_from(self):
        q = self.situation.get("qb_from")
        return 99 if q is None else q  # no QB need: a drafted QB doesn't play in the window

    def titles(self, pts: np.ndarray) -> float:
        """Super Bowls added over 5 seasons (fraction of one title)."""
        return self.ctx.sb(self.base + pts).sum(axis=1).mean() - self.base_titles

    def effective_talent(self, talent, pick) -> int:
        """Board rank, shaded toward the pick if he's still there past his rank."""
        if pick <= talent:
            return int(talent)
        lt, lp = np.log(talent + 10), np.log(pick + 10)
        return int(round(np.exp(lt + self.fall_weight * (lp - lt)) - 10))

    def _player_value(self, grp, talent, pick, level=None, qb_filled=False, nth=0) -> float:
        return self.titles(self.player_points(grp, self.effective_talent(talent, pick), pick, level, qb_filled, nth))

    def player_value(self, grp, talent, pick, level=None, qb_filled=False, nth=0) -> float:
        return self._value(grp, int(talent), int(pick), level, qb_filled, int(nth))

    def step_value(self, grp, talent, pick, m) -> float:
        """Value after we've taken m players at his group this draft (non-QB), per NEED_STEP."""
        if NEED_STEP == "measured":
            return self.player_value(grp, talent, pick, nth=m)
        b = LEVELS.index(self.need.get(grp, "starter"))
        return self.player_value(grp, talent, pick, LEVELS[min(b + m, 2)])

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
        (players, picks, 3). Index m = we've already taken m players at that position
        this draft (NEED_STEP; for QBs: after one QB, the need is filled)."""
        V = np.full((len(board), len(picks), 3), -1.0)
        for i, (g, r) in enumerate(zip(board.grp, board["rank"])):
            if g == "SPEC":
                continue
            for j, k in enumerate(picks):
                if g == "QB":
                    v0, v1 = self.player_value(g, r, k), self.player_value(g, r, k, qb_filled=True)
                    V[i, j] = [v0, v1, v1]
                else:
                    V[i, j] = [self.step_value(g, r, k, m) for m in range(3)]
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
        # Everyone on the live board is still undrafted at `now`, so draw each player's draft spot
        # given that it's `now` or later. Without this, a faller (consensus #5 still there at #20) is
        # drawn mostly before #20 and treated as gone at once; in fact fallers keep falling. Rolling
        # test 2017-2026 (scripts/availability_test.py): log loss 0.552 -> 0.536, better in 9/10 drafts.
        mu, sd = np.log(adp + 10) + drift(adp), spread(adp)
        lo = ndtr((np.log(now + 10) - mu) / sd)
        z = ndtri(np.clip(lo + (1 - lo) * rng.random((sims, len(board))), 1e-12, 1 - 1e-12))
        realized = np.exp(mu + sd * z) - 10
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

    # ---------- looking ahead to our later picks ----------

    def _draws(self, board, now, sims, seed):
        """(sims, players) order keys for other teams' picks: lower = taken sooner, each player's
        spot drawn given he's still on the board at `now` (same rule as simulate)."""
        adp = board.adp.to_numpy(float)
        mu, sd = np.log(adp + 10) + drift(adp), spread(adp)
        lo = ndtr((np.log(now + 10) - mu) / sd)
        rng = np.random.default_rng(seed)
        return mu + sd * ndtri(np.clip(lo + (1 - lo) * rng.random((sims, len(board))), 1e-12, 1 - 1e-12))

    @staticmethod
    def _others_take(avail, key, n):
        """Other teams take n players: the available ones with the lowest keys (in place)."""
        if n <= 0:
            return
        k = np.where(avail, key, np.inf)
        n = min(n, avail.shape[1] - 1)
        np.put_along_axis(avail, np.argpartition(k, n, axis=1)[:, :n], False, axis=1)

    def lookahead(self, board: pd.DataFrame, picks: list[int], now: int, top: int = 12, sims: int = 600,
                  seed: int = 1) -> pd.DataFrame:
        """For each serious candidate at our next pick: titles from him + expected titles from our
        later picks this draft if we take him (later picks: best available for the roster), over
        simulated drafts. The same simulated drafts are used for every candidate, so differences
        between candidates aren't simulation noise. Candidates: our top `top` by value now, plus
        the best at each position."""
        picks = sorted(p for p in picks if p >= now)
        if not picks:
            return pd.DataFrame()
        V = self.value_matrix(board, picks)
        g = board.grp.to_numpy()
        codes, gidx = np.unique(g, return_inverse=True)
        v0 = np.where(g == "SPEC", -np.inf, V[:, 0, 0])
        cand = list(np.argsort(-v0)[:top])
        for c in range(len(codes)):
            m = np.where(gidx == c, v0, -np.inf)
            if np.isfinite(m.max()) and codes[c] != "SPEC":
                cand.append(int(np.argmax(m)))
        cand = sorted(set(int(c) for c in cand))
        C, N = len(cand), len(board)
        key = np.tile(self._draws(board, now, sims, seed), (C, 1))
        avail = np.ones((C * sims, N), bool)
        count = np.zeros((C * sims, len(codes)), int)
        forced = np.repeat(np.array(cand), sims)
        rows = np.arange(C * sims)
        got = np.zeros((C * sims, len(picks)))
        had = np.zeros(C * sims, bool)
        prev = now
        for j, k in enumerate(picks):
            self._others_take(avail, key, k - prev)
            vals = np.where(avail, V[np.arange(N)[None, :], j, np.minimum(count[:, gidx], 2)], -np.inf)
            i = np.argmax(vals, axis=1)
            if j == 0:  # take the candidate if he's still there, else the best available
                had = avail[rows, forced]
                i = np.where(had, forced, i)
            v = vals[rows, i]
            ok = np.isfinite(v)
            got[ok, j] = v[ok]
            avail[rows[ok], i[ok]] = False
            np.add.at(count, (rows[ok], gidx[i[ok]]), 1)
            prev = k + 1
        got = got.reshape(C, sims, len(picks))
        out = board.iloc[cand][["name", "pos", "rank", "adp"]].copy()
        out["titles_now"] = V[cand, 0, 0] * 100
        out["titles_later_picks"] = got[:, :, 1:].sum(axis=2).mean(axis=1) * 100
        out["titles_total"] = got.sum(axis=2).mean(axis=1) * 100
        out["p_there_at_our_pick"] = had.reshape(C, sims).mean(axis=1)
        out["vs_best_now"] = out.titles_total - out.titles_total.loc[out.titles_now.idxmax()]
        return out.sort_values("titles_total", ascending=False).reset_index(drop=True)

    def position_dropoff(self, board: pd.DataFrame, picks: list[int], now: int, sims: int = 2000,
                         seed: int = 1) -> pd.DataFrame:
        """For each position: the best value available at our next pick vs the expected best at
        that position at the pick after, if we don't take one now. A big drop = a cliff."""
        picks = sorted(p for p in picks if p >= now)
        if len(picks) < 2:
            return pd.DataFrame()
        V = self.value_matrix(board, picks[:2])
        g = board.grp.to_numpy()
        key = self._draws(board, now, sims, seed)
        avail = np.ones((sims, len(board)), bool)
        self._others_take(avail, key, picks[0] - now)
        best = {}
        for j in (0, 1):
            if j == 1:  # we take nobody at this position now; others pick in between
                self._others_take(avail, key, picks[1] - picks[0] - 1)
            for grp in sorted(set(g) - {"SPEC"}):
                m = (g == grp)
                vals = np.where(avail[:, m], V[m, j, 0][None, :], -np.inf).max(axis=1)
                best[(grp, j)] = np.where(np.isfinite(vals), vals, 0.0)
        rows = []
        for grp in sorted(set(g) - {"SPEC"}):
            a, b = best[(grp, 0)], best[(grp, 1)]
            rows.append({"position": grp, f"best at #{picks[0]}": a.mean() * 100,
                         f"expected best at #{picks[1]}": b.mean() * 100, "drop_off": (a - b).mean() * 100,
                         "chance of a big drop (>half)": float(((a - b) > 0.5 * np.abs(a)).mean())})
        return pd.DataFrame(rows).sort_values("drop_off", ascending=False).reset_index(drop=True)

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
        """For moving our next pick (the first of `picks` at or after `now`) down to each target
        pick: titles lost by moving down, and how that compares with the value of an extra pick
        at a few slots. Targets we already hold are skipped (that would be giving the pick away)."""
        base = sum(self.simulate(board, picks, now, sims)["exp_value"])
        frm = min(p for p in picks if p >= now)
        rows = []
        for t in targets:
            if t in picks:
                continue
            moved = sorted(set(picks) - {frm} | {t})
            v = sum(self.simulate(board, moved, now, sims)["exp_value"])
            rows.append({"move_to": t, "titles_lost": (base - v) * 100})
        return pd.DataFrame(rows)


# ---------- QB reach flag (scripts/qb_reach_flag.py, README 'QB valuation') ----------

def qb_reach_note(res: pd.DataFrame, pick: int, next_pick: int | None = None) -> tuple[str, str] | None:
    """When the best-value player is a QB the past says to be wary of, a (level, message) for the app:
    "warning" for a QB ranked 33+ on the consensus board (the model's QB choices there delivered well under
    what it predicted, 2014-22), "info" for a QB ranked 1-32 sitting 16+ spots below the pick. None otherwise.
    `res` is on_the_clock's output (best first)."""
    import json
    from outlook import OUT
    if res.empty or res.iloc[0].pos != "QB":
        return None
    path = OUT / "qb_reach.json"
    if not path.exists():
        return None
    look = json.load(open(path))
    best, rank = res.iloc[0], float(res.iloc[0]["rank"])
    alt = res[res.pos != "QB"].head(1)
    col = f"p_avail_at_{next_pick}" if next_pick else None
    there = (f" He's still there at our next pick (#{next_pick}) in {best[col]:.0%} of simulated drafts."
             if col and col in res.columns else "")
    other = (f" Best non-QB: {alt.iloc[0]['name']} ({alt.iloc[0].pos}), +{alt.iloc[0].titles_if_taken_now:.2f}."
             if len(alt) else "")
    b = next(r for r in look["rank"] if r["lo"] <= rank <= r["hi"])
    past = (f"In past drafts (2014-22) the model's picks of QBs {b['bucket']} delivered {b['ratio']:.2f}x the titles it "
            f"predicted (95% range {b['ratio_lo']:.2f}-{b['ratio_hi']:.2f}); {round(b['starters'] * b['n'])} of {b['n']} "
            f"became starters (3+ seasons).")
    if rank > 32:
        return "warning", (f"QB caution: {best['name']} is #{rank:.0f} on the consensus board. {past} The model's value "
                           f"for QBs ranked this low is likely too high (too few past QBs to pin down how often they "
                           f"become starters).{there}{other}")
    gap = rank - pick
    if gap >= 16:
        d = next(r for r in look["dist"] if r["lo"] <= gap <= r["hi"])
        return "info", (f"QB reach: {best['name']} is #{rank:.0f} on the board, {gap:.0f} spots below this pick. The model's "
                        f"QB choices this far below the pick delivered {d['ratio']:.2f}x predicted (95% "
                        f"{d['ratio_lo']:.2f}-{d['ratio_hi']:.2f}, n {d['n']}). Teams know more about QBs than the "
                        f"boards do: where a QB is expected to go says more than his board rank.{there}{other}")
    return None
