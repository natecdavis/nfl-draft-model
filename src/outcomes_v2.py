"""Outcome engine v2: possible careers for a prospect of a given position and talent rank.

Replaces the star/starter/bust buckets (outlook.py + value_library sampling).

  1. Expected value in each of seasons 1-4 from consensus rank + position, trained on
     cfdarko's season-by-season rookie-value labels for all classes through 2025
     (prospect_model.py, version "consensus_pre"; player features were tested and made
     choices worse, so they're not used).
  2. Possible careers = the actual seasons 1-4 of the 60 past drafted players at the same
     position family whose expected value is closest (fewer for small pools: QB ~20) (tested: actual outcomes land below
     the 10th / above the 90th percentile 6% / 9% of the time).
  3. Units: cfdarko's values are rescaled per position family to this project's value
     measure (snap share x market price), which the cap-to-team-strength bridge used
     (scripts/units_check.py: overall within 2-16%; RB, TE, EDGE, WR, DL run higher in
     cfdarko, OL and LB lower).

Interface matches pick_value.sample_outcomes: returns (shares (n,5), price (n,)) so that
shares x price reproduces each season's value and the replacement-level adjustment in
pick_effect scales with playing time. Season 5 (round-1 option year) repeats season 4.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

import prospect_model as pm

FAMILY = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "ED": "EDGE",
          "IDL": "DL", "LB": "LB", "CB": "DB", "S": "DB"}
# cfdarko / this project, 4-season totals by family (scripts/units_check.py)
CF_OVER_OURS = {"DB": 1.09, "DL": 1.24, "EDGE": 1.28, "LB": 0.91, "OL": 0.87, "QB": 0.97, "RB": 1.64,
                "TE": 1.50, "WR": 1.25}
# "single" (default since Nov 2026): one overall factor, keeping cfdarko's within-position pricing (a role
# is priced at what veterans at that position earn for that snap share and grade). "family": one factor
# per position family (above), matching this project's snap share x own-pay measure, which discounts
# rotational positions (RB, TE, EDGE, DL) and favors OL. Override with the environment variable
# OUTCOME_UNITS. Choice test: scripts/units_choice.py, scripts/plan_choice.py (README).
CF_OVER_OURS_ALL = 1.10  # 4-season totals, all positions (scripts/units_check.py)
UNITS = os.environ.get("OUTCOME_UNITS", "single")


def unit_factor(family: str, units: str | None = None) -> float:
    return CF_OVER_OURS_ALL if (units or UNITS) == "single" else CF_OVER_OURS.get(family, 1.0)


# cfdarko prices a season on one veteran pay curve per family, but within OL and DB the market pays
# positions differently: veteran starters (>= 60% of snaps, 2016-2025, contract in force) earn, relative
# to their family, OT 1.27 (LT 6.7% / RT 4.6% of cap), OG 0.86, C 0.77, CB 1.11, S 0.89 (factors average
# ~1 over starters). A real season (realized value) is scaled by the player's pay factor.
# Expectations also need how each position turns out at the same consensus rank, since careers are
# shared across the family: with pooled pricing, drafted board players 2014-2022 delivered vs the
# model's expectation OT 0.90, OG 1.07, C 1.35, CB 0.85, S 1.17 (interior OL and safeties start and
# grade well sooner). Net expectation factor = pay x turnout: OT 1.14, OG 0.92, C 1.04, CB 0.94, S 1.04
# (each within +-0.06 leaving out any one class). Environment variable POS_PAY: "off"; "pay" (pay
# factor in expectations too, overcorrects: tested); "on" (pay in realized, pay x turnout in expectations).
POS_PAY_FACTOR = {"T": 1.27, "G": 0.86, "C": 0.77, "CB": 1.11, "S": 0.89}
POS_TURNOUT = {"T": 0.90, "G": 1.07, "C": 1.35, "CB": 0.85, "S": 1.17}
POS_PAY = os.environ.get("POS_PAY", "off")


# POS_PAY="market" (test only): scale each position's pay-curve value so the average absent starter's
# credit matches the betting market's starter-minus-backup estimate (scripts/absence_curve.py factors,
# README "Betting-line absence test"). Applied to realized and expected seasons alike.
MARKET_FACTOR = {"QB": 0.89, "RB": 0.51, "WR": 0.95, "TE": 1.06, "T": 0.94, "G": 0.94, "C": 0.94,
                 "IDL": 0.65, "ED": 0.42, "LB": 1.21, "CB": 0.82, "S": 0.65}
_SHRINK = float(os.environ.get("MARKET_SHRINK") or 0.0)  # 0.5 = halfway to 1 (test)
MARKET_FACTOR = {k: v + _SHRINK * (1 - v) for k, v in MARKET_FACTOR.items()}


def pay_factor(grp: str, on: bool | None = None) -> float:
    """For realized seasons: the position's pay relative to its family."""
    use = (POS_PAY in ("on", "pay", "market")) if on is None else on
    if not use:
        return 1.0
    return (MARKET_FACTOR if POS_PAY == "market" else POS_PAY_FACTOR).get(grp, 1.0)


def expect_factor(grp: str) -> float:
    """For expected seasons (outcome sampling): pay, and with "on" also the position's turnout."""
    if POS_PAY == "on":
        return POS_PAY_FACTOR.get(grp, 1.0) * POS_TURNOUT.get(grp, 1.0)
    return pay_factor(grp)


N_COMPS = 60
# Comparison-career correction (scripts/misses.py, scripts/comp_correct.py). The comparison set is the k
# nearest drafted players by expected value; at the top of a family every neighbour is below the player,
# so the sampled careers ran ~0.71x what consensus top-10 non-QBs actually did (every family 0.60-0.85x),
# while matching elsewhere. "ll": rescale the sampled careers so their weighted mean equals a local-linear
# fit (weighted regression of comparison 4-season value on log expected value, read at the player's own
# expectation): the standard fix for kernel boundary bias; shape of the distribution kept. "off": before Nov 2026.
# Adopted Nov 2026 (held-out calibration better, choices tie or improve on every yardstick: README). Environment
# variable COMP_CORRECT overrides it.
COMP_CORRECT = os.environ.get("COMP_CORRECT", "ll")
SCALE_CLIP = (0.5, 2.0)
# Draft classes without a consensus board (2012-2013: Hasan's board starts in 2014). Every player in them
# reads as off-board, first-rounders included, so they made up ~2/3 of the off-board comparison pool
# (off-board QBs drew Luck, Wilson, Cousins) and drafted off-board players were valued ~1.8x what they
# delivered (QBs ~10x). "drop" (adopted Nov 2026): leave those classes out of the regression and the
# comparison pool. "keep": before Nov 2026. Environment variable NO_BOARD_CLASSES overrides it.
NO_BOARD_CLASSES = os.environ.get("NO_BOARD_CLASSES", "drop")
# Monotone value in consensus rank. Small pools leave bumps late in the board: QBs ranked 250-297 drew
# ~0.08 of cap over 4 seasons (vs 0.024 at rank 200), from one comparison (Brock Purdy, rank 290, 0.44),
# and leave-one-draft-out reviews recommended QBs ranked ~270-300 in rounds 2-4. Per family, the mean of
# the sampled careers (after COMP_CORRECT) over board ranks 1-329 is made non-increasing in rank, shape of
# each distribution kept. "min" (adopted Nov 2026): each rank capped at the lowest value of any better rank
# (only lowers values). "iso": isotonic fit (violating stretches pooled; lifted QBs ranked 150-250 and
# their recommendations delivered ~0: tested, not adopted). "off": none. Off-board players keep their own
# value (drafted off-board players did better than board players ranked 251-300; pooling them put them
# at 0.78x actual). Test: scripts/no_board.py (README). Environment variable MONOTONE overrides it.
MONOTONE = os.environ.get("MONOTONE", "min")
MIN_COMPS, COMP_SHARE = 15, 1 / 6  # small position pools (QB): use the nearest sixth, at least 15
REG = dict(min_child_samples=150, n_estimators=300, colsample_bytree=0.5, num_leaves=7, reg_lambda=5.0)

# Backtests only (scripts/holdout_review.py): None = all classes (the live model), ("lodo", Y) = leave
# out draft class Y, ("expand", Y) = only what was known on draft day Y (classes before Y, seasons
# through Y-1). Set it, then reset _MODEL = None, before valuing draft Y.
HOLDOUT = None


def apply_holdout(df: pd.DataFrame, holdout) -> pd.DataFrame:
    if holdout is None:
        return df
    mode, y = holdout
    if mode == "lodo":
        return df[df.cls != y].copy()
    if mode == "expand":
        df = df[df.cls <= y - 1].copy()
        for t in pm.SEASONS:
            df.loc[df.cls + t - 1 > y - 1, f"y{t}"] = np.nan
        return df
    raise ValueError(holdout)


class OutcomeModel:
    def __init__(self):
        df = apply_holdout(pm.load(), HOLDOUT)
        if NO_BOARD_CLASSES == "drop":
            ranked = df.hasan_rank.notna().where(df.drafted).groupby(df.cls).mean()  # share of drafted on board
            df = df[df.cls.map(ranked) > 0.5]
        last = pm.LAST_SEASON_PLAYED if HOLDOUT is None or HOLDOUT[0] != "expand" else HOLDOUT[1] - 1
        for t in pm.SEASONS:
            df[f"y{t}"] = df[f"y{t}"] / df.grp.map(unit_factor)
        self.cats = df.grp.astype("category").cat.categories
        saved = dict(pm.PARAMS)
        pm.PARAMS.update(REG)
        try:
            self.models = [pm.fit(df, "consensus_pre", t) for t in pm.SEASONS]
        finally:
            pm.PARAMS.clear(); pm.PARAMS.update(saved)
        comps = df[df.drafted & (df.cls + 3 <= last)].copy()
        comps["p14"] = self._predict(comps)
        self.comps = {g: (np.log(d.p14.to_numpy() + 1e-3), d[[f"y{t}" for t in pm.SEASONS]].to_numpy())
                      for g, d in comps.groupby("grp")}
        self._near = lru_cache(maxsize=None)(self._nearest)
        self._scale = lru_cache(maxsize=None)(self._ll_scale)
        self._mono = {}

    def _predict(self, d: pd.DataFrame) -> np.ndarray:
        return sum(m.predict(d[f]) for m, f in self.models)

    def expected(self, family: str, rank: float) -> np.ndarray:
        """Expected value (fraction of cap, this project's units) in seasons 1-4."""
        on = rank < pm.OFF_BOARD
        row = pd.DataFrame({"log_rank": [np.log(min(rank, pm.OFF_BOARD) + 10)], "on_board": [int(on)],
                            "grp_code": [list(self.cats).index(family)]})
        return np.array([m.predict(row[f])[0] for m, f in self.models])

    def _nearest(self, family: str, rank: int):
        """Comparison careers and their weights: Gaussian in distance of expected value,
        width set by the k-th nearest (k = 60, or a sixth of a small pool, at least 15),
        so the set shifts smoothly as rank changes instead of jumping."""
        lp = np.log(self.expected(family, rank).sum() + 1e-3)
        keys, paths = self.comps[family]
        k = int(np.clip(len(keys) * COMP_SHARE, MIN_COMPS, N_COMPS))
        d = np.abs(keys - lp)
        k = min(k, len(d))
        width = max(np.sort(d)[k - 1] / 1.5, 1e-3)
        w = np.exp(-0.5 * (d / width) ** 2)
        return paths, w / w.sum()

    def _ll_scale(self, family: str, rank: int) -> float:
        """Local-linear comparison mean / plain weighted comparison mean (1.0 when COMP_CORRECT is off)."""
        lp = np.log(self.expected(family, rank).sum() + 1e-3)
        keys, _ = self.comps[family]
        paths, w = self._near(family, rank)
        y, x = paths.sum(1), keys - lp
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(2), X.T @ (w * y))
        plain = float((y * w).sum())
        return float(np.clip(max(beta[0], 1e-6) / max(plain, 1e-9), *SCALE_CLIP))

    def _ll(self, family: str, r: int) -> float:
        return self._scale(family, r) if COMP_CORRECT == "ll" else 1.0

    def _mono_factor(self, family: str) -> np.ndarray:
        """Non-increasing-in-rank fit / raw mean of the sampled careers, ranks 1..OFF_BOARD (off-board: 1)."""
        if family not in self._mono:
            from sklearn.isotonic import IsotonicRegression
            r = np.arange(1, pm.OFF_BOARD)  # board ranks; off-board keeps its own value (factor 1)
            raw = np.array([float((lambda pw: (pw[0].sum(1) * pw[1]).sum())(self._near(family, int(k))))
                            * self._ll(family, int(k)) for k in r])
            iso = (IsotonicRegression(increasing=False).fit_transform(r, raw) if MONOTONE == "iso"
                   else np.minimum.accumulate(raw))  # "min": capped at the lowest value of any better rank
            self._mono[family] = np.append(np.where(raw > 1e-9, iso / np.maximum(raw, 1e-9), 1.0), 1.0)
        return self._mono[family]

    def scale(self, family: str, rank: int) -> float:
        """Factor on the sampled careers: local-linear correction x monotone fit (1.0 with both off)."""
        r = int(np.clip(rank, 1, pm.OFF_BOARD))
        mono = self._mono_factor(family)[r - 1] if MONOTONE in ("iso", "min") else 1.0
        return self._ll(family, r) * float(mono)

    def sample(self, grp: str, rank: int, rng, n: int):
        """(shares (n,5), price (n,)) for pick_value.pick_effect."""
        fam = FAMILY.get(grp, grp)
        paths, w = self._near(fam, int(np.clip(rank, 1, pm.OFF_BOARD)))
        y = paths[rng.choice(len(paths), size=n, p=w)]
        y5 = np.column_stack([y, y[:, -1]])  # option year ~ season 4
        price = np.maximum(y5.max(axis=1), 1e-6)
        return y5 / price[:, None], price * expect_factor(grp) * self.scale(fam, rank)


_MODEL = None


def model() -> OutcomeModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = OutcomeModel()
    return _MODEL
