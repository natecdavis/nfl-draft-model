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


N_COMPS = 60
MIN_COMPS, COMP_SHARE = 15, 1 / 6  # small position pools (QB): use the nearest sixth, at least 15
REG = dict(min_child_samples=150, n_estimators=300, colsample_bytree=0.5, num_leaves=7, reg_lambda=5.0)


class OutcomeModel:
    def __init__(self):
        df = pm.load()
        for t in pm.SEASONS:
            df[f"y{t}"] = df[f"y{t}"] / df.grp.map(unit_factor)
        self.cats = df.grp.astype("category").cat.categories
        saved = dict(pm.PARAMS)
        pm.PARAMS.update(REG)
        try:
            self.models = [pm.fit(df, "consensus_pre", t) for t in pm.SEASONS]
        finally:
            pm.PARAMS.clear(); pm.PARAMS.update(saved)
        comps = df[df.drafted & (df.cls + 3 <= pm.LAST_SEASON_PLAYED)].copy()
        comps["p14"] = self._predict(comps)
        self.comps = {g: (np.log(d.p14.to_numpy() + 1e-3), d[[f"y{t}" for t in pm.SEASONS]].to_numpy())
                      for g, d in comps.groupby("grp")}
        self._near = lru_cache(maxsize=None)(self._nearest)

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

    def sample(self, grp: str, rank: int, rng, n: int):
        """(shares (n,5), price (n,)) for pick_value.pick_effect."""
        fam = FAMILY.get(grp, grp)
        paths, w = self._near(fam, int(np.clip(rank, 1, pm.OFF_BOARD)))
        y = paths[rng.choice(len(paths), size=n, p=w)]
        y5 = np.column_stack([y, y[:, -1]])  # option year ~ season 4
        price = np.maximum(y5.max(axis=1), 1e-6)
        return y5 / price[:, None], price


_MODEL = None


def model() -> OutcomeModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = OutcomeModel()
    return _MODEL
