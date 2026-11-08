"""Robustness checks and fresh 2023-25 replication of the leads from scripts/lit_bias_test.py.

Discovery (2016-22 vs slot, 2018-22 vs consensus): recruiting composite x0.91 per sd vs consensus (p 0.01; vs slot
x0.98). Weaker leads: one-year wonder x0.78 vs slot (p 0.04), Senior Bowl x1.12 vs consensus (p 0.06), non-Power-5
x1.12 vs consensus (p 0.12). None survived the six-test Bonferroni bar.

Part A (discovery robustness for recruiting, vs consensus): class halves, + team SP+ / age / non-P5 controls,
offense vs defense, without five-stars.
Part B (fresh replication, specified before 2023-25 outcomes were looked at): same features and definitions, value
over the seasons played so far (2023: 3, 2024: 2, 2025: 1) against out-of-sample expectations for those seasons.
  primary    recruiting composite vs consensus, predict < 1 (one-sided p)
  secondary  one-year wonder vs slot (< 1), Senior Bowl vs consensus (> 1), non-Power-5 vs consensus (> 1)
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from lit_bias_test import OFF, load
from outcomes_v2 import REG
from outlook import OUT

NEW = (2023, 2024, 2025)


def fit(s, e, x, covs=(), fe=True):
    s = s.dropna(subset=[e, x, *covs]).copy()
    s[x] = s[x].astype(float)
    parts = [s[[x, *covs]].astype(float)]
    if fe and s.cls.nunique() > 1:
        parts.append(pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float))
    m = sm.GLM(s.actual, sm.add_constant(pd.concat(parts, axis=1)), family=sm.families.Poisson(),
               offset=np.log(s[e].clip(lower=1e-4))).fit(cov_type="HC1")
    return m, len(s), (int(s[x].sum()) if s[x].isin([0, 1]).all() else None)


def show(label, m, n, k, x, side=None):
    lo, hi = np.exp(m.conf_int().loc[x])
    z = m.params[x] / m.bse[x]
    p = (stats.norm.cdf(z) if side == "<" else 1 - stats.norm.cdf(z)) if side else m.pvalues[x]
    ptxt = f"one-sided p {p:.3f}" if side else f"p {p:.2f}"
    print(f"    {label:44s} n={n:4d}{f' ({k} flagged)' if k is not None else ''}  x{np.exp(m.params[x]):.2f} "
          f"(95% {lo:.2f}-{hi:.2f}, {ptxt})")


def main():
    uni, u = load()
    d = pd.read_parquet(OUT / "lit_bias.parquet")
    bd = d[(d.on_board == 1) & d.exp_cons.notna()].copy()
    bd["off"] = bd.grp.isin(OFF)
    print("PART A: recruiting composite vs consensus, discovery classes 2018-22")
    show("all", *fit(bd, "exp_cons", "rec_z"), "rec_z")
    show("2018-20", *fit(bd[bd.cls <= 2020], "exp_cons", "rec_z"), "rec_z")
    show("2021-22", *fit(bd[bd.cls >= 2021], "exp_cons", "rec_z"), "rec_z")
    show("+ team SP+", *fit(bd, "exp_cons", "rec_z", ("sp_z",)), "rec_z")
    show("+ age at draft", *fit(bd, "exp_cons", "rec_z", ("age_z",)), "rec_z")
    show("+ non-P5", *fit(bd.assign(small=bd.small.astype(float)), "exp_cons", "rec_z", ("small",)), "rec_z")
    show("offense", *fit(bd[bd.off], "exp_cons", "rec_z"), "rec_z")
    show("defense", *fit(bd[~bd.off], "exp_cons", "rec_z"), "rec_z")
    show("no five-stars", *fit(bd[bd.five_star != 1], "exp_cons", "rec_z"), "rec_z")
    q = bd.dropna(subset=["rec_z"]).assign(qt=lambda t: pd.qcut(t.rec_z, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]))
    print("    actual / expected by recruiting quartile:",
          {k: round(g.actual.sum() / g.exp_cons.sum(), 2) for k, g in q.groupby("qt", observed=True)})

    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("slot", "consensus_pre"), test_classes=NEW)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R["player_id"] = R.player_id.astype(str)
    n_seen = {c: pm.LAST_SEASON_PLAYED - c + 1 for c in NEW}
    for col, v in [("actual", "y"), ("exp_slot", "slot_p"), ("exp_cons", "consensus_pre_p")]:
        R[col] = [sum(r[f"{v}{t}"] for t in pm.SEASONS[:n_seen[r.cls]]) for _, r in R.iterrows()]
    r = R[["player_id", "cls", "actual", "exp_slot", "exp_cons"]].merge(u, on=["player_id", "cls"], how="left")
    dr = r[r.drafted.astype(bool)]
    br = r[r.on_board == 1]
    print(f"\nPART B: FRESH REPLICATION 2023-25 (seasons seen {n_seen})")
    show("PRIMARY recruiting composite vs consensus", *fit(br, "exp_cons", "rec_z"), "rec_z", "<")
    for c in NEW:
        show(f"  class {c}", *fit(br[br.cls == c], "exp_cons", "rec_z"), "rec_z", "<")
    show("  (recruiting vs slot, for reference)", *fit(dr, "exp_slot", "rec_z"), "rec_z", "<")
    show("one-year wonder vs slot", *fit(dr, "exp_slot", "one_year_wonder"), "one_year_wonder", "<")
    show("Senior Bowl vs consensus", *fit(br, "exp_cons", "senior_bowl"), "senior_bowl", ">")
    show("non-Power-5 vs consensus", *fit(br, "exp_cons", "small"), "small", ">")


if __name__ == "__main__":
    main()
