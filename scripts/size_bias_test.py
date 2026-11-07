"""Are players mispriced for height or for falling just short of scouts' size cutoffs?

A bias toward a trait shows up as both: (1) the trait moves draft position relative to the consensus board,
and (2) players with it then miss their draft slot (overvalued) while players without it beat it (undervalued).
Measurements: cfdarko combine_metrics (raw height / arm / hand, inches; arm and hand from 2011), positions
OT / CB from data/raw/combine.parquet (PFR). Specified before looking:
  height      within position group (z), all groups; short QB = under 6'1" (73 in)
  cutoffs     QB hand < 9.0 in, OT arm < 33 in, CB height < 5'11" (71 in); whole position and a band
              around the cutoff (hand 8.5-9.5, arm 32-34, height 69-73) where players are otherwise similar
Part 1 (pricing): drafted board players 2014-2024, log((board rank + 10) / (pick + 10)) > 0 = taken earlier
than the board; regression on the trait with position fixed effects.
Part 2 (value): drafted 2016-22 vs draft slot, board 2018-22 vs consensus, out of sample
(scripts/multisport_test.expectations), class fixed effects.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT

CUTS = [("QB hand < 9.0 in", "QB", None, "hand", 9.0, (8.5, 9.5)),
        ("OT arm < 33 in", "OL", "OT", "arm", 33.0, (32.0, 34.0)),
        ("CB height < 5'11\"", "DB", "CB", "ht", 71.0, (69.0, 73.0))]


def load():
    uni = pm.load()
    c = pd.read_parquet(pm.CF / "data/processed/combine_metrics.parquet")[
        ["player_id", "combine_ht", "combine_wt", "combine_arm", "combine_hand"]].rename(
        columns=lambda x: x.replace("combine_", ""))
    c["player_id"] = c.player_id.astype(str)
    k = pd.read_parquet("data/raw/combine.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last")
    u = uni[["player_id", "cls", "grp", "pfr_id", "hasan_rank", "pick", "drafted", "on_board"]].copy()
    u["player_id"] = u.player_id.astype(str)
    u = u.merge(c.drop_duplicates("player_id"), on="player_id", how="left").merge(
        k[["pfr_id", "pos"]], on="pfr_id", how="left")
    u["ht_z"] = u.groupby("grp").ht.transform(lambda s: (s - s.mean()) / s.std())
    u["wt_z"] = u.groupby("grp").wt.transform(lambda s: (s - s.mean()) / s.std())
    return uni, u


def flags(d):
    d = d.copy()
    d["short_qb"] = np.where((d.grp == "QB") & d.ht.notna(), d.ht < 73, np.nan)
    for lab, g, pos, col, cut, band in CUTS:
        at = (d.grp == g) & (d.pos == pos if pos else True) & d[col].notna()
        d[lab] = np.where(at, d[col] < cut, np.nan)
        d[lab + " band"] = np.where(at & d[col].between(*band), d[col] < cut, np.nan)
    return d


def reach(d, x, label, extra=()):
    b = d[d.drafted.astype(bool) & d.hasan_rank.notna() & d.pick.notna() & d[x].notna()].copy()
    if len(b) < 20:
        return
    b["reach"] = np.log((b.hasan_rank + 10) / (b.pick + 10))
    fe = pd.get_dummies(b.grp, prefix="g", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([b[[x, *extra]].astype(float), fe], axis=1))
    m = sm.OLS(b.reach, X).fit(cov_type="HC1")
    print(f"    {label:40s} n={len(b):4d} ({int(b[x].sum()) if b[x].isin([0, 1]).all() else '-'} flagged)  "
          f"taken x{np.exp(m.params[x]):.2f} earlier than board (p {m.pvalues[x]:.2f})")


def value_cont(s, e, x, label):
    s = s.dropna(subset=[e, x]).copy()
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([s[[x]].astype(float), fe], axis=1))
    m = sm.GLM(s.actual, X, family=sm.families.Poisson(), offset=np.log(s[e].clip(lower=1e-4))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc[x])
    print(f"    {label:40s} n={len(s):4d}  per 1 sd: value x{np.exp(m.params[x]):.2f} (95% {lo:.2f}-{hi:.2f}, p {m.pvalues[x]:.2f})")


def main():
    uni, u = load()
    u = flags(u)
    fl = ["short_qb"] + [c for lab, *_ in CUTS for c in (lab, lab + " band")]
    print("PART 1: PRICING (drafted board players 2014-24; > 1 = taken earlier than board rank)")
    reach(u, "ht_z", "height (per sd within position)", extra=("wt_z",))
    for g in ["QB", "EDGE", "DB", "WR", "OL"]:
        reach(u[u.grp == g], "ht_z", f"  height, {g}", extra=("wt_z",))
    for f in fl:
        reach(u[u[f].notna()].assign(**{f: lambda x, f=f: x[f].astype(float)}), f, f)

    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R[["player_id", "cls", "actual", "exp_slot", "exp_cons"]].merge(u, on=["player_id", "cls"], how="left")
    dr = d[d.drafted.astype(bool) & d.exp_slot.notna()]
    bd = d[(d.on_board == 1) & d.exp_cons.notna()]
    for e, s, nm in [("exp_slot", dr, "DRAFTED vs SLOT 2016-22"), ("exp_cons", bd, "BOARD vs CONSENSUS 2018-22")]:
        print(f"\nPART 2: VALUE, {nm}")
        value_cont(s, e, "ht_z", "height (per sd within position)")
        for g in ["QB", "EDGE", "DB", "WR", "OL"]:
            value_cont(s[s.grp == g], e, "ht_z", f"  height, {g}")
        for f in fl:
            t = s[s[f].notna()].copy()
            t[f] = t[f].astype(bool)
            fe = pd.get_dummies(t.cls, prefix="c", drop_first=True, dtype=float)
            t = pd.concat([t, fe], axis=1)
            compare(t, f, e, covs=tuple(fe.columns), label=f, min_n=5)
    d.to_parquet(OUT / "size_bias.parquet")


if __name__ == "__main__":
    main()
