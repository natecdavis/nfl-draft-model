"""Draft biases from the literature that this project had not yet tested (Nov 2026).

Each is tested the same way as scripts/size_bias_test.py:
  Part 1 (pricing): drafted board players 2014-24, log((board rank + 10) / (pick + 10)), > 1 = taken earlier than
          the consensus board, position fixed effects.
  Part 2 (value): drafted 2016-22 vs draft slot, board 2018-22 vs consensus, out of sample
          (multisport_test.expectations), Poisson with class fixed effects. < 1 = the trait is overvalued.

Pre-specified (written before running), one primary measure per bias:
  1. Program strength - Kitchens (2015, Economic Inquiry): players from top programs are drafted earlier than
     their ability warrants, with no better careers. Primary: final-season team SP+ (z). Predict value < 1.
  2. Small school - Hendricks, DeBrock & Koenker (2003, J Labor Econ): statistical discrimination vs option value
     for players with noisier signals. Primary: non-Power-5 (G5 or FCS). Direction open (two theories disagree).
  3. Recruiting pedigree (anchoring on high-school ratings; no published test vs slot found). Primary: 247
     composite rating (z within position, rated players only). Predict value < 1.
  4. Final-season jump / "one-year wonder" - PFF (2016-18 classes; recency bias). Primary: last college season's
     PFF grade minus best earlier season (z within position; seasons with 150+ snaps; needs 2+ seasons).
     Predict value < 1.
  5. All-star showcase exposure - Treme & Allen-style media exposure. Primary: Senior Bowl roster. Direction open.
  6. Relative age - Heneghan & Herron (2019, JQAS) find no draft inefficiency; Beyer et al. (2016) do. Primary:
     born Sep-Dec (oldest in a Sep 1 school cohort) vs May-Aug, drafted players only (birth dates from nflverse
     cover NFL players), controlling age at draft. Predict none.
Six primary tests x two yardsticks: p < 0.004 survives Bonferroni; p < 0.05 alone is a lead, not a finding.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT
from size_bias_test import reach, value_cont

CFP = os.path.expanduser("~/cfdarko/data/processed/")
OFF = {"QB", "RB", "WR", "TE", "OL"}
MIN_SNAPS = 150


def zgrp(d, col):
    return d.groupby("grp")[col].transform(lambda s: (s - s.mean()) / s.std())


def pff_jump(u):
    g = pd.read_parquet(CFP + "pff_season_grades.parquet",
                        columns=["player_id", "season_year", "pff_snap_counts_offense", "pff_snap_counts_defense",
                                 "pff_grades_offense", "pff_grades_defense"])
    g["player_id"] = g.player_id.astype(str)
    g = g.merge(u[["player_id", "cls", "grp"]].drop_duplicates("player_id"), on="player_id")
    off = g.grp.isin(OFF)
    g["grade"] = np.where(off, g.pff_grades_offense, g.pff_grades_defense)
    g["snaps"] = np.where(off, g.pff_snap_counts_offense, g.pff_snap_counts_defense)
    g = g[(g.snaps >= MIN_SNAPS) & g.grade.notna() & (g.season_year <= g.cls - 1)]
    g = g.groupby(["player_id", "season_year"], as_index=False).agg(grade=("grade", "max"), cls=("cls", "first"))
    last = g[g.season_year == g.cls - 1].set_index("player_id").grade.rename("pff_last")
    prior = g[g.season_year < g.cls - 1].groupby("player_id").grade.max().rename("pff_prior_best")
    out = pd.concat([last, prior], axis=1).dropna()
    out["pff_jump"] = out.pff_last - out.pff_prior_best
    return out.reset_index()


def load():
    uni = pm.load()
    u = uni[["player_id", "cls", "grp", "hasan_rank", "pick", "drafted", "on_board", "age_at_draft", "pfr_id",
             "recruiting_composite", "recruiting_stars"]].copy()
    u["player_id"] = u.player_id.astype(str)
    a = pd.read_parquet(CFP + "player_attributes.parquet",
                        columns=["player_id", "school", "conference_tier", "team_sp_overall"])
    a["player_id"] = a.player_id.astype(str)
    u = u.merge(a.drop_duplicates("player_id"), on="player_id", how="left")
    u = u.merge(pff_jump(u), on="player_id", how="left")
    s = pd.read_parquet(CFP + "allstar_invites.parquet")
    s["player_id"] = s.player_id.astype(str)
    sb = s[s.game == "senior"][["player_id", "year"]].rename(columns={"year": "cls"}).assign(senior_bowl=1)
    u = u.merge(sb.drop_duplicates(["player_id", "cls"]), on=["player_id", "cls"], how="left")
    p = pd.read_parquet("data/raw/players.parquet", columns=["pfr_id", "birth_date"]).dropna().drop_duplicates("pfr_id")
    u = u.merge(p, on="pfr_id", how="left")

    u["sp_z"] = (u.team_sp_overall - u.team_sp_overall.mean()) / u.team_sp_overall.std()
    u["small"] = np.where(u.conference_tier.notna(), u.conference_tier >= 2, np.nan)
    u["fcs"] = np.where(u.conference_tier.notna(), u.conference_tier == 3, np.nan)
    u["rec_z"] = zgrp(u, "recruiting_composite")
    u["five_star"] = np.where(u.recruiting_stars.notna(), u.recruiting_stars == 5, np.nan)
    u["jump_z"] = zgrp(u, "pff_jump")
    u["one_year_wonder"] = np.where(u.pff_jump.notna(), (u.pff_last >= 80) & (u.pff_prior_best < 70), np.nan)
    sb_years = set(s[s.game == "senior"].year)
    u["senior_bowl"] = np.where(u.cls.isin(sb_years), u.senior_bowl.fillna(0), np.nan)
    m = pd.to_datetime(u.birth_date, errors="coerce").dt.month
    u["old_in_cohort"] = np.where(m.isin([9, 10, 11, 12]), 1.0, np.where(m.isin([5, 6, 7, 8]), 0.0, np.nan))
    u["age_z"] = (u.age_at_draft - u.age_at_draft.mean()) / u.age_at_draft.std()
    return uni, u


CONT = [("sp_z", "1 program strength: team SP+ (per sd)"),
        ("rec_z", "3 recruiting composite (per sd in pos)"),
        ("jump_z", "4 final-season PFF jump (per sd in pos)")]
FLAGS = [("small", "2 non-Power-5 school"),
         ("fcs", "2b FCS / lower division"),
         ("five_star", "3b five-star recruit"),
         ("one_year_wonder", "4b one-year wonder (80+ after <70)"),
         ("senior_bowl", "5 Senior Bowl roster"),
         ("old_in_cohort", "6 born Sep-Dec (vs May-Aug)")]


def main():
    uni, u = load()
    print("PART 1: PRICING (drafted board players 2014-24; > 1 = taken earlier than board rank)")
    for x, lab in CONT:
        reach(u, x, lab)
    for x, lab in FLAGS:
        reach(u[u[x].notna()].assign(**{x: lambda d, x=x: d[x].astype(float)}), x, lab)
    reach(u[u.old_in_cohort.notna()], "old_in_cohort", "6 born Sep-Dec, age controlled", extra=("age_z",))

    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R[["player_id", "cls", "actual", "exp_slot", "exp_cons"]].merge(
        u.drop(columns=["age_at_draft", "pfr_id"]), on=["player_id", "cls"], how="left")
    dr = d[d.drafted.astype(bool) & d.exp_slot.notna()]
    bd = d[(d.on_board == 1) & d.exp_cons.notna()]
    for e, s, nm in [("exp_slot", dr, "DRAFTED vs SLOT 2016-22"), ("exp_cons", bd, "BOARD vs CONSENSUS 2018-22")]:
        print(f"\nPART 2: VALUE, {nm}  (< 1 = trait overvalued)")
        for x, lab in CONT:
            value_cont(s, e, x, lab)
        for x, lab in FLAGS:
            if x == "old_in_cohort" and e == "exp_cons":
                print(f"    {lab:38s} (skipped: birth dates only for players who reached the NFL)")
                continue
            t = s[s[x].notna()].copy()
            t[x] = t[x].astype(bool)
            covs = ["age_z"] if x == "old_in_cohort" else []
            fe = pd.get_dummies(t.cls, prefix="c", drop_first=True, dtype=float)
            t = pd.concat([t, fe], axis=1)
            compare(t.dropna(subset=covs), x, e, covs=tuple(covs) + tuple(fe.columns), label=lab, min_n=5)
    d.to_parquet(OUT / "lit_bias.parquet")


if __name__ == "__main__":
    main()
