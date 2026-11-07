"""Body growth from recruit to combine, and position changes since high school.

Recruit height / weight / position: College Football Data API (/recruiting/players, high-school
recruits 2006-2022, cached in data/raw/cfbd_recruits/), joined on the CFBD athlete id (= player_id).
Combine height / weight: data/raw/combine.parquet, joined on pfr_id.

Questions (out of sample, scripts/multisport_test.expectations):
  1. At the same final size and class, do players who grew more since high school beat their draft slot
     (drafted 2016-2022) or consensus rank (board 2018-2022)? Growth = combine minus recruit
     listing, standardized within position; controls: combine weight (or height) z within position
     and years since the recruiting class.
  2. Do players recruited as 'athletes' (ATH) or at a different position group beat expectations?
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT

# recruit position -> NFL position groups it is consistent with
FITS = {"WR": {"WR"}, "CB": {"DB"}, "S": {"DB"}, "OT": {"OL"}, "OG": {"OL"}, "OC": {"OL"}, "IOL": {"OL"},
        "OLB": {"LB", "EDGE"}, "ILB": {"LB"}, "LB": {"LB", "EDGE"}, "SDE": {"EDGE", "DL"}, "WDE": {"EDGE", "DL"},
        "EDGE": {"EDGE", "DL", "LB"}, "DT": {"DL"}, "DL": {"DL", "EDGE"}, "TE": {"TE"}, "FB": {"TE", "RB"},
        "PRO": {"QB"}, "DUAL": {"QB"}, "QB": {"QB"}, "RB": {"RB"}, "APB": {"RB"}}
BIG = {"OL", "DL", "EDGE", "TE"}


def recruits():
    r = pd.concat([pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob("data/raw/cfbd_recruits/*.json"))])
    r["pid"] = pd.to_numeric(r.athleteId, errors="coerce")
    r = r.dropna(subset=["pid"]).sort_values("year").drop_duplicates("pid", keep="last")  # latest listing (e.g. JUCO -> HS dupes)
    return r[["pid", "year", "position", "height", "weight", "rating"]].rename(
        columns={"year": "rec_year", "position": "rec_pos", "height": "rec_ht", "weight": "rec_wt"})


def inches(s):
    try:
        f, i = str(s).split("-")
        return int(f) * 12 + int(i)
    except ValueError:
        return np.nan


def build(R, uni):
    c = pd.read_parquet("data/raw/combine.parquet").dropna(subset=["pfr_id"])
    c = c.assign(cmb_ht=c.ht.map(inches), cmb_wt=c.wt).sort_values("season").drop_duplicates("pfr_id", keep="last")
    add = [c for c in ["grp", "drafted", "pick"] if c not in R]
    d = R.merge(uni[["player_id", "cls", *add]], on=["player_id", "cls"], how="left") if add else R.copy()
    d["pid"] = pd.to_numeric(d.player_id, errors="coerce")
    d = d.merge(recruits(), on="pid", how="left").merge(c[["pfr_id", "cmb_ht", "cmb_wt"]], on="pfr_id", how="left")
    d["years"] = d.cls - d.rec_year
    d.loc[(d.years < 2) | (d.years > 7), ["rec_ht", "rec_wt", "rec_pos"]] = np.nan  # bad joins
    d["wt_gain"] = d.cmb_wt - d.rec_wt
    d["ht_gain"] = d.cmb_ht - d.rec_ht
    z = lambda col: d.groupby("grp")[col].transform(lambda s: (s - s.mean()) / s.std())
    for col in ["wt_gain", "ht_gain", "cmb_wt", "cmb_ht"]:
        d[col + "_z"] = z(col)
    d["gain_top"] = d.wt_gain_z > d.groupby("grp").wt_gain_z.transform(lambda s: s.quantile(0.75))
    d["ath"] = d.rec_pos.eq("ATH")
    d["moved"] = [isinstance(p, str) and p in FITS and g not in FITS[p] for p, g in zip(d.rec_pos, d.grp)]
    return d


def cont(d, e, x, covs, label):
    d = d.dropna(subset=[e, x, *covs])
    X = sm.add_constant(d[[x, *covs]].astype(float))
    m = sm.GLM(d.actual, X, family=sm.families.Poisson(), offset=np.log(d[e].clip(lower=1e-4))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc[x])
    print(f"    {label:44s} n={len(d):4d}  per 1 sd: value x{np.exp(m.params[x]):.2f} (95% {lo:.2f}-{hi:.2f}, p {m.pvalues[x]:.2f})")


def main():
    uni = pm.load()
    R = expectations(uni, SLOT_CLASSES)
    d = build(R, uni)
    # class fixed effects: 2021-22 recruits gained less (no 2021 combine; shorter careers) and the out-of-sample
    # models miss by class, so pooling without them inflated the consensus result (x1.15 -> x1.09)
    for c in SLOT_CLASSES:
        d[f"c{c}"] = (d.cls == c).astype(float)
    dr = d[d.drafted.astype(bool) & d.exp_slot.notna()]
    bd = d[(d.on_board == 1) & d.exp_cons.notna()]
    for lab, s in [("drafted 2016-22", dr), ("board 2018-22", bd)]:
        print(f"{lab}: {len(s)}; recruit listing {s.rec_wt.notna().sum()}; + combine weight {s.wt_gain.notna().sum()}; "
              f"mean gain {s.wt_gain.mean():.0f} lb, {s.ht_gain.mean():.1f} in; ATH {s.ath.sum()}; moved group {s.moved.sum()}")
    print("\nWeight gained per position, drafted (lb, median):",
          dr.groupby("grp").wt_gain.median().dropna().round(0).astype(int).to_dict())

    for e, s, nm in [("exp_slot", dr, "VS DRAFT SLOT"), ("exp_cons", bd, "VS CONSENSUS")]:
        print(f"\n{nm}")
        fe = [f"c{c}" for c in sorted(s.cls.unique())[1:]]  # one dummy per class in this sample, first class as baseline
        print("  weight gained (same combine weight, same years since recruiting)")
        cont(s, e, "wt_gain_z", ["cmb_wt_z", "years", *fe], "all positions")
        cont(s[s.grp.isin(BIG)], e, "wt_gain_z", ["cmb_wt_z", "years", *fe], "OL / DL / EDGE / TE")
        cont(s[~s.grp.isin(BIG)], e, "wt_gain_z", ["cmb_wt_z", "years", *fe], "other positions")
        cont(s[s.pick <= 100], e, "wt_gain_z", ["cmb_wt_z", "years", *fe], "picks 1-100")
        compare(s.dropna(subset=["wt_gain_z"]), "gain_top", e, covs=("cmb_wt_z", "years", *fe), label="top quarter of gain at position")
        print("  height gained (same combine height)")
        cont(s, e, "ht_gain_z", ["cmb_ht_z", "years", *fe], "all positions")
        print("  recruited elsewhere")
        known = s[s.rec_pos.notna()]
        compare(known, "ath", e, covs=fe, label="recruited as ATH")
        compare(known, "moved", e, covs=fe, label="recruited at another position group")
        compare(known[~known.grp.isin(["QB"])], "moved", e, covs=fe, label="  same, excluding QBs")
        cont(known, e, "years", fe, "years from recruiting class to draft")
    d.to_parquet(OUT / "growth.parquet")


if __name__ == "__main__":
    main()
