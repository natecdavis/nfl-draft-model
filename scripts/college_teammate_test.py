"""Does a rookie do better when his NFL team has a former college teammate on the roster?

Specified before looking (Nov 22 2026):
  Former college teammate = someone on the drafting team's roster in the rookie season (weekly rosters, any week,
  active / inactive / reserve / practice squad) with the same college (players.college_name, last school only) whose
  final college season is within 3 years of the rookie's (final season = draft year - 1, or first NFL season - 1
  for undrafted players). Includes same-class draftees.
  PRIMARY: drafted players 2016-22 (4 NFL seasons of value) with a possible teammate (at least one NFL team had
  one), drafted by a team that had one vs not, value vs draft slot out of sample (multisport_test.compare),
  controlling for how many teams had one (log1p) and the coaching link (a staff member coached at his college;
  coaching picks beat slot x1.47 odds, and teams drafting a school's players may share staff).
  SECONDARY: veteran teammate only (not same class); window 2 years; same position group; seasons 1-2 vs 3-4;
  picks 1-100 vs 101+.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT

RAW = OUT.parent / "raw"
TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "GB": "GNB", "KC": "KAN", "LA": "LAR", "SL": "LAR",
        "SD": "LAC", "SDG": "LAC", "LV": "LVR", "OAK": "LVR", "NE": "NWE", "NO": "NOR", "SF": "SFO", "TB": "TAM"}
GRP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
       "OT": "OL", "OG": "OL", "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB",
       "LB": "LB", "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "SAF": "DB"}


def rosters():
    r = pd.read_parquet(RAW / "rosters_weekly.parquet")
    r = r[~r.status.isin(["CUT", "RET", "TRD"])]
    r["team"] = r.team.replace(TEAM)
    p = pd.read_parquet(RAW / "players.parquet")
    p["fin"] = p.draft_year.fillna(p.rookie_season) - 1
    p["pgrp"] = p.position.map(GRP)
    r = r[["season", "team", "gsis_id"]].drop_duplicates().merge(
        p[["gsis_id", "college_name", "fin", "pgrp", "draft_year"]], on="gsis_id")
    return r.dropna(subset=["college_name", "fin"]), p


def flags(d, r, window, vet_only=False, same_grp=False):
    """Per rookie: did his team have a teammate; how many teams had one."""
    key = r.set_index(["season", "college_name"]).sort_index()
    has, nteams = [], []
    for gsis, col, yr, tm, fin, grp in zip(d.gsis_id, d.college_name, d.cls, d.team, d.fin, d.pgrp):
        try:
            m = key.loc[[(yr, col)]]
        except KeyError:
            has.append(False); nteams.append(0); continue
        m = m[(m.gsis_id != gsis) & ((m.fin - fin).abs() <= window)]
        if vet_only:
            m = m[m.draft_year.fillna(0) != yr]
            m = m[m.fin < fin]
        if same_grp:
            m = m[m.pgrp == grp]
        has.append(tm in set(m.team)); nteams.append(m.team.nunique())
    return np.array(has), np.array(nteams)


def main():
    uni = pm.load()
    r, p = rosters()
    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    R = R[R.drafted.astype(bool) & R.exp_slot.notna() & R.pfr_id.notna()]
    d = R.merge(p[["pfr_id", "gsis_id", "college_name", "pgrp"]].dropna(subset=["pfr_id"]), on="pfr_id", how="left")
    dp = pd.read_parquet(RAW / "draft_picks.parquet")[["season", "pfr_player_id", "team"]]
    d = d.merge(dp.rename(columns={"season": "cls", "pfr_player_id": "pfr_id"}), on=["cls", "pfr_id"], how="left")
    d["team"] = d.team.replace(TEAM)
    d["fin"] = d.cls - 1
    c = pd.read_parquet(OUT / "coaching_picks.parquet")[["pfr_id", "cls", "conn"]].dropna(subset=["pfr_id"])
    d = d.merge(c.drop_duplicates(["pfr_id", "cls"]), on=["pfr_id", "cls"], how="left")
    d["conn"] = d.conn.fillna(False).astype(float)
    miss = d.college_name.isna() | d.team.isna() | d.gsis_id.isna()
    print(f"drafted 2016-22 with slot expectation: {len(d)}; unlinked (college/team/gsis) {miss.sum()}")
    d = d[~miss].copy()

    d["tm8"], d["n8"] = flags(d, r, 3)
    d["possible"] = d.n8 > 0
    d["log_n"] = np.log1p(d.n8)
    covs = ("log_n", "conn")
    print(f"possible teammate {d.possible.sum()}; drafted by a team with one {d.tm8.sum()} "
          f"(mean teams with one {d.n8[d.possible].mean():.1f}); coaching-linked among them {int(d.conn[d.tm8].sum())}")
    p_ = d[d.possible]
    print("\nPRIMARY: possible-teammate players, team had a former college teammate vs not (vs draft slot)")
    compare(p_, "tm8", "exp_slot", covs=covs, label="college teammate on roster (+-3 yrs)")
    print("\nSECONDARY")
    compare(d, "tm8", "exp_slot", covs=covs, label="all drafted players")
    compare(p_[p_.conn == 0], "tm8", "exp_slot", covs=("log_n",), label="no coaching link")
    for lab, kw, nm in [("veteran teammate only", dict(vet_only=True), "vet"),
                        ("window 2 years", dict(window=2), "w2"),
                        ("same position group", dict(same_grp=True), "grp")]:
        w = kw.pop("window", 3)
        d[f"tm_{nm}"], d[f"n_{nm}"] = flags(d, r, w, **kw)
        q = d[d[f"n_{nm}"] > 0].copy()
        q["log_n"] = np.log1p(q[f"n_{nm}"])
        compare(q, f"tm_{nm}", "exp_slot", covs=covs, label=lab)
    compare(p_[p_.pick <= 100], "tm8", "exp_slot", covs=covs, label="picks 1-100")
    compare(p_[p_.pick > 100], "tm8", "exp_slot", covs=covs, label="picks 101+")
    for s, lab in [((1, 2), "seasons 1-2"), ((3, 4), "seasons 3-4")]:
        q = p_.copy()
        q["actual"] = q[[f"y{k}" for k in s]].sum(axis=1)
        q["exp_s"] = q[[f"slot_p{k}" for k in s]].sum(axis=1)
        compare(q, "tm8", "exp_s", covs=covs, label=lab)
    for y, g in p_.groupby("cls"):
        rr = lambda x: x.actual.sum() / x.exp_slot.sum()
        print(f"    {y}: teammate {g.tm8.sum():3d} of {len(g):3d}  value x{rr(g[g.tm8]) / rr(g[~g.tm8]):.2f}")
    d.drop(columns=[x for x in d.columns if d[x].dtype == object and x not in ("player_name", "college_name", "team")]) \
        .to_parquet(OUT / "college_teammate.parquet")


if __name__ == "__main__":
    main()
