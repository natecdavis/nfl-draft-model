"""Old-class replication, part 3: the remaining borderline leads that can reach pre-2016 classes (vs draft slot).

Same design as scripts/old_class_replication.py: drafted classes 2006-2015 (careers complete, never in the original
tests), slot model fit leave-one-class-out within 2006-15, seasons 1-4, compare() with class fixed effects.

Specified before looking (originals in brackets; direction predicted from the original):
  E. Sons of NFL players [hand-coded 2018-22: value x1.15-1.16, beat slot 56-61% vs 38%, p 0.02-0.04; > 1].
     Old classes have no hand coding, so a name rule: player with suffix Jr. / II / III / IV whose base name equals
     an NFL player's (nflreadpy players, rookie season 1974+) who entered 18-40 years before the son's class.
     Validated first on drafted 2018-22 against the hand-coded fathers (precision and recall printed).
     RESULT: rule fails (2018-22 precision 56%, recall 11%; nflverse 2006-15 draft names carry no suffixes, 0 flagged).
     Wikipedia's 'List of second-generation NFL players' was checked as a source and rejected: it lists 83% of
     hand-coded drafted sons with above-median careers but 40% below and 29% of undrafted (selected on outcome).
  F. MLB-drafted [too few in 2016-22; guide-text baseball x1.23 vs slot p 0.07; > 1]. src/multisport.match_mlb on
     MLB drafts 1999-2015 (1999-2004 fetched for this test), so every class gets its 7-year window.
  G. One-year wonder: 80+ PFF grade in the final college season after < 70 best before [x0.78 vs slot p 0.04;
     2023-25 replication x1.02; < 1]. PFF college grades begin with the 2009 season: classes 2011-15.
     Context: continuous final-season jump per sd [x0.97].
     NOT RUN: cfdarko's PFF college grades (processed and raw) start with the 2014 season, i.e. the 2016 class;
     raw 2009-13 folders hold only pass-rush productivity / slot coverage / time in pocket. Untestable here.
  H. OT arm under 33 in [x1.31 vs slot p 0.13, 10 players; > 1]. Arms from 2011: classes 2011-15. Also the
     32-34 in band.
"""
import re
import sys

import nflreadpy as nr
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import multisport as ms
import prospect_model as pm
from lit_bias_test import pff_jump
from multisport_test import compare
from nfl_fathers import DRAFTED, PLAYED
from old_class_replication import expected_slot

OUT = pm.OUT
SUFFIX = r"\s+(jr|sr|ii|iii|iv|v)\.?$"


def base(name):
    n = re.sub(r"[^a-z' ]", " ", str(name).lower().replace(".", "")).replace("'", "")
    n = " ".join(n.split())
    m = re.search(SUFFIX, n)
    return re.sub(SUFFIX, "", n), (m.group(1) if m else None)


def son_flag(d, players):
    """d: player_name, cls. True where the name carries Jr./II/III/IV and an NFL player with the base name entered
    18-40 years before the class."""
    p = players.dropna(subset=["display_name"]).copy()
    p["b"] = [base(x)[0] for x in p.display_name]
    p["yr"] = p.rookie_season.fillna(p.draft_year)
    yrs = p.dropna(subset=["yr"]).groupby("b").yr.agg(list).to_dict()
    out = []
    for name, c in zip(d.player_name, d.cls):
        b, s = base(name)
        out.append(s in {"jr", "ii", "iii", "iv"} and any(c - 40 <= y <= c - 18 for y in yrs.get(b, [])))
    return np.array(out)


def fe_cols(s):
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    return pd.concat([s.reset_index(drop=True), fe.reset_index(drop=True)], axis=1), tuple(fe.columns)


def sons_part(d, players):
    # validation on 2018-22 drafted players (universe names) against the hand-coded fathers
    u = pm.load()
    v = u[u.drafted.astype(bool) & u.cls.between(2018, 2022)][["player_name", "cls"]].copy()
    v["rule"] = son_flag(v, players)
    coded = {(base(n)[0], y) for dct in (PLAYED, DRAFTED) for y, ns in dct.items() for n in ns}
    v["coded"] = [(base(n)[0], c) in coded for n, c in zip(v.player_name, v.cls)]
    tp = (v.rule & v.coded).sum()
    print(f"\nE. SONS: rule on drafted 2018-22: flagged {v.rule.sum()}, of which hand-coded sons {tp} "
          f"(precision {tp / max(v.rule.sum(), 1):.0%}); recall {tp}/{v.coded.sum()} ({tp / max(v.coded.sum(), 1):.0%})")
    print("   flagged, not hand-coded: " + ", ".join(v[v.rule & ~v.coded].player_name.head(15)))
    d = d.copy()
    d["son"] = son_flag(d, players)
    print(f"   2006-15 flagged: {d.son.sum()}; by class {d[d.son].groupby('cls').size().to_dict()}")
    print("   e.g. " + ", ".join(d[d.son].sort_values("pick").player_name.head(20)))
    s, fe = fe_cols(d)
    compare(s, "son", "exp_slot", label="E primary: sons vs slot")
    compare(s, "son", "exp_slot", covs=fe, label="+ class fixed effects")
    compare(s[s.cls <= 2010], "son", "exp_slot", label="2006-10")
    compare(s[s.cls >= 2011], "son", "exp_slot", label="2011-15")
    return d[["pfr_id", "son"]]


def mlb_part(d, players):
    birth = players.dropna(subset=["pfr_id", "birth_date"]).drop_duplicates("pfr_id").set_index("pfr_id").birth_date
    u = d.copy()
    u["player_id"] = u.player_id.where(u.player_id.notna(), u.pfr_id).astype(str)
    u["age_at_draft"] = (pd.to_datetime(u.cls.astype(str) + "-04-25") - pd.to_datetime(u.pfr_id.map(birth), errors="coerce")).dt.days / 365.25
    f = ms.mlb_flags(u)
    u = u.merge(f, on="player_id", how="left")
    u["mlb"] = u.mlb_drafted.fillna(False).astype(bool)
    m = u[u.mlb]
    print(f"\nF. MLB-DRAFTED: {len(m)} drafted NFL players 2006-15 (birth-date confirmed {int(m.mlb_confirmed.sum())}); "
          f"by class {m.groupby('cls').size().to_dict()}")
    print("   " + ", ".join(f"{n} ({g}, MLB rd {int(r) if r == r else '?'})" for n, g, r in
                            zip(m.sort_values("pick").player_name, m.sort_values("pick").grp, m.sort_values("pick").mlb_best_round)))
    s, fe = fe_cols(u)
    compare(s, "mlb", "exp_slot", label="F primary: MLB-drafted vs slot", min_n=5)
    compare(s, "mlb", "exp_slot", covs=fe, label="+ class fixed effects", min_n=5)
    compare(s[s.grp.isin(["QB", "WR", "DB"])], "mlb", "exp_slot", label="QB / WR / DB", min_n=5)
    return u[["pfr_id", "mlb"]]


def wonder_part(d):
    u = d.assign(player_id=d.player_id.astype(str))
    j = pff_jump(u[u.player_id != "None"])
    u = u.merge(j, on="player_id", how="left")
    u = u[u.pff_jump.notna()].copy()
    u["oyw"] = (u.pff_last >= 80) & (u.pff_prior_best < 70)
    u["jump_z"] = u.groupby("grp").pff_jump.transform(lambda s: (s - s.mean()) / s.std())
    print(f"\nG. ONE-YEAR WONDER: drafted with a final and an earlier PFF season {len(u)}; by class "
          f"{u.groupby('cls').size().to_dict()}; one-year wonders {u.oyw.sum()}")
    s, fe = fe_cols(u)
    compare(s, "oyw", "exp_slot", label="G primary: 80+ after < 70 vs slot")
    compare(s, "oyw", "exp_slot", covs=fe, label="+ class fixed effects")
    from size_bias_test import value_cont
    value_cont(u, "exp_slot", "jump_z", "context: jump per sd in position")
    return u[["pfr_id", "oyw", "jump_z"]]


def arm_part(d):
    c = pd.read_parquet(pm.CF / "data/processed/combine_metrics.parquet")[["player_id", "combine_arm"]]
    c["player_id"] = c.player_id.astype(str)
    k = pd.read_parquet("data/raw/combine.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last")
    u = d.assign(player_id=d.player_id.astype(str)).merge(c.drop_duplicates("player_id"), on="player_id", how="left") \
        .merge(k[["pfr_id", "pos"]], on="pfr_id", how="left")
    u = u[(u.grp == "OL") & (u.pos == "OT") & u.combine_arm.notna()].copy()
    u["short_arm"] = u.combine_arm < 33
    print(f"\nH. OT ARM < 33 in: drafted OTs with arm length {len(u)} (classes {u.cls.min()}-{u.cls.max()}); "
          f"short {u.short_arm.sum()}")
    s, fe = fe_cols(u)
    compare(s, "short_arm", "exp_slot", label="H primary: OT arm < 33 vs slot", min_n=5)
    compare(s, "short_arm", "exp_slot", covs=fe, label="+ class fixed effects", min_n=5)
    compare(s[s.combine_arm.between(32, 34)], "short_arm", "exp_slot", label="band 32-34 in", min_n=5)
    return u[["pfr_id", "short_arm", "combine_arm"]]


def main():
    d = expected_slot()
    players = nr.load_players().to_pandas()
    out = d[["pfr_id", "player_name", "cls", "grp", "pick", "actual", "exp_slot"]]
    for part in (sons_part(d, players), mlb_part(d, players), arm_part(d)):  # G untestable, see docstring
        out = out.merge(part.drop_duplicates("pfr_id"), on="pfr_id", how="left")
    out.to_parquet(OUT / "old_class_leads.parquet")


if __name__ == "__main__":
    main()
