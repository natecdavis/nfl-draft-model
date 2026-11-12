"""Injury and medical history: do public injury proxies predict NFL value beyond draft slot / the consensus board,
and do teams price them (draft players later than the board)?

No medical grades are public. Proxies (pre-specified before running, one primary each):
  A. Final-season snap drop (PFF college snaps, classes 2015+): final college season's snaps / his best earlier
     season (earlier seasons with 300+ snaps). Primary flag: ratio < 0.6. Class 2021 players with 0 final-season
     snaps (2020 opt-outs) are excluded, not injuries.
  B. Games missed, last two college seasons (college game logs; positions that record stats, so not OL; classes
     2006+): 1 - his games / his team's games, in seasons he played at all. Primary flag: >= 25% missed.
  C. No combine testing (classes 2017+; earlier files barely record non-testers): invited, measured, but no forty,
     vertical, broad, bench, cone or shuttle. Mixes medical with top prospects saving drills for pro day.
  D. Injury concern in Brugler's Beast weaknesses section (2019-23 profiles): injury / surgery / medical /
     durability terms (not "broken tackles"). Secondary: serious (surgery, tear, ACL, Achilles, fracture,
     season-ending, medicals a major/serious concern).
Hypotheses: teams draft flagged players later than the board (pricing < 1). Value vs slot / consensus: direction
open (sports-medicine studies find injured players have shorter careers AND go later; whether slot prices it is
the question). Four primary flags x two yardsticks: p < 0.006 survives Bonferroni; p < 0.05 alone is a lead.
Extra: among fallers (drafted 1+ round after board rank), flagged vs not, vs slot: are medical fallers worse or
better than the app's fall shading assumes?

Yardsticks as in scripts/lit_bias_test.py: value vs slot for drafted players 2016-22, vs consensus for board players
2018-22 (multisport_test.expectations: out of sample, 4 seasons), Poisson with class fixed effects (x < 1 = flagged
players deliver less than expected). Pricing: drafted board players 2014-25, log((rank+10)/(pick+10)), position FE.

Run:  python scripts/injury_test.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm  # noqa: E402
from beast_test import match  # noqa: E402
from multisport_test import SLOT_CLASSES, compare, expectations  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from size_bias_test import reach  # noqa: E402

CF = Path.home() / "cfdarko/data"
OFF = {"QB", "RB", "WR", "TE", "OL"}
DRILLS = ["forty", "vertical", "broad_jump", "bench", "cone", "shuttle"]
INJ = re.compile(r"torn|tore |ACL|MCL|PCL|Achilles|surger|fractur|broke (?:his|a)|broken (?:left |right )?(?:arm|leg|foot|hand|"
                 r"wrist|ankle|collarbone|fibula|tibia|bone|finger|thumb|jaw)|dislocat|labrum|concussion|medical|injur|"
                 r"durab|sidelined|season-ending|missed (?:\w+ ){0,3}(?:games?|season|time|contests?)", re.I)
SERIOUS = re.compile(r"torn|tore |ACL|Achilles|surger|fractur|labrum|season-ending|medicals? (?:are|is|will be) "
                     r"(?:a )?(?:major|serious|big)|(?:major|serious) (?:injury|medical)", re.I)


def snap_drop(u):
    p = pd.read_parquet(CF / "processed/pff_season_grades.parquet",
                        columns=["player_id", "season_year", "pff_snap_counts_offense", "pff_snap_counts_defense"])
    p = p.dropna(subset=["player_id"]).assign(player_id=lambda d: d.player_id.astype(str))
    p = p.merge(u[["player_id", "cls", "grp"]], on="player_id")
    p["snaps"] = np.where(p.grp.isin(OFF), p.pff_snap_counts_offense, p.pff_snap_counts_defense)
    p = p.groupby(["player_id", "season_year"], as_index=False).agg(snaps=("snaps", "sum"), cls=("cls", "first"))
    last = p[p.season_year == p.cls - 1].set_index("player_id").snaps.rename("final_snaps")
    prior = p[(p.season_year < p.cls - 1) & (p.snaps >= 300)].groupby("player_id").snaps.max().rename("best_prior")
    out = pd.concat([last, prior], axis=1)
    out["final_snaps"] = out.final_snaps.fillna(0.0)  # in PFF earlier but no final season = didn't play
    out = out[out.best_prior.notna()]
    out["snap_ratio"] = (out.final_snaps / out.best_prior).clip(upper=1.0)
    return out.reset_index()


def games_missed(u):
    g = pd.read_parquet(CF / "raw/player_gamelogs.parquet", columns=["player_id", "school", "season_year", "game_id"])
    g["player_id"] = g.player_id.astype(str)
    team = g.groupby(["school", "season_year"]).game_id.nunique().rename("team_games")
    me = g[g.player_id.isin(set(u.player_id))].groupby(["player_id", "school", "season_year"]).game_id.nunique() \
        .rename("games").reset_index().merge(team.reset_index(), on=["school", "season_year"])
    me = me.merge(u[["player_id", "cls"]], on="player_id")
    me = me[me.season_year.between(me.cls - 2, me.cls - 1)]
    s = me.groupby("player_id")[["games", "team_games"]].sum()
    s["missed_share"] = (1 - s.games / s.team_games).clip(lower=0)
    return s.reset_index()[["player_id", "missed_share", "games", "team_games"]]


def no_test(u):
    c = pd.read_parquet(RAW / "combine.parquet")
    c = c[(c.season >= 2017) & c.wt.notna()]
    c["none"] = c[DRILLS].isna().all(axis=1)
    by_pfr = dict(zip(c.pfr_id.dropna(), c.loc[c.pfr_id.notna(), "none"]))
    from public_board import _norm
    by_name = {(s, _norm(n)): v for s, n, v in zip(c.season, c.player_name, c.none)}
    out = []
    for pid, cls, pfr, nm in zip(u.player_id, u.cls, u.pfr_id, u.player_name):
        if cls < 2017:
            continue
        v = by_pfr.get(pfr) if isinstance(pfr, str) else None
        if v is None:
            v = by_name.get((cls, _norm(nm)))
        if v is not None:
            out.append((pid, float(v)))
    return pd.DataFrame(out, columns=["player_id", "no_test"]).drop_duplicates("player_id")


def beast_flags(uni):
    b = pd.read_parquet(OUT / "beast_profiles.parquet")
    b = b[b.weaknesses.fillna("").str.len() > 0].copy()
    b["beast_inj"] = b.weaknesses.map(lambda s: bool(INJ.search(s))).astype(float)
    b["beast_serious"] = b.weaknesses.map(lambda s: bool(SERIOUS.search(s))).astype(float)
    m = match(b, uni.assign(player_id=uni.player_id.astype(str)))
    m = m.dropna(subset=["player_id"])
    return m[["player_id", "beast_inj", "beast_serious"]].drop_duplicates("player_id")


def load():
    uni = pm.load()
    u = uni[["player_id", "player_name", "cls", "grp", "hasan_rank", "pick", "drafted", "on_board", "pfr_id"]].copy()
    u["player_id"] = u.player_id.astype(str)
    sd = snap_drop(u)
    gm = games_missed(u[~u.grp.isin(["OL", "K", "P", "OTHER"])])
    nt = no_test(u)
    bf = beast_flags(uni)
    for x in (sd, gm, nt, bf):
        u = u.merge(x, on="player_id", how="left")
    opt_out = (u.cls == 2021) & (u.final_snaps == 0)
    u.loc[opt_out, ["snap_ratio", "final_snaps"]] = np.nan
    u["snap_drop"] = np.where(u.snap_ratio.notna(), u.snap_ratio < 0.6, np.nan)
    u["missed_25"] = np.where(u.missed_share.notna(), u.missed_share >= 0.25, np.nan)
    print(f"Coverage among drafted players: snap ratio {u[u.drafted.astype(bool)].snap_ratio.notna().mean():.0%} "
          f"(2015+), games {u[u.drafted.astype(bool)].missed_share.notna().mean():.0%}, combine 2017+ "
          f"{u[u.drafted.astype(bool) & (u.cls >= 2017)].no_test.notna().mean():.0%}, Beast {u.beast_inj.notna().sum()} profiles; "
          f"2020 opt-outs excluded: {int(opt_out.sum())}")
    return uni, u


FLAGS = [("snap_drop", "A final-season snaps < 60% of best"),
         ("missed_25", "B missed >= 25% of games, last 2 yrs"),
         ("no_test", "C no combine testing (2017+)"),
         ("beast_inj", "D Beast: injury concern in weaknesses"),
         ("beast_serious", "D2 Beast: serious injury (secondary)")]


def main():
    uni, u = load()
    d_ = u[u.drafted.astype(bool)]
    print("\nFlag rates among drafted players, by round:")
    rows = []
    for x, lab in FLAGS:
        s = d_[d_[x].notna()]
        rows.append({"flag": lab, "n": len(s), **{f"R{r}": s[s.pick.between(32 * (r - 1) + 1, 32 * r)][x].mean()
                                                  for r in range(1, 8)}})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nPART 1: PRICING (drafted board players; > 1 = taken earlier than board rank, < 1 = later)")
    for x, lab in FLAGS:
        reach(u[u[x].notna()].assign(**{x: lambda d, x=x: d[x].astype(float)}), x, lab)

    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R[["player_id", "cls", "actual", "exp_slot", "exp_cons"]].merge(
        u.drop(columns=["pfr_id"]), on=["player_id", "cls"], how="left")
    dr = d[d.drafted.astype(bool) & d.exp_slot.notna()]
    bd = d[(d.on_board == 1) & d.exp_cons.notna()]
    dr = dr.assign(fell=lambda x: np.log((x.pick + 10) / (x.hasan_rank + 10)) > np.log(2))
    for e, s, nm in [("exp_slot", dr, "DRAFTED vs SLOT 2016-22"), ("exp_cons", bd, "BOARD vs CONSENSUS 2018-22"),
                     ("exp_slot", dr[dr.fell & dr.hasan_rank.notna()], "FALLERS (pick > ~2x board rank) vs SLOT")]:
        print(f"\nPART 2: VALUE, {nm}  (< 1 = flagged players deliver less than expected)")
        for x, lab in FLAGS:
            t = s[s[x].notna()].copy()
            t[x] = t[x].astype(bool)
            fe = pd.get_dummies(t.cls, prefix="c", drop_first=True, dtype=float)
            compare(pd.concat([t, fe], axis=1), x, e, covs=tuple(fe.columns), label=lab, min_n=8)
    d.to_parquet(OUT / "injury_test.parquet")
    u.to_parquet(OUT / "injury_flags.parquet")


if __name__ == "__main__":
    main()
