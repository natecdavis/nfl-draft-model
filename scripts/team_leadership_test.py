"""Do teams with more veterans, or more former college captains, punch above their talent over a season?

Leadership would build through camp, practice and games, so a week-to-week absence (scripts/leader_absence_test.py)
may not show it. Here the unit is the team-season. Specified before looking (Nov 22 2026).
  Talent: snap-weighted pre-season talent (darko_pff talent_pre: past seasons only, age drift included) for each of
  the 9 position groups, entered separately. Outcome: opponent-adjusted points margin per game (synergy_test.ratings,
  'net'). 'Punching above weight' = the veteran / captain term given talent.
  VETERANS (2013-25, 416 team-seasons): share of the week-1 active roster in its 6th+ NFL season (players.rookie_season).
    PRIMARY: net ~ vet share (per 10 points) + 9 talent groups + season effects, clustered by team.
    SECONDARY: team fixed effects; snap-weighted mean NFL seasons; share 8+; with snap-weighted mean age (veteran vs
    old); outcome = season mean of actual margin minus closing spread (does the market miss it); out-of-sample
    (fit through 2019, predict 2020-25).
  CAPTAINS (2019-25): Brugler 'The Beast' profiles 2019-23 that mention 'captain' (~28%; the stats line says
    'team captain'). Team-season: captains on the week-1 active roster among its profiled players (classes 2019-23).
    PRIMARY: net ~ captains + profiled players on roster + talent + season effects (only classes 2019-23 can be
    flagged, so the profiled count is the exposure). SECONDARY: captain share; player level own value vs draft slot
    (multisport_test.compare) for drafted profiled players 2019-22.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from synergy_test import ratings

RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
RTEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"}
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


def talent():
    s = pd.read_parquet(OUT / "darko_pff.parquet")
    s["w"] = s.talent_pre * s.snaps
    t = s.groupby(["season", "tm", "g"])[["w", "snaps"]].sum()
    t = (t.w / t.snaps).unstack("g")[GROUPS]
    t.columns = [f"t_{g}" for g in GROUPS]
    t = t.reset_index().rename(columns={"tm": "team"})
    a = s.assign(wa=s.age * s.snaps).groupby(["season", "tm"])[["wa", "snaps"]].sum()
    t = t.merge((a.wa / a.snaps).rename("age_w").reset_index().rename(columns={"tm": "team"}), on=["season", "team"])
    ent = s.dropna(subset=["entry"]).assign(ex=lambda x: (x.season - x.entry + 1) * x.snaps)
    e = ent.groupby(["season", "tm"])[["ex", "snaps"]].sum()
    return t.merge((e.ex / e.snaps).rename("exp_w").reset_index().rename(columns={"tm": "team"}), on=["season", "team"])


def week1():
    r = pd.read_parquet(RAW / "rosters_weekly.parquet")
    r = r[(r.status == "ACT")]
    r = r[r.week == r.groupby("season").week.transform("min")]
    r["team"] = r.team.replace(RTEAM)
    p = pd.read_parquet(RAW / "players.parquet")[["gsis_id", "pfr_id", "rookie_season", "draft_year"]]
    r = r.merge(p, on="gsis_id", how="left", suffixes=("", "_p"))
    r["yrs"] = r.season - r.rookie_season.fillna(r.draft_year) + 1
    return r


def captains():
    import prospect_model as pm
    from beast_test import match
    b = pd.read_parquet(OUT / "beast_profiles.parquet")
    txt = (b.background.fillna("") + " " + b.strengths.fillna("") + " " + b.summary.fillna("")).str.lower()
    b["captain"] = txt.str.contains("captain")
    b["log_bg"] = np.log(b.background.fillna("").str.len().clip(lower=100))
    m = match(b, pm.load())
    m = m[m.player_id.notna()].drop_duplicates("player_id")
    return m


def run(lab, f, D, term, fe=False):
    if fe:
        f += " + C(team)"
    m = smf.ols(f, D).fit(cov_type="cluster", cov_kwds={"groups": D.team})
    ci = m.conf_int().loc[term]
    print(f"   {lab:46s} n={len(D):4d} | {term} {m.params[term]:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}, p {m.pvalues[term]:.2f})")
    return m


def main():
    T = talent()
    R = ratings()
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.result.notna() & s.spread_line.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(RTEAM)
    ms = pd.concat([s.assign(team=s.home_team, ms=s.result - s.spread_line),
                    s.assign(team=s.away_team, ms=-(s.result - s.spread_line))]).groupby(["season", "team"]).ms.mean()
    r = week1()
    v = r.groupby(["season", "team"]).agg(n=("gsis_id", "size"), vet6=("yrs", lambda x: (x >= 6).mean()),
                                         vet8=("yrs", lambda x: (x >= 8).mean())).reset_index()
    D = T.merge(R, on=["season", "team"]).merge(v, on=["season", "team"]).merge(ms.reset_index(), on=["season", "team"])
    D = D[D.season.between(2013, 2025)].copy()
    D[["vet6", "vet8"]] *= 10
    tal = " + ".join(f"t_{g}" for g in GROUPS)
    print(f"VETERANS: team-seasons {len(D)}; week-1 roster {D.n.mean():.0f}; 6th+ season share mean "
          f"{D.vet6.mean() * 10:.1f}% (sd {D.vet6.std() * 10:.1f}); net sd {D.net.std():.2f} pts/game")
    m0 = smf.ols(f"net ~ {tal} + C(season)", D).fit()
    print(f"   talent-only R2 {m0.rsquared:.2f}; corr(vet share, talent fit) {np.corrcoef(D.vet6, m0.fittedvalues)[0, 1]:+.2f}")
    print("   PRIMARY (pts/game per +10 points of roster share in 6th+ season)")
    run("vet share 6+", f"net ~ vet6 + {tal} + C(season)", D, "vet6")
    print("   SECONDARY")
    run("team fixed effects", f"net ~ vet6 + {tal} + C(season)", D, "vet6", fe=True)
    run("snap-weighted mean NFL seasons (per season)", f"net ~ exp_w + {tal} + C(season)", D, "exp_w")
    run("vet share 8+", f"net ~ vet8 + {tal} + C(season)", D, "vet8")
    run("vet share 6+, with snap-weighted mean age", f"net ~ vet6 + age_w + {tal} + C(season)", D, "vet6")
    run("margin minus spread (market misses?)", f"ms ~ vet6 + {tal} + C(season)", D, "vet6")
    run("no talent control (raw)", "net ~ vet6 + C(season)", D, "vet6")
    tr, te = D[D.season <= 2019], D[D.season >= 2020]
    rm = {k: np.sqrt(((te.net - smf.ols(f, tr).fit().predict(te)) ** 2).mean())
          for k, f in [("talent", f"net ~ {tal}"), ("+vet", f"net ~ vet6 + {tal}")]}
    print(f"   out of sample 2020-25 RMSE: talent {rm['talent']:.3f}, + vet share {rm['+vet']:.3f}")

    # captains
    cap = captains()
    print(f"\nCAPTAINS: Brugler profiles matched {len(cap)}; captain {cap.captain.mean():.0%}")
    cp = cap[["pfr_id", "captain"]].dropna(subset=["pfr_id"])
    rc = r[r.season >= 2019].merge(cp, on="pfr_id", how="inner")
    c = rc.groupby(["season", "team"]).agg(prof=("captain", "size"), capt=("captain", "sum")).reset_index()
    C = D.merge(c, on=["season", "team"], how="left").fillna({"prof": 0, "capt": 0})
    C = C[C.season >= 2019].copy()
    C["cshare"] = C.capt / C.prof.clip(lower=1) * 10
    print(f"   team-seasons {len(C)}; profiled on week-1 roster mean {C.prof.mean():.1f}, captains {C.capt.mean():.1f} "
          f"(sd {C.capt.std():.1f})")
    print("   PRIMARY (pts/game per captain on roster, given profiled players)")
    run("captains", f"net ~ capt + prof + {tal} + C(season)", C, "capt")
    print("   SECONDARY")
    run("captain share (per +10 points)", f"net ~ cshare + prof + {tal} + C(season)", C[C.prof > 0], "cshare")
    run("captains, margin minus spread", f"ms ~ capt + prof + {tal} + C(season)", C, "capt")
    run("captains, team fixed effects", f"net ~ capt + prof + {tal} + C(season)", C, "capt", fe=True)
    from multisport_test import compare, expectations
    import prospect_model as pm
    E = expectations(pm.load(), (2019, 2020, 2021, 2022))
    E["player_id"] = E.player_id.astype(str)
    cap["player_id"] = cap.player_id.astype(str)
    P = E.merge(cap[["player_id", "captain", "log_bg"]], on="player_id")
    P = P[P.drafted.astype(bool)]
    P["captain"] = P.captain.astype(bool)
    print("   player level: own 4-season value vs draft slot")
    compare(P, "captain", "exp_slot", covs=("log_bg",), label="captain (drafted 2019-22)")
    compare(P[P.on_board == 1], "captain", "exp_cons", covs=("log_bg",), label="captain, vs consensus board")
    D.to_parquet(OUT / "team_leadership_vets.parquet")
    C.to_parquet(OUT / "team_leadership_captains.parquet")


if __name__ == "__main__":
    main()
