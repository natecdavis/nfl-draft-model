"""Audit for racial mispricing in the draft, using name- and place-based race probabilities (BIFSG).

Race is not in any of our data and is NOT a model input. Each board / drafted player gets a probability of each
race from public aggregates only, the method used in fair-lending audits (Elliott et al. 2009; Voicu 2018):
  P(race | surname, first name, home county)  proportional to  P(race | surname) x P(first | race) x P(county | race)
  surname     Census 2010 surname table (data/raw/race_proxy/Names_2010Census.csv; suppressed cells share the rest)
  first name  Tzioumis 2018 (Harvard Dataverse doi:10.7910/DVN/TYJKEZ; 4,251 names, so many players' names are
              absent and contribute nothing)
  county      Census county population estimates, April 2010 (cc-est2019-alldata.csv), from the CFBD recruiting
              hometown county code; missing for players without a recruiting record (then no geography term)
No per-player labels are reported; results are group-level only, and the derived file stays in data/ (not in git).

Caveats: names separate Black and white Americans poorly, and the general-population prior understates the share
of Black players in football, so probabilities are noisy and compressed toward the population mix. Regression on
the probability then gives the right sign but a scaled, less precise slope.

Tests (specified before looking), by position group (QB; C / LB / S; WR / CB / RB; all others), x = P(Black):
  1. pricing: drafted board players 2014-24, log((board rank + 10) / (pick + 10)), position fixed effects
  2. value: drafted 2016-22 vs draft slot, board 2018-22 vs consensus (out of sample), class fixed effects
Coefficients compare a player certain to be Black with one certain not to be.
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
from multisport_test import SLOT_CLASSES, expectations
from outlook import OUT

RP = "data/raw/race_proxy/"
RACES = ["white", "black", "hispanic", "api", "other"]
SUFFIX = {"JR", "SR", "II", "III", "IV", "V"}
GROUPS = {"QB": {"QB"}, "C / LB / S": {"C", "LB", "S"}, "WR / CB / RB": {"WR", "CB", "RB"}}


def surname_table():
    s = pd.read_csv(RP + "Names_2010Census.csv", na_values=["(S)"], keep_default_na=False)
    cols = ["pctwhite", "pctblack", "pcthispanic", "pctapi", "pctaian", "pct2prace"]
    s[cols] = s[cols].astype(float)
    miss = s[cols].isna().sum(axis=1)
    rest = (100 - s[cols].sum(axis=1)).clip(lower=0) / miss.where(miss > 0, 1)
    for c in cols:
        s[c] = s[c].fillna(rest)
    p = pd.DataFrame({"white": s.pctwhite, "black": s.pctblack, "hispanic": s.pcthispanic, "api": s.pctapi,
                      "other": s.pctaian + s.pct2prace}) / 100
    p.index = s.name.astype(str)
    return p.div(p.sum(axis=1), axis=0)


def first_table():
    f = pd.read_csv(RP + "firstnames.csv")
    pr = pd.DataFrame({"white": f.pctwhite, "black": f.pctblack, "hispanic": f.pcthispanic, "api": f.pctapi,
                       "other": f.pctaian + f.pct2prace}) / 100
    cnt = pr.mul(f.obs, axis=0)
    lik = cnt / cnt.sum()  # P(first | race) within the list
    lik.index = f.firstname.str.upper()
    return lik


def county_table():
    c = pd.read_csv(RP + "cc-est2019-alldata.csv", encoding="latin-1",
                    usecols=["STATE", "COUNTY", "YEAR", "AGEGRP", "NHWA_MALE", "NHWA_FEMALE", "NHBA_MALE", "NHBA_FEMALE",
                             "H_MALE", "H_FEMALE", "NHAA_MALE", "NHAA_FEMALE", "NHNA_MALE", "NHNA_FEMALE",
                             "NHIA_MALE", "NHIA_FEMALE", "NHTOM_MALE", "NHTOM_FEMALE"])
    c = c[(c.YEAR == 1) & (c.AGEGRP == 0)]
    s = lambda *k: sum(c[f"{x}_MALE"] + c[f"{x}_FEMALE"] for x in k)
    pop = pd.DataFrame({"white": s("NHWA"), "black": s("NHBA"), "hispanic": s("H"), "api": s("NHAA", "NHNA"),
                        "other": s("NHIA", "NHTOM")})
    pop.index = (c.STATE * 1000 + c.COUNTY).astype(int)
    return pop / pop.sum()  # P(county | race)


def parts(name):
    w = [x for x in str(name).upper().replace(".", "").replace("'", "").split() if x not in SUFFIX]
    if len(w) < 2:
        return None, None
    return w[0], w[-1].split("-")[-1]


def recruit_counties():
    rows = []
    for f in sorted(glob.glob("data/raw/cfbd_recruits/*.json")):
        for r in json.load(open(f)):
            fips = (r.get("hometownInfo") or {}).get("fipsCode")
            if r.get("athleteId") and fips:
                rows.append({"pid": str(r["athleteId"]), "fips": int(fips)})
    return pd.DataFrame(rows).drop_duplicates("pid", keep="last")


def bifsg(u):
    S, F, G = surname_table(), first_table(), county_table()
    prior = S.loc["ALL OTHER NAMES"] if "ALL OTHER NAMES" in S.index else S.mean()
    out = []
    for first, last, fips in zip(u.first, u.last, u.fips):
        p = (S.loc[last] if last in S.index else prior).to_numpy(copy=True)
        if first in F.index:
            p *= F.loc[first].to_numpy()
        if pd.notna(fips) and int(fips) in G.index:
            p *= G.loc[int(fips)].to_numpy()
        out.append(p / p.sum())
    return pd.DataFrame(out, columns=[f"p_{r}" for r in RACES], index=u.index)


def reach(b, label):
    b = b.dropna(subset=["hasan_rank", "pick", "p_black"])
    if len(b) < 30:
        return
    y = np.log((b.hasan_rank + 10) / (b.pick + 10))
    fe = pd.get_dummies(b.pos2, prefix="p", drop_first=True, dtype=float)
    m = sm.OLS(y, sm.add_constant(pd.concat([b[["p_black"]], fe], axis=1))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc["p_black"])
    print(f"    {label:34s} n={len(b):4d}  taken x{np.exp(m.params.p_black):.2f} earlier than board "
          f"(95% {lo:.2f}-{hi:.2f}, p {m.pvalues.p_black:.2f})")


def value(s, e, label):
    s = s.dropna(subset=[e, "p_black"])
    if len(s) < 30:
        return
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    m = sm.GLM(s.actual, sm.add_constant(pd.concat([s[["p_black"]], fe], axis=1)), family=sm.families.Poisson(),
               offset=np.log(s[e].clip(lower=1e-4))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc["p_black"])
    print(f"    {label:34s} n={len(s):4d}  value x{np.exp(m.params.p_black):.2f} (95% {lo:.2f}-{hi:.2f}, p {m.pvalues.p_black:.2f})")


def main():
    uni = pm.load()
    u = uni[(uni.on_board == 1) | uni.drafted.astype(bool)][
        ["player_id", "player_name", "cls", "grp", "pfr_id", "hasan_rank", "pick", "drafted", "on_board"]].copy()
    u["player_id"] = u.player_id.astype(str)
    u[["first", "last"]] = u.player_name.map(parts).apply(pd.Series)
    u = u.merge(recruit_counties(), left_on="player_id", right_on="pid", how="left").drop(columns="pid")
    k = pd.read_parquet("data/raw/combine.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last")
    u = u.merge(k[["pfr_id", "pos"]], on="pfr_id", how="left")
    u["pos2"] = np.select([u.pos.isin(["S", "SAF"]), u.pos.eq("CB"), u.pos.eq("C")], ["S", "CB", "C"], u.grp)
    u = pd.concat([u, bifsg(u)], axis=1)
    S, F = surname_table(), first_table()
    print(f"board / drafted players: {len(u)}; surname found {u['last'].isin(S.index).mean():.0%}, "
          f"first name found {u['first'].isin(F.index).mean():.0%}, home county {u.fips.notna().mean():.0%}")
    print("mean P(Black) by position (sanity check: K / P lowest, CB / WR / RB highest expected):")
    print("   " + "  ".join(f"{p} {v:.2f}" for p, v in u.groupby("pos2").p_black.mean().sort_values().items()))
    print(f"   share of players with P(Black) > 0.5: {(u.p_black > 0.5).mean():.0%}; < 0.2: {(u.p_black < 0.2).mean():.0%}")

    sel = lambda d, s: d[d.pos2.isin(s)] if s else d[~d.pos2.isin(set().union(*GROUPS.values()))]
    groups = list(GROUPS.items()) + [("all others", None)]
    print("\n1. PRICING: drafted board players 2014-24 (> 1 = taken earlier than board rank)")
    b = u[u.drafted.astype(bool) & u.hasan_rank.notna()]
    reach(b, "all positions")
    for g, s in groups:
        reach(sel(b, s), g)

    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R[["player_id", "cls", "actual", "exp_slot", "exp_cons"]].merge(u, on=["player_id", "cls"], how="inner")
    for e, s, nm in [("exp_slot", d[d.drafted.astype(bool) & d.exp_slot.notna()], "drafted vs slot 2016-22"),
                     ("exp_cons", d[(d.on_board == 1) & d.exp_cons.notna()], "board vs consensus 2018-22")]:
        print(f"\n2. VALUE, {nm} (> 1 = Black players beat expectations)")
        value(s, e, "all positions")
        for g, gs in groups:
            value(sel(s, gs), e, g)
    u.drop(columns=["first", "last"]).to_parquet(OUT / "race_audit.parquet")


if __name__ == "__main__":
    main()
