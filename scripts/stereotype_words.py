"""Coded scouting language, by position (Brugler's guides, 2019-2022).

Question: at positions with strong stereotypes, do 'smart / gritty' or 'athletic / raw' descriptions
predict outcomes beyond consensus rank (board players) or draft slot (drafted players)? If the
words carry a bias in how players are rated, the players described that way should miss their
expectation in a consistent direction. Race is not used; this tests the language itself.

Word rates are per 1,000 words of STRENGTHS + WEAKNESSES + SUMMARY; each group is compared as
'above vs at/below the median for his position'. Expectations from scripts/multisport_test.py
(out of sample, each class predicted by models of earlier classes).
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
from outlook import OUT

WORDS = {
    "cerebral": r"football iq|instinct|cerebral|intelligen|\bsmart\b|savvy|heady|student of the game|film|process(?:or|es|ing)|"
                r"anticipat|diagnos|recogni[sz]",
    "gritty": r"gritty|grit\b|scrappy|overachiev|blue[- ]collar|lunch[- ]pail|hustle|effort|motor|toughness|tough\b|competitive toughness",
    "athletic": r"athletic|athlete|freak|explosive|twitch|natural ability|gifted|rare\b|elite (?:speed|athlet|burst)",
    "raw": r"\braw\b|unrefined|project\b|inconsistent|undisciplined|freelanc|lacks? (?:awareness|discipline|instincts|feel)",
}
GROUPS = {"QB": {"QB"}, "C / LB / S": {"C", "LB", "S"}, "WR / CB / RB": {"WR", "CB", "RB"}, "all others": None}


def main():
    t = pd.read_parquet(OUT / "multisport_text.parquet")
    b = pd.read_parquet(OUT / "beast_profiles.parquet")[["year", "name", "school", "section", "strengths", "weaknesses", "summary"]]
    d = t.merge(b, on=["year", "name", "school"], how="left", suffixes=("", "_b"))
    sec = d.section if "section" in d else d.section_b
    d["sec"] = sec
    txt = (d.strengths.fillna("") + " " + d.weaknesses.fillna("") + " " + d.summary.fillna("")).str.lower()
    n = txt.str.split().str.len().clip(lower=50)
    for k, v in WORDS.items():
        d[k] = txt.str.count(v) / n * 1000
    named = set().union(*[s for s in GROUPS.values() if s])
    print(f"profiles with text: {int((n > 50).sum())} of {len(d)}; guide positions: {sorted(d.sec.dropna().unique())}")
    for gname, secs in GROUPS.items():
        g = d[d.sec.isin(secs)] if secs else d[~d.sec.isin(named)]
        print(f"\n{gname}  (drafted {int(g.drafted.sum())}, board {int((g.on_board == 1).sum())})")
        for k in WORDS:
            hi = g[k] > g.groupby("sec")[k].transform("median")
            for lab, sub, e in [("drafted vs slot", g[g.drafted], "exp_slot"), ("board vs consensus", g[g.on_board == 1], "exp_cons")]:
                sub = sub.dropna(subset=[e]); h = hi.loc[sub.index].astype(float)
                if h.sum() < 8 or (1 - h).sum() < 8:
                    continue
                m = sm.GLM(sub.actual, sm.add_constant(pd.DataFrame({"hi": h, "log_bg": sub.log_bg})),
                           family=sm.families.Poisson(), offset=np.log(sub[e].clip(lower=1e-4))).fit(cov_type="HC1")
                beat = sub.actual > sub[e]
                print(f"   more '{k}' words  {lab:19s}  n {int(h.sum()):3d}/{int((1 - h).sum()):3d}  value x{np.exp(m.params.hi):.2f} "
                      f"(p {m.pvalues.hi:.2f})  beat {beat[h == 1].mean():.0%} vs {beat[h == 0].mean():.0%}")


if __name__ == "__main__":
    main()
