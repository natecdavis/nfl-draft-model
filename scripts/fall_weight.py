"""For players who fell below consensus: is their career predicted by board rank or by
the pick where they went? Fit outcome ~ a*log(pick+10) + b*log(rank+10); the share of
weight on the pick, w = a/(a+b), sets how far the app shades a falling player's talent
toward his actual slot (w=0: board rank is right; w=1: the pick is right)."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from public_board import _norm

h = pd.read_csv("data/raw/hasan.csv", encoding="utf-8-sig").assign(key=lambda d: d.name.map(_norm))
dp = pd.read_parquet("data/raw/draft_picks.parquet").assign(key=lambda d: d.pfr_player_name.map(_norm))
m = h.merge(dp[["season", "key", "pick"]], left_on=["year", "key"], right_on=["season", "key"])
out = pd.read_parquet("data/derived/pick_outcomes.parquet")
o = out.merge(m[["year", "pick", "rank"]], left_on=["draft_year", "pick"], right_on=["year", "pick"])
o = o[o.grp.notna()].assign(lpick=lambda d: np.log(d.pick + 10), lrank=lambda d: np.log(d["rank"] + 10),
                            star=lambda d: (d.score == 2).astype(int), plus=lambda d: (d.score >= 1).astype(int))
fell = o[o.lpick > o.lrank]
print(f"Players on the board who fell (2014-21): {len(fell)}")
rng = np.random.default_rng(0)
for y in ["star", "plus"]:
    r = smf.logit(f"{y} ~ lpick + lrank + C(grp)", fell).fit(disp=False)
    a, b = -r.params["lpick"], -r.params["lrank"]
    boots = []
    for _ in range(300):
        s = fell.iloc[rng.integers(0, len(fell), len(fell))]
        rb = smf.logit(f"{y} ~ lpick + lrank + C(grp)", s).fit(disp=False)
        ab, bb = -rb.params["lpick"], -rb.params["lrank"]
        boots.append(ab / (ab + bb))
    print(f"{y}: weight on pick w = {a / (a + b):.2f} (95% {np.percentile(boots, 2.5):.2f} to {np.percentile(boots, 97.5):.2f})")
