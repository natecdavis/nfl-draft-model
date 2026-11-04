"""Fall weight again, now including players who fell all the way out of the draft
(counted as pick 300; no NFL career = bust). Uses review.board() matching."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from review import board, draft_night

out = pd.read_parquet("data/derived/pick_outcomes.parquet")
tier = {(y, p): s for y, p, s in zip(out.draft_year, out.pick, out.score)}
rows = []
for y in range(2014, 2022):
    picks, _ = draft_night(y)
    b = board(y, picks)
    b = b[b.grp != "SPEC"]
    for r in b.itertuples():
        pk = r.drafted_at if pd.notna(r.drafted_at) else 300
        sc = tier.get((y, int(pk)), 0) if pd.notna(r.drafted_at) else 0
        rows.append({"year": y, "rank": r.rank, "pick": pk, "grp": r.grp, "score": sc, "undrafted": pd.isna(r.drafted_at)})
d = pd.DataFrame(rows).assign(lpick=lambda x: np.log(x.pick + 10), lrank=lambda x: np.log(x["rank"] + 10),
                              star=lambda x: (x.score == 2).astype(int), plus=lambda x: (x.score >= 1).astype(int))
fell = d[d.lpick > d.lrank]
print(f"Board players who fell: {len(fell)}, of whom undrafted: {fell.undrafted.sum()}")
rng = np.random.default_rng(0)
for y in ["star", "plus"]:
    r = smf.logit(f"{y} ~ lpick + lrank + C(grp)", fell).fit(disp=False)
    a, b_ = -r.params["lpick"], -r.params["lrank"]
    bs = []
    for _ in range(300):
        s = fell.iloc[rng.integers(0, len(fell), len(fell))]
        rb = smf.logit(f"{y} ~ lpick + lrank + C(grp)", s).fit(disp=False)
        bs.append(-rb.params["lpick"] / (-rb.params["lpick"] - rb.params["lrank"]))
    print(f"{y}: weight on pick w = {a / (a + b_):.2f} (95% {np.percentile(bs, 2.5):.2f} to {np.percentile(bs, 97.5):.2f})")
