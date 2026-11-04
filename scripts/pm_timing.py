"""1. Does anything beyond rank + position predict WHEN value arrives (years 1-2)?
2. Are outcome ranges honest when we sample real careers of similar prospects?"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import prospect_model as pm
df = pd.read_parquet("data/derived/pm_universe.parquet")
pm.PARAMS.update(min_child_samples=150, n_estimators=300, colsample_bytree=0.5, num_leaves=7, reg_lambda=5.0)
pm.FEATURES["age_pre"] = pm.FEATURES["consensus_pre"] + ["age_at_draft"]
pm.FEATURES["polish_pre"] = pm.FEATURES["age_pre"] + ["pff_grades_offense_last", "pff_grades_defense_last", "career_touches"]
V = ("consensus_pre", "age_pre", "polish_pre", "full_pre")
r = pm.rolling(df, versions=V)
d = r[r.drafted].assign(y12=lambda x: x.y1 + x.y2, y14=lambda x: x.y1 + x.y2 + x.y3 + x.y4)
d = d[d.y14 > 0]  # among players who contributed at all: how much came early?
d["early_share"] = d.y12 / d.y14
print("1. Drafted players who contributed (2019-22): predicting the EARLY SHARE of their 4-yr value")
for v in V:
    p12 = d[f"{v}_p1"] + d[f"{v}_p2"]; p14 = p12 + d[f"{v}_p3"] + d[f"{v}_p4"]
    ps = p12 / p14
    print(f"   {v:14s} corr(predicted early share, actual) = {ps.corr(d.early_share):.3f} | "
          f"rank corr of early value (yrs 1-2) = {p12.corr(d.y12, method='spearman'):.3f}")

# 2. Outcome ranges from comps: for each test player, 60 nearest training players (same
# position group, closest predicted 4-yr value on log scale); their actual careers = range.
print("\n2. Honest ranges? (consensus_pre, drafted test players; target 10% below / 10% above)")
rows = []
for c in (2019, 2020, 2021, 2022):
    tr = df[(df.cls < c) & (df.cls + 3 <= c - 1)].copy()  # training players with all 4 seasons seen
    te = r[(r.cls == c) & r.drafted].copy()
    trp = pm.predict_seasons(df[df.cls < c], tr, "consensus_pre", cutoff_cls=c).sum(axis=1)
    tr["p14"], tr["y14"] = trp, tr[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    te["p14"] = te[[f"consensus_pre_p{t}" for t in pm.SEASONS]].sum(axis=1)
    te["y14"] = te[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    for _, x in te.iterrows():
        pool = tr[(tr.grp == x.grp) & tr.drafted]
        k = int(np.clip(len(pool) // 6, 15, 60))
        near = pool.iloc[np.argsort(np.abs(np.log(pool.p14 + 1e-3) - np.log(x.p14 + 1e-3)).to_numpy())[:k]]
        lo, hi = np.percentile(near.y14, [10, 90])
        rows.append({"below": x.y14 < lo, "above": x.y14 > hi, "p14": x.p14,
                     "star_thr": np.percentile(near.y14, 90)})
cal = pd.DataFrame(rows)
print(f"   actual below the 10th percentile: {cal.below.mean():.0%} | above the 90th: {cal.above.mean():.0%} (n={len(cal)})")
