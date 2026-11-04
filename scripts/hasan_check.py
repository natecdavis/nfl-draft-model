"""Arif Hasan consensus boards 2014-2024 (from ~/cfdarko/hasan.csv; data/raw, not committed):
1. how far actual picks land from consensus rank (calibrates the app's availability model)
2. does falling below / rising above consensus predict NFL outcomes beyond the pick?"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from public_board import _norm

h = pd.read_csv("data/raw/hasan.csv", encoding="utf-8-sig")
dp = pd.read_parquet("data/raw/draft_picks.parquet")
dp = dp[dp.season.between(2014, 2024)].assign(key=lambda d: d.pfr_player_name.map(_norm))
h["key"] = h.name.map(_norm)
m = h.merge(dp[["season", "key", "pick"]], left_on=["year", "key"], right_on=["season", "key"], how="left")
print(f"Board players matched to a draft pick: {m.pick.notna().mean():.0%} "
      f"(the rest went undrafted or have name mismatches)")
m["real"] = m.pick.fillna(300)
m["gap"] = np.log(m.real + 10) - np.log(m["rank"] + 10)
m["bucket"] = pd.cut(m["rank"], [0, 10, 32, 64, 100, 150, 200, 320])
t = m.groupby("bucket", observed=True).agg(gap_mean=("gap", "mean"), gap_sd=("gap", "std"),
                                           undrafted=("pick", lambda x: x.isna().mean()), n=("gap", "size"))
print("\n1. Actual pick vs consensus rank (log(pick+10) - log(rank+10)):\n", t.round(2))
d = m[m.pick.notna()]
print(f"   drafted players only: sd {d.gap.std():.2f}; ESPN board gave about 0.25 (top 10) to 0.40")
# app's current spread for comparison
from draftday import spread
print("   app currently assumes sd:", {r: round(float(spread(r)), 2) for r in [5, 20, 50, 100, 200]})

# 2. outcomes (drafts 2014-2021 have tiers)
out = pd.read_parquet("data/derived/pick_outcomes.parquet")
o = out.merge(m[["year", "pick", "rank"]].dropna(), left_on=["draft_year", "pick"], right_on=["year", "pick"], how="left")
o = o[o.draft_year.between(2014, 2021) & o.grp.notna()]
o["on_board"] = o["rank"].notna()
o["lp"] = np.log(o.pick + 10)
o["cgap"] = np.where(o.on_board, o.lp - np.log(o["rank"] + 10), 0)
o["star"] = (o.score == 2).astype(int)
o["plus"] = (o.score >= 1).astype(int)
print(f"\n2. Picks 2014-2021: {len(o)}; on Hasan's board: {o.on_board.mean():.0%}")
mod = smf.logit("star ~ lp + C(grp)", o).fit(disp=False)
o["exp_star"] = mod.predict(o)
for lab, mask in [("fell 1+ round below consensus", o.cgap > 0.5), ("about where ranked", o.on_board & (o.cgap.abs() <= 0.5)),
                  ("taken 1+ round above consensus", o.cgap < -0.5), ("not on the board", ~o.on_board)]:
    x = o[mask]
    print(f"   {lab:32s} n={len(x):4d}  stars {x.star.mean():.3f}  expected from slot {x.exp_star.mean():.3f}")
m2 = smf.logit("star ~ lp + C(grp) + cgap + on_board", o).fit(disp=False)
print("   consensus gap effect on star odds:", round(m2.params["cgap"], 3), "p =", round(m2.pvalues["cgap"], 3))
# rolling test like prospects.py
ll = {"slot": [], "slot+consensus": []}
for yr in range(2016, 2022):
    tr, te = o[o.draft_year < yr], o[o.draft_year == yr]
    for name, f in [("slot", "lp + C(grp)"), ("slot+consensus", "lp + C(grp) + cgap + on_board")]:
        ps = smf.logit(f"star ~ {f}", tr).fit(disp=False).predict(te).clip(1e-4, 1 - 1e-4)
        pp = smf.logit(f"plus ~ {f}", tr).fit(disp=False).predict(te).clip(1e-4, 1 - 1e-4)
        pp = np.maximum(pp, ps)
        pr = np.column_stack([1 - pp, pp - ps, ps]).clip(1e-4, 1)
        ll[name] += list(-np.log(pr[np.arange(len(te)), te.score.to_numpy()]))
print("   rolling test 2016-21 log loss:", {k: round(np.mean(v), 4) for k, v in ll.items()})

# 3. Split the gap: is it all in reaches? (falls and reaches as separate inputs)
o["fell"] = o.cgap.clip(lower=0)
o["reach"] = (-o.cgap).clip(lower=0)
o["onb"] = o.on_board.astype(int)
for y in ["star", "plus"]:
    r = smf.logit(f"{y} ~ lp + C(grp) + fell + reach + onb", o).fit(disp=False)
    print(f"\n3. {y}: " + ", ".join(f"{k} {r.params[k]:+.2f} (p {r.pvalues[k]:.3f})" for k in ["fell", "reach", "onb"]))
ll = {"slot": [], "slot+reach": [], "slot+fell+reach": []}
for yr in range(2016, 2022):
    tr, te = o[o.draft_year < yr], o[o.draft_year == yr]
    for name, f in [("slot", "lp + C(grp)"), ("slot+reach", "lp + C(grp) + reach + onb"),
                    ("slot+fell+reach", "lp + C(grp) + fell + reach + onb")]:
        ps = smf.logit(f"star ~ {f}", tr).fit(disp=False).predict(te).clip(1e-4, 1 - 1e-4)
        pp = np.maximum(smf.logit(f"plus ~ {f}", tr).fit(disp=False).predict(te).clip(1e-4, 1 - 1e-4), ps)
        pr = np.column_stack([1 - pp, pp - ps, ps]).clip(1e-4, 1)
        ll[name] += list(-np.log(pr[np.arange(len(te)), te.score.to_numpy()]))
base = np.array(ll["slot"])
rng = np.random.default_rng(0)
for k, v in ll.items():
    d = base - np.array(v)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"   rolling 2016-21 {k:16s} {np.mean(v):.4f}  gain vs slot {d.mean():+.4f} "
          f"(95% {np.percentile(b, 2.5):+.4f} to {np.percentile(b, 97.5):+.4f})")
