import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import prospect_model as pm
df = pm.load()
pd.set_option("display.width", 200)
V = ("consensus_pre", "full_pre")
REG = dict(min_child_samples=150, n_estimators=300, colsample_bytree=0.5, num_leaves=7, reg_lambda=5.0)
for name, over in [("default", None), ("more regularized", REG)]:
    ch = pm.choice_test(df, over, versions=V)
    s = pm.summarize_choice(ch, V)
    print(f"\n{name}: avg actual 4-season value of the player chosen at each pick (% of cap), drafts 2019-2022")
    print(s.round(2).to_string())
    d = (ch.full_pre_value - ch.consensus_pre_value).to_numpy()
    rng = np.random.default_rng(0); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"   full minus consensus per pick: {d.mean()*100:+.2f} (95% {np.percentile(b,2.5)*100:+.2f} to {np.percentile(b,97.5)*100:+.2f})")
    for r, g in ch.assign(rnd=pd.cut(ch.pick, [0, 32, 140, 262], labels=["R1", "R2-4", "R5-7"])).groupby("rnd", observed=True):
        d = (g.full_pre_value - g.consensus_pre_value).to_numpy()
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"     {r}: {d.mean()*100:+.2f} (95% {np.percentile(b,2.5)*100:+.2f} to {np.percentile(b,97.5)*100:+.2f})")
    print(f"   off-board choices: consensus {1-ch.consensus_pre_on_board.mean():.0%}, full {1-ch.full_pre_on_board.mean():.0%}")
    ch.to_parquet(f"data/derived/pm_choice_pre_fixed_{name.replace(' ', '_')}.parquet")
