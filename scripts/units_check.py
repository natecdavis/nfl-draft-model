"""Are cfdarko's season values (value_yt) on the same scale as this project's value
measure (snap share x player's market price), which the cap-to-strength bridge used?"""
import numpy as np, pandas as pd
from pathlib import Path
CF = Path.home() / "cfdarko"
L = pd.read_parquet(CF / "data/processed/rookie_value_labels.parquet").dropna(subset=["pfr_id"])
lib = pd.read_parquet("data/derived/value_library.parquet")
m = lib.merge(L[["pfr_id"] + [f"value_y{t}" for t in range(1, 5)]], left_on="pfr_id", right_on="pfr_id")
print(f"Matched players: {len(m)} of {len(lib)} in this project's library (drafts 2013-2021)")
fam = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "ED": "EDGE",
       "IDL": "DL", "LB": "LB", "CB": "DB", "S": "DB"}
m["fam"] = m.grp.map(fam)
rows = []
for t in range(1, 5):
    ours = m[f"share_{t}"] * m.price
    cf = m[f"value_y{t}"].clip(lower=0)
    rows.append({"season": t, "ours (avg % cap)": ours.mean() * 100, "cfdarko (avg % cap)": cf.mean() * 100,
                 "ratio cf/ours": cf.sum() / ours.sum(), "corr": np.corrcoef(ours, cf)[0, 1]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
m["ours14"] = sum(m[f"share_{t}"] * m.price for t in range(1, 5))
m["cf14"] = sum(m[f"value_y{t}"].clip(lower=0) for t in range(1, 5))
g = m.groupby("fam").apply(lambda d: pd.Series({"n": len(d), "ours": d.ours14.mean() * 100, "cfdarko": d.cf14.mean() * 100,
                                                "ratio": d.cf14.sum() / d.ours14.sum(), "corr": d.ours14.corr(d.cf14)}),
                           include_groups=False)
print("\n4-season totals by position family (% of cap):"); print(g.round(2).to_string())
m["rnd"] = pd.cut(m.pick, [0, 32, 100, 262], labels=["1-32", "33-100", "101+"])
print("\nBy pick range:"); print(m.groupby("rnd", observed=True)[["ours14", "cf14"]].mean().mul(100).round(2).to_string())
