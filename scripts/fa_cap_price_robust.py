"""Complaint #1 (free-agency list, Nov 20 2026): "the tool says to spend on almost everyone".
Value is credited at the bridge rate (0.19 non-QB / 0.34 QB per 1% of cap of value) but cap is charged at
pick_value.COST_RATE (0.10). Score every 2027 free agent at his predicted contract at both charges (0.10 and the strict
0.187 = a cap dollar costs a fair-priced dollar of value) and count firm / depends / no, by situation and group."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fa_market as fm  # noqa: E402
import pick_value as pv  # noqa: E402
import veteran as vt  # noqa: E402

STRICT = fm.STRICT_RATE
fa = fm.free_agents(vt.LAST + 2)
fa = fa[~fa.restricted]
rows = []
for name, sit in pv.SITUATIONS.items():
    R = fm.rank(fa, sit, n=3000, strict=True, break_even_=False)
    R["sit"] = name
    rows.append(R)
D = pd.concat(rows)
D["label"] = fm.robust_label(D.titles, D.titles_strict)
print(f"{len(fa)} unrestricted free agents x 4 situations; strict rate {STRICT:.3f}\n")
print(pd.crosstab(D.sit, D.label, normalize="index").round(2))
print()
pos = D[D.titles > 0]
print("of players positive at 0.10, share firm, by group (all situations):")
print((pos.label == "firm").groupby(pos.grp).mean().round(2).to_string())
print("\nby predicted APY (% of cap), positive at 0.10:")
b = pd.cut(pos.apy_pred, [0, 0.01, 0.02, 0.05, 1], labels=["<1%", "1-2%", "2-5%", "5%+"])
print((pos.label == "firm").groupby(b, observed=True).agg(["mean", "size"]).round(2).to_string())
print("\ntitles added summed over all positive signings (% of one title), mean per situation:")
print(D.groupby("sit").apply(lambda d: pd.Series({"sum@0.10": d.titles[d.titles > 0].sum() * 100,
                                                    "sum_strict_of_firm": d.titles_strict[d.label == "firm"].sum() * 100}),
                             include_groups=False).round(1))
D.to_parquet(vt.OUT / "fa_cap_price_robust.parquet")
