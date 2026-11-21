"""Probe for scripts/age_forecast_test.py: where the age over-forecast sits (forecast size) and whether the
season-in-progress features (p_share, p_gp) cause it. Reads data/derived/age_forecast_test.parquet."""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import veteran as vt  # noqa: E402

S = pd.read_parquet(vt.OUT / "age_forecast_test.parquet")
C = vt.comps()
r = lambda x, k: (x.w * x[f"y{k}"]).sum() / (x.w * x[f"f{k}"]).sum()
for k in (3, 5):
    d = S[S[f"y{k}"].notna()].copy()
    d["size"] = pd.cut(d[f"f{k}"], [-1, 0.005, 0.01, 0.02, 1], labels=["<0.5%", "0.5-1%", "1-2%", "2%+"])
    t = d.groupby(["band", "size"], observed=True).apply(lambda x: f"{r(x, k):.2f} ({len(x)})", include_groups=False)
    print(f"k={k} actual / forecast (n) by age band x forecast size:\n{t.unstack()}\n")
# without the season-in-progress features, a subsample
sub = S.groupby("band", observed=True).sample(600, random_state=1, replace=False)
for k in (3, 5):
    d = sub[sub[f"y{k}"].notna()].copy()
    d["g"] = [C.outcomes({**x, "p_share": np.nan, "p_gp": np.nan}, k)[0].mean() for x in d.to_dict("records")]
    print(f"k={k}: with partial features vs without, actual / forecast by band:")
    print(d.groupby("band", observed=True).apply(lambda x: pd.Series({
        "with": (x.w * x[f"y{k}"]).sum() / (x.w * x[f"f{k}"]).sum(),
        "without": (x.w * x[f"y{k}"]).sum() / (x.w * x.g).sum(), "n": len(x)}), include_groups=False).round(2))
