"""Complaint #2a, part 2: does the age over-forecast found on all anchors (scripts/age_forecast_test.py) show up for
players who actually signed as unrestricted free agents? Production forecast (veteran.comps(), with the season in
progress features), UFA contracts 2015-23 with APY >= 1% of cap (scripts/vet_value.py part B): value anywhere and with the
signing team vs forecast, by age at signing and contract season; and contract points predicted vs realized by age
(part C). Run: .venv/bin/python scripts/age_forecast_signed.py"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src", "scripts"]
import veteran as vt  # noqa: E402
import vet_value as vv  # noqa: E402


def main():
    pd.set_option("display.width", 220)
    C = vt.comps()
    D = vv.part_b(C.A, C, "UFA")
    D["ab"] = pd.cut(D.age, [0, 28, 31, 33, 99], labels=["<28", "28-30", "31-32", "33+"], right=False)
    rows = []
    for k in range(1, 6):
        d = D[(D.yrs >= k) & D[f"y{k}"].notna()]
        for b, x in d.groupby("ab", observed=True):
            rows.append({"k": k, "age": b, "n": len(x), "anywhere/fc": x[f"y{k}"].sum() / x[f"f{k}"].sum(),
                         "with team/fc": x[f"w{k}"].sum() / x[f"f{k}"].sum()})
    R = pd.DataFrame(rows)
    print("\nSigned UFAs, actual / forecast by age at signing (rows) and contract season (cols), within contract years:")
    for c in ["anywhere/fc", "with team/fc", "n"]:
        print(f"-- {c}\n{R.pivot(index='age', columns='k', values=c).round(2)}")
    D.to_parquet(vt.OUT / "age_forecast_signed.parquet")
    F, th = vv.part_c(C.A, C, D, "UFA")
    F["ab4"] = pd.cut(F.age, [0, 28, 31, 33, 99], labels=["<28", "28-30", "31-32", "33+"], right=False)
    print(F.groupby("ab4", observed=True).agg(n=("apy", "size"), pred=("pred_pts", "mean"), real=("real_pts", "mean"))
          .round(3).to_string())
    F.to_parquet(vt.OUT / "age_forecast_signed_points.parquet")


if __name__ == "__main__":
    main()
