"""Do traded veterans deliver less than forecast? (step 3 check: players in the trade calculator)

A team trading a player away may know something the forecast doesn't (injury, decline, fit, attitude): a lemons
problem. Specified before running:
  sample    veterans traded in the offseason (trade date Feb 1 - Sep 1 of season Y; nflverse trades), 2014-2024,
            with an anchor for (player, Y) (value in Y-1 or Y-2). Draft-day pick swaps excluded (players only).
  forecast  src/veteran.Comps, leave-player-out (same as every other check), seasons 1-2
  outcome   realized value anywhere (his quality, whoever he plays for) and with the acquiring team
  baseline  all anchors (actual / forecast 1.00 in seasons 1-5, scripts/vet_value.py part A) and outside UFA signees
  reading   actual / forecast for traded players well below 1 (any team) = sellers know more -> haircut traded
            players in the calculator; ~1 = no adjustment. Bootstrap CI by player. Splits: forecast value >= 3% of
            cap (real starters), age 30+.
Run: .venv/bin/python scripts/traded_player_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["scripts", "src"]
import roster_need as rn  # noqa: E402
import veteran as vt  # noqa: E402


def main():
    t = pd.read_parquet(vt.RAW / "trades.parquet")
    t["date"] = pd.to_datetime(t.trade_date)
    t = t[t.pfr_id.notna() & t.date.notna()]
    t["Y"] = t.date.dt.year
    t = t[(t.date.dt.month >= 2) & (t.date < pd.to_datetime(t.Y.astype(str) + "-09-01")) & t.Y.between(2014, 2024)]
    t["to"] = t.received.map(rn.team)
    t = t.drop_duplicates(["pfr_id", "Y"], keep="last")
    A = pd.read_parquet(vt.OUT / "vet_anchors.parquet")
    D = t[["pfr_id", "Y", "to", "pfr_name"]].merge(A, on=["pfr_id", "Y"])
    C = vt.comps()
    F = pd.DataFrame([C.forecast(r) for r in D.to_dict("records")], index=D.index)
    D = D.join(F)
    played = rn._played_for()
    V = vt.season_table().set_index(["pfr_id", "season"]).value_ours.to_dict()
    for k in (1, 2):
        D[f"w{k}"] = [V.get((p, y + k - 1), 0.0) if tm in played.get((p, y + k - 1), set()) else 0.0
                      for p, y, tm in zip(D.pfr_id, D.Y, D.to)]
    D = D[D.y1.notna()]
    rng = np.random.default_rng(0)

    def ratio(d, a, f):
        r = d[a].sum() / d[f].sum()
        bs = []
        for _ in range(2000):
            s = d.iloc[rng.integers(0, len(d), len(d))]
            bs.append(s[a].sum() / s[f].sum())
        return f"{r:.2f} ({np.percentile(bs, 2.5):.2f} to {np.percentile(bs, 97.5):.2f})"

    print(f"offseason veteran trades 2014-24 with an anchor: {len(D)} players; mean forecast season 1 "
          f"{D.f1.mean() * 100:.2f}% of cap")
    rows = []
    for lab, d in [("all", D), ("forecast >= 3% of cap", D[D.f1 >= 0.03]), ("forecast < 3%", D[D.f1 < 0.03]),
                   ("age 30+", D[D.age >= 30]), ("under 30", D[D.age < 30])]:
        d2 = d[d.y2.notna()]
        rows.append({"group": lab, "n": len(d), "any team, season 1": ratio(d, "y1", "f1"),
                     "with acquirer, season 1": ratio(d, "w1", "f1"),
                     "any team, season 2": ratio(d2, "y2", "f2") if len(d2) > 20 else "",
                     "with acquirer, season 2": ratio(d2, "w2", "f2") if len(d2) > 20 else ""})
    pd.set_option("display.width", 220)
    print("actual / forecast value (1.00 = as forecast; all anchors 1.00, outside UFA signees with team ~0.86):")
    print(pd.DataFrame(rows).to_string(index=False))
    big = D[D.f1 >= 0.03].assign(r=lambda d: d.y1 / d.f1).sort_values("f1", ascending=False)
    print("\nbiggest forecasts (season 1, % of cap): forecast vs actual")
    print(big.head(15)[["pfr_name", "Y", "grp", "age", "f1", "y1", "w1"]].assign(
        f1=lambda d: d.f1 * 100, y1=lambda d: d.y1 * 100, w1=lambda d: d.w1 * 100).round(2).to_string(index=False))
    D.to_parquet(vt.OUT / "traded_player_check.parquet")


if __name__ == "__main__":
    main()
