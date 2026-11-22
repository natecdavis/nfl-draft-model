"""Extend a rookie-deal QB after year 3, or wait? (the extension model prices both at the same APY)

Population: drafted QBs, classes 2012-2021 (decision-point data from 2013; waiting to year 5 observable by 2026).
Second contract = first contract after the rookie deal with APY >= 5% of cap (extension, franchise / transition tag,
free-agent deal; any team; 5th-year options ('Other') and 1-year non-tag deals excluded). same = signed with
the team that held him after year 3 (the re-sign-later case the planner compares with). Timing d = year signed - draft year: early d = 3 (after his 3rd season, the first year
allowed), late d >= 4.
Decision-point quality: fa_market.features at Y = draft year + 3, lag 1 (season values for seasons 3 and 2, games
share), past-only value column (value_ours_f) - what the team knew when it could first extend.
Questions:
  1 price: log APY (share of cap) of the second contract, late vs early, controlling for decision-point quality
    (and signing year). Positive = waiting costs more for the same year-3 player.
  2 same information: the same, controlling for quality at the actual signing (lag 1 at his signing year) - the pure
    market drift of waiting, not what he showed in the extra season.
  3 information value: QBs who could have been extended early (decision-point v1 in the early group's range) but
    weren't - how many never got a 5%+ second deal by year 5 (the money waiting saved).
  Also every position, round 1, for a bigger sample of the same timing premium.
Run: .venv/bin/python scripts/qb_extension_timing.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402

fm.VALUE_COL = "value_ours_f"
BIG = 0.05
CLASSES = range(2012, 2022)


def second_deals(P):
    H = fm.history()
    rows = []
    for r in P.itertuples():
        h = [x for x in H.get(r.pfr_id, []) if x[0] > r.draft_year and x[1] not in ("Drafted", "UDFA", "Other", None, "")
             and x[2] >= BIG and (x[3] >= 2 or x[1] in ("Franchise", "Transition"))]
        prev = [x for x in H.get(r.pfr_id, []) if x[0] <= r.draft_year + 3]
        if h:
            y = h[0][0]
            same = [x for x in h if x[0] == y]
            x = max(same, key=lambda z: z[2])  # tag then long deal in the same year: the long deal
            rows.append({"pfr_id": r.pfr_id, "Y2": y, "kind": x[1], "apy": x[2], "yrs": x[3], "team2": x[5],
                         "same": x[1] in ("Extension", "Franchise", "Transition") or (bool(prev) and x[5] == prev[-1][5])})
    return pd.DataFrame(rows)


def build(pos=None, rnd=None):
    p = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id", "draft_year"]).drop_duplicates("pfr_id")
    p = p[p.draft_year.isin(CLASSES)]
    if pos:
        p = p[p.position == pos]
    if rnd:
        p = p[p.draft_round == rnd]
    p = p[["pfr_id", "display_name", "position", "draft_year", "draft_round", "draft_pick"]].copy()
    p["draft_year"] = p.draft_year.astype(int)
    F = pd.concat([fm.features(g.pfr_id, int(y) + 3, 1) for y, g in p.groupby("draft_year")])
    D = p.merge(F[["pfr_id", "v1", "v2", "g1"]], on="pfr_id").merge(second_deals(p), on="pfr_id", how="left")
    D["d"] = D.Y2 - D.draft_year
    D["grp"] = D.position
    # quality at the actual signing (lag 1)
    S = pd.concat([fm.features(g.pfr_id, int(y), 1)[["pfr_id", "v1", "v2"]].rename(columns={"v1": "s1", "v2": "s2"})
                   for y, g in D.dropna(subset=["Y2"]).groupby("Y2")])
    D = D.merge(S, on="pfr_id", how="left")
    D["late"] = (D.d >= 4).astype(float)
    D["log_apy"] = np.log(D.apy)
    return D


def price(D, lab, extra=""):
    d = D[(D.d >= 3) & (D.d <= 6)].copy()
    d["yc"] = d.Y2 - 2020
    out = {}
    for name, f in [("decision-point quality", "log_apy ~ late + v1 + v2 + g1 + yc" + extra),
                    ("quality at signing", "log_apy ~ late + s1 + s2 + yc" + extra)]:
        m = smf.ols(f, d).fit(cov_type="HC1")
        lo, hi = m.conf_int().loc["late"]
        out[name] = (m.params["late"], lo, hi)
        print(f"  {lab:28s} {name:24s} late vs early: x{np.exp(m.params['late']):.2f} APY "
              f"(95% x{np.exp(lo):.2f}-x{np.exp(hi):.2f}), n early {int((d.late == 0).sum())} late {int(d.late.sum())}")
    return d, out


def main():
    pd.set_option("display.width", 220)
    Q = build(pos="QB")
    print(f"drafted QBs {CLASSES.start}-{CLASSES.stop - 1}: {len(Q)}; with a 5%+ second deal: {Q.Y2.notna().sum()} "
          f"(timing d: {Q.d.value_counts().sort_index().to_dict()})")
    print("\n1-2 price of waiting (log APY share of cap; ratio late / early):")
    dq, _ = price(Q, "QB, all rounds", " + C(draft_round > 1)")
    price(Q[Q.draft_round == 1], "QB, round 1")
    price(Q[Q.same.fillna(False).astype(bool)], "QB, same team", " + C(draft_round > 1)")
    print("\nQB second deals:")
    show = dq.sort_values(["late", "draft_year"])[["display_name", "draft_year", "draft_round", "d", "kind", "same", "apy",
                                                    "v1", "v2", "s1"]]
    print(show.round(3).to_string(index=False))

    print("\n3 information value: decision-point v1 at least the early group's 25th percentile")
    E = Q[Q.d == 3]
    thr = E.v1.quantile(0.25)
    C = Q[(Q.v1 >= thr)].copy()
    C["outcome"] = np.select([C.d == 3, C.d.between(4, 5), C.d >= 6], ["early", "late (yr 4-5)", "later"], "no 5%+ deal by yr 5")
    C.loc[C.Y2.isna(), "outcome"] = "no 5%+ deal by yr 5"
    print(f"  threshold v1 >= {thr:.3f}; {len(C)} QBs")
    print(C.groupby("outcome").agg(n=("pfr_id", "size"), v1=("v1", "mean"), v2=("v2", "mean"), apy=("apy", "mean"),
                                   later_v=("s1", "mean")).round(3).to_string())
    print("  not extended early, no deal by yr 5:",
          ", ".join(C[C.outcome == "no 5%+ deal by yr 5"].display_name))

    print("\nall positions, round 1 (same timing comparison):")
    A = build(rnd=1)
    price(A, "round 1, all positions", " + C(position)")
    price(A[A.same.fillna(False).astype(bool)], "round 1, same team", " + C(position)")
    A.to_csv(vt.ROOT / "reports" / "extension_timing_r1.csv", index=False)
    Q.to_csv(vt.ROOT / "reports" / "qb_extension_timing.csv", index=False)


if __name__ == "__main__":
    main()
