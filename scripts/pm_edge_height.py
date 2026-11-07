"""Does EDGE height improve the pre-draft model (consensus_pre) out of sample?

scripts/size_bias_test.py: tall edge rushers beat consensus (x1.26 per sd, 2018-22). Two ways to use it:
  (a) feature: edge_ht = height z within EDGE (NaN elsewhere) added to consensus_pre ('edge_pre'); placebo: the
      same with heights shuffled among EDGE players within class ('edge_plac_pre'). Same REG parameters.
  (b) adjustment: consensus_pre prediction x exp(b * edge_ht), b fit by Poisson on EARLIER test classes'
      out-of-sample residuals only (2019 uses 2018 alone; 2018 gets no adjustment).
Scored on test classes 2018-2022: EDGE board players (Poisson deviance, rank correlation of 4-yr value within
class) and all board players (rank correlation, top-32 / top-100 value captured).
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from outcomes_v2 import REG

CLASSES = (2018, 2019, 2020, 2021, 2022)


def dev(y, p):
    p = np.clip(p, 1e-5, None)
    return 2 * np.mean(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0) - (y - p))


def main():
    uni = pm.load()
    e = uni.grp == "EDGE"
    uni["edge_ht"] = np.where(e, uni.combine_ht_z, np.nan)
    rng = np.random.default_rng(0)
    uni["edge_plac"] = uni.edge_ht
    for c, g in uni[e & uni.edge_ht.notna()].groupby("cls"):
        uni.loc[g.index, "edge_plac"] = rng.permutation(g.edge_ht.to_numpy())
    pm.FEATURES["edge_pre"] = pm.FEATURES["consensus_pre"] + ["edge_ht"]
    pm.FEATURES["edge_plac_pre"] = pm.FEATURES["consensus_pre"] + ["edge_plac"]
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        r = pm.rolling(uni, versions=("consensus_pre", "edge_pre", "edge_plac_pre"), test_classes=CLASSES)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    r = r.merge(uni[["player_id", "cls", "edge_ht", "on_board"]], on=["player_id", "cls"], how="left")
    r["y14"] = r[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    for v in ("consensus_pre", "edge_pre", "edge_plac_pre"):
        r[v] = r[[f"{v}_p{t}" for t in pm.SEASONS]].sum(axis=1)

    # (b) rolling adjustment
    r["adj"] = r.consensus_pre
    bs = {}
    for c in CLASSES[1:]:
        tr = r[(r.cls < c) & r.edge_ht.notna() & (r.on_board == 1)]
        m = sm.GLM(tr.y14, sm.add_constant(tr[["edge_ht"]]), family=sm.families.Poisson(),
                   offset=np.log(tr.consensus_pre.clip(lower=1e-4))).fit()
        b = m.params.edge_ht
        bs[c] = round(float(np.exp(b)), 2)
        i = (r.cls == c) & r.edge_ht.notna()
        r.loc[i, "adj"] = r.loc[i, "consensus_pre"] * np.exp(b * r.loc[i, "edge_ht"])
    print("adjustment per sd, fit on earlier classes:", bs)

    V = ["consensus_pre", "edge_pre", "edge_plac_pre", "adj"]
    eb = r[(r.on_board == 1) & r.edge_ht.notna()]
    bd = r[r.on_board == 1]
    rc = lambda s, v: s.groupby("cls").apply(lambda g: g[v].corr(g.y14, method="spearman"), include_groups=False)

    def cap(s, v, k):
        return np.mean([g.nlargest(k, v).y14.sum() / g.nlargest(k, "y14").y14.sum() for _, g in s.groupby("cls")])

    rows = {}
    for v in V:
        rows[v] = {"EDGE: Poisson deviance": dev(eb.y14.to_numpy(), eb[v].to_numpy()),
                   "EDGE: rank corr (mean of classes)": rc(eb, v).mean(),
                   "EDGE: rank corr 2019-22 only": rc(eb[eb.cls > 2018], v).mean(),
                   "board: rank corr": rc(bd, v).mean(),
                   "board: top-32 captured": cap(bd, v, 32), "board: top-100 captured": cap(bd, v, 100)}
    print(f"EDGE board players with height: {len(eb)}; board players: {len(bd)}; test classes {CLASSES}")
    print(pd.DataFrame(rows).round(4).to_string())
    print("\nEDGE rank corr by class:\n" + pd.DataFrame({v: rc(eb, v) for v in V}).round(3).to_string())
    r.to_parquet("data/derived/pm_edge_height.parquet")


if __name__ == "__main__":
    main()
