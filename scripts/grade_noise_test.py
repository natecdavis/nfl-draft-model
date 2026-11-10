"""Grade noise at DB / RB / LB: does the model's currency make elite prospects there look too much like average ones?

PFF season grades carry over poorly at RB (year-to-year r ~0.11) and DB (~0.19) vs ~0.4-0.5 elsewhere, and the
label's pay curves price quality almost flat there (full-time DB 3.5 -> 4.4% of cap from q 0.1 to 0.9, LB 2.6 -> 3.1,
RB 5.1 -> 5.5; WR 4.2 -> 9.5, EDGE 7.1 -> 10.1). If the flatness is noise, the label (and so the model's expected
value by board rank) understates how much better top DB / LB / RB prospects turn out.

Part 1. Spread by board rank, per group, in four currencies, drafted board players (Hasan rank), classes 2012-20:
  label  = the model's 4-year value (talent-model label, y1-4, % of cap)
  pay    = best second-contract APY (% of cap; pick_outcomes.best_pay, 0 if none)
  av     = PFR weighted career AV;   starts = seasons as a starter
  Poisson GLM, Y ~ log(rank + 10) per group; elasticity b (how fast value falls with rank) and the ratio
  E[Y | rank 5] / E[Y | rank 50]. Compression = label elasticity / outside elasticity, compared across groups
  (all four share the slot-driven playing-time path, so the cross-group comparison is what matters).
  CIs: bootstrap over players.
Part 2. Is the curve's flat quality slope noise? Veteran contracts (not rookie deals) signed 2014-25: APY % of cap
  on the player's quality before signing, measured (a) by last season's shrunk-grade-like percentile (talent-model
  'post' talent of the last season only = z), and (b) the talent model's smoothed multi-season talent; both as
  within group-season percentiles, controlling for last season's snap share. Attenuation = slope(a) / slope(b).

Usage: python scripts/grade_noise_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT, RAW = ROOT / "data/derived", ROOT / "data/raw"
GROUPS = ["WR", "TE", "OL", "DL", "EDGE", "RB", "LB", "DB"]
pd.set_option("display.width", 220)


def part1():
    import prospect_model as pm
    d = pm.load()
    d = d[d.drafted & d.hasan_rank.notna() & d.cls.between(2012, 2020) & d.grp.isin(GROUPS)].copy()
    d["label"] = d[["y1", "y2", "y3", "y4"]].clip(lower=0).sum(axis=1) * 100
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    pay = {(y, p): v * 100 for y, p, v in zip(po.draft_year, po.pick, po.best_pay.fillna(0))}
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    av = {(y, p): v for y, p, v in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): v for y, p, v in zip(dp.season, dp.pick, dp.seasons_started)}
    key = list(zip(d.cls, d.pick.astype(int)))
    d["pay"] = [pay.get(k, 0.0) for k in key]
    d["av"] = [max(av.get(k, 0.0) or 0.0, 0.0) for k in key]
    d["starts"] = [gs.get(k, 0.0) or 0.0 for k in key]
    d[["label", "pay", "av", "starts"]] = d[["label", "pay", "av", "starts"]].astype(float).fillna(0.0)
    d["lr"] = np.log(d.hasan_rank + 10)
    print(f"Part 1: {len(d)} drafted board players, classes 2012-20")
    rng = np.random.default_rng(0)
    cur = ["label", "pay", "av", "starts"]

    def elast(D, y):
        m = sm.GLM(D[y], sm.add_constant(D[["lr"]]), family=sm.families.Poisson()).fit()
        return m.params["lr"], m

    rows = []
    for g in GROUPS:
        D = d[d.grp == g].reset_index(drop=True)
        r = {"group": g, "n": len(D)}
        el = {y: elast(D, y)[0] for y in cur}
        boots = {y: [] for y in cur}
        for _ in range(300):
            s = D.iloc[rng.integers(0, len(D), len(D))]
            for y in cur:
                boots[y].append(elast(s, y)[0])
        for y in cur:
            r[f"b_{y}"] = el[y]
            m = elast(D, y)[1]
            p5, p50 = (float(m.predict(np.array([[1.0, np.log(k + 10)]]))[0]) for k in (5, 50))
            r[f"x5/50_{y}"] = p5 / p50
        for y in ["pay", "av", "starts"]:
            ratio = np.array(boots["label"]) / np.array(boots[y])
            r[f"label/{y}"] = f"{el['label'] / el[y]:.2f} ({np.percentile(ratio, 5):.2f}-{np.percentile(ratio, 95):.2f})"
        rows.append(r)
    T = pd.DataFrame(rows)
    print("\nElasticity of value with board rank (more negative = steeper fall from top prospects; Poisson GLM on log(rank+10)),")
    print("x5/50 = expected value at rank 5 / at rank 50; label/X = label elasticity / X's (90% bootstrap range)")
    print(T.round(2).to_string(index=False))
    # within group: how well does the label track pay among drafted board players
    print("\nSpearman correlation of the label with outside yardsticks, within group:")
    print(d.groupby("grp")[cur].corr(method="spearman").xs("label", level=1)[["pay", "av", "starts"]].reindex(GROUPS).round(2).to_string())
    return d


def part2():
    tk = pd.read_parquet(OUT / "darko_pff.parquet")
    tk = tk[tk.g.isin(GROUPS + ["ED"])].copy()
    pl_ = pd.read_parquet(RAW / "players.parquet")
    c = pd.read_parquet(RAW / "contracts.parquet")
    c = c[(c.year_signed.between(2014, 2025)) & (c.draft_year.isna() | (c.year_signed > c.draft_year))
          & c.apy_cap_pct.notna() & c.gsis_id.notna()]
    # link PFF player_id -> gsis via players.parquet (pff_id column)
    idcol = "pff_id" if "pff_id" in pl_.columns else None
    if idcol is None:
        print("no pff_id in players.parquet; part 2 skipped")
        return
    ids = pl_.assign(pff=pd.to_numeric(pl_[idcol], errors="coerce")).dropna(subset=["pff", "gsis_id"])
    p2g = ids.drop_duplicates("pff").set_index("pff").gsis_id
    tk["gsis"] = tk.player_id.astype(float).map(p2g)
    tk = tk.dropna(subset=["gsis"])
    # percentiles within group-season, among players with >= 300 snaps
    tk = tk[tk.snaps >= 300].copy()
    for col in ["z", "talent_post", "talent_smooth"]:
        tk[f"pct_{col}"] = tk.groupby(["g", "season"])[col].rank(pct=True)
    last = tk.set_index(["gsis", "season"])
    rows = []
    for gs, y, apy in zip(c.gsis_id, c.year_signed, c.apy_cap_pct):
        if (gs, y - 1) in last.index:
            r = last.loc[(gs, y - 1)]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            rows.append({"g": r.g, "apy": apy * 100, "snaps": r.snaps, "q_last": r.pct_z, "q_filt": r.pct_talent_post,
                         "season": y})
    D = pd.DataFrame(rows)
    D["g"] = D.g.replace({"ED": "EDGE"})
    print(f"\nPart 2: {len(D)} veteran contracts linked to the prior season's PFF row (>= 300 snaps)")
    out = []
    for g in GROUPS:
        X = D[D.g == g]
        if len(X) < 40:
            continue
        r = {"group": g, "n": len(X), "mean apy": X.apy.mean()}
        for q in ["q_last", "q_filt"]:
            m = smf.ols(f"apy ~ {q} + snaps + C(season)", X).fit(cov_type="HC1")
            lo, hi = m.conf_int().loc[q]
            r[f"slope {q}"] = f"{m.params[q]:.2f} ({lo:.2f} to {hi:.2f})"
            r[f"_{q}"] = m.params[q]
            r[f"r {q}"] = X[[q, "apy"]].corr(method="spearman").iloc[0, 1]
        r["attenuation last/filtered"] = r["_q_last"] / r["_q_filt"]
        out.append(r)
    T = pd.DataFrame(out).drop(columns=["_q_last", "_q_filt"])
    full = []
    curve = {"DB": (3.45, 4.41), "LB": (2.62, 3.11), "RB": (5.07, 5.54), "TE": (4.06, 5.82), "OL": (2.60, 5.89),
             "DL": (7.09, 9.73), "EDGE": (7.06, 10.07), "WR": (4.16, 9.50)}
    for g, (lo, hi) in curve.items():  # full-time players only (>= 750 PFF snaps), like the curve at 90% share
        X = D[(D.g == g) & (D.snaps >= 750)]
        m = smf.ols("apy ~ q_filt + C(season)", X).fit(cov_type="HC1")
        b, ci = m.params.q_filt * 0.8, m.conf_int().loc["q_filt"] * 0.8
        full.append({"group": g, "n": len(X), "market q .1->.9": f"{b:.2f} ({ci[0]:.2f} to {ci[1]:.2f})",
                     "curve q .1->.9": round(hi - lo, 2), "curve / market": round((hi - lo) / b, 2)})
    print("APY (% of cap) per 0-1 quality percentile, controlling for snaps and signing year; slope on last season's grade")
    print("vs the talent model's filtered talent (multi-season). Pay curve's own range (q 0.1 -> 0.9 at 90% snaps):")
    print("  DB +0.96, LB +0.49, RB +0.47, TE +1.76, OL +3.29, DL +2.64, EDGE +3.01, WR +5.34")
    print(T.round(2).to_string(index=False))
    print("\nFull-time players (>= 750 snaps): APY gain from quality percentile 0.1 to 0.9, market vs the label's curve")
    print(pd.DataFrame(full).to_string(index=False))


if __name__ == "__main__":
    part1()
    part2()
