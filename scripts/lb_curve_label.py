"""LB pay curve refit at signing (lead from scripts/lb_value_test.py; README 'Grade noise and flat quality pricing').

cfdarko's pay_curves_wide fits price(share, q) on the contract in force each season against THAT season's quality.
A contract is priced on the play before it was signed, so where quality persists poorly the fitted q slope flattens
(LB: curve / market slope 0.20; top LB ceiling ~5.6% vs pay to 10%). Here the LB curve only is refit at signing:
  rows      LB veteran contracts (same filter as cfdarko: not the rookie deal, not a 5th-year option), signed in
            PAY_WIDE, with snaps the season before signing
  features  snap share and quality q in the season before signing (same q table as the label)
  target    APY % of cap at signing; monotone LightGBM, same PAY_PARAMS as cfdarko
  level     rescaled so the new curve's total over the in-force LB seasons of PAY_LEVEL equals the old curve's
            (today's market level unchanged; only the shape across share / quality changes)
Other groups keep cfdarko's curves. cfdarko stays read-only (patched in memory, as vet_season_values.py does).

Run: .venv/bin/python scripts/lb_curve_label.py label   -> data/derived/labels_pickadj_narrow_darko_lbsign.parquet
     .venv/bin/python scripts/lb_curve_label.py vet     -> data/derived/vet_season_values_lbsign.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "label"
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
if MODE == "vet":
    import vet_season_values as vsv  # noqa: E402
    dl = vsv.dl
else:
    import darko_label as dl  # noqa: E402
rv = dl.rv
ORIG = rv.pay_curves_wide
GROUPS = ("LB",)


def at_signing(snaps, pfr2gsis, c, qv, g):
    sh = snaps["share"].to_dict()
    g2p = {}
    for p, gs in pfr2gsis.items():
        if isinstance(gs, str):
            g2p.setdefault(gs, p)
    rookie = c["year_signed"] == c["draft_year"]
    fifth = (c["draft_round"] == 1) & (c["years"] == 1) & (c["year_signed"] == c["draft_year"] + 4)
    v = c[~rookie & ~fifth & (c["position_group"] == g)].dropna(subset=["gsis_id", "year_signed", "apy_cap_pct"])
    v = v[v.year_signed.between(*rv.PAY_WIDE)]
    rows = []
    for r in v.itertuples():
        p = g2p.get(r.gsis_id)
        y = int(r.year_signed) - 1
        s = sh.get((p, y), 0.0) if p is not None else 0.0
        if s > 0:
            rows.append((s, qv(r.gsis_id, g, y), r.apy_cap_pct))
    return pd.DataFrame(rows, columns=["share", "q", "apy"])


def in_force(snaps, pfr2gsis, c, qv, g):
    """The rows cfdarko's level step uses: LB seasons of PAY_LEVEL with the contract in force."""
    d = snaps.reset_index()
    d = d[d["season"].between(*rv.PAY_LEVEL) & (d["share"] > 0)].copy()
    d["gsis"] = d["pfr_player_id"].map(pfr2gsis)
    rookie = c["year_signed"] == c["draft_year"]
    fifth = (c["draft_round"] == 1) & (c["years"] == 1) & (c["year_signed"] == c["draft_year"] + 4)
    vet = c[~rookie & ~fifth & (c["position_group"] == g)].dropna(subset=["gsis_id", "year_signed", "years", "apy_cap_pct"])
    m = d.dropna(subset=["gsis"]).merge(vet, left_on="gsis", right_on="gsis_id")
    m = m[(m["year_signed"] <= m["season"]) & (m["season"] < m["year_signed"] + m["years"])]
    m = m.sort_values("year_signed").drop_duplicates(["gsis", "season"], keep="last")
    m["q"] = [qv(gs, g, y) for gs, y in zip(m["gsis"], m["season"])]
    return m


def patched(snaps, pfr2gsis, c, qv, qgroups, fold=None, k=5):
    curves = ORIG(snaps, pfr2gsis, c, qv, qgroups, fold, k)
    import lightgbm as lgb
    for g in GROUPS:
        S = at_signing(snaps, pfr2gsis, c, qv, g)
        mdl = lgb.LGBMRegressor(**rv.PAY_PARAMS, monotone_constraints=[1, 1]).fit(S[["share", "q"]], S["apy"])
        memo = {}

        def price(sh, qq, mdl=mdl, memo=memo):
            key = (round(sh, 3), round(qq, 2))
            if key not in memo:
                memo[key] = max(float(mdl.predict(pd.DataFrame([key], columns=["share", "q"]))[0]), 0.0)
            return memo[key]
        M = in_force(snaps, pfr2gsis, c, qv, g)
        old = curves[g]
        scale = sum(old(a, b) for a, b in zip(M.share, M.q)) / max(sum(price(a, b) for a, b in zip(M.share, M.q)), 1e-12)

        def new(sh, qq, f=price, s=scale):
            return f(sh, qq) * s
        grid = [(sh, qq) for sh in (0.5, 0.9) for qq in (0.1, 0.5, 0.75, 0.9, 0.97, 1.0)]
        print(f"\nLB curve refit at signing: {len(S)} contracts, level scale {scale:.3f}")
        print("  share, q: old -> new (APY % cap)  " + "; ".join(
            f"{sh},{qq}: {old(sh, qq) * 100:.2f}->{new(sh, qq) * 100:.2f}" for sh, qq in grid))
        curves[g] = new
    return curves


rv.pay_curves_wide = patched

if MODE == "vet":
    vsv.OUT = ROOT / "data/derived/vet_season_values_lbsign.parquet"
    vsv.main()
else:
    dl.OUT = ROOT / "data/derived/labels_pickadj_narrow_darko_lbsign.parquet"
    dl.main()
