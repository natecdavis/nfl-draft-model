"""Fresh replication of the top-30-visit lead (scripts/visits_udfa_test.py, V4) on classes the test never saw.

Lead (2018-2022): an off-board undrafted signer brought in for a top-30 private visit by a team other than the one
that signed him did x2.4 better than the public pre-draft model expected (94 players, 3 of 4 classes).
Replication classes: 2016 and 2017 (4 seasons) and 2023 (3 seasons so far: actual and expectation both use
seasons 1-3). 2015 is left out: the outcome labels for undrafted players start with the 2016 class.
Same definitions as the original: UDFA contract = OverTheCap contract with no draft round signed in the draft year;
matched to the prospect universe by normalized name + class (unique keys); public model = full_pre fit leave-one-class-out on 2014-2022
(board + college stats + combine + recruiting, no draft position); visits by teams other than the signer.
Specified before looking:
  R1 primary: off-board signers, Poisson value ~ top-30 flag + position + class + log public expectation
     (covariate, as in the original); success = effect > 1 with p < 0.05 one-sided
  R2 the same given the guarantee (z of log1p $k)
  R3 per log1p(top-30 visits); R4 each class separately; R5 teams met (any meeting; original null) for context
Run: python scripts/visits_udfa_replication.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import multisport_test as mt
import prospect_model as pm
from public_board import _norm
from visits_test import visits
from visits_udfa_test import NICK

CLASSES = (2016, 2017, 2023)
NSEAS = {2016: 4, 2017: 4, 2023: 3}


def signers(uni):
    u = uni[uni.cls.isin(CLASSES) & ~uni.drafted.astype(bool)].copy()
    u["ns"] = u.cls.map(NSEAS)
    for t in pm.SEASONS:
        u[f"y{t}"] = np.where(t <= u.ns, u[f"value_y{t}"].fillna(0.0).clip(lower=0), 0.0)
    u["actual"] = u[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    u["k"] = u.player_name.map(_norm)
    u = u[~u.duplicated(["k", "cls"], keep=False)]
    c = pd.read_parquet("data/raw/contracts.parquet")
    c = c[c.draft_round.isna() & (c.draft_year == c.year_signed) & c.year_signed.isin(CLASSES)].copy()
    c = c.sort_values(["player", "year_signed", "guaranteed"], ascending=[True, True, False])
    c["k"] = c.player.map(_norm)
    c = c.drop_duplicates(["k", "year_signed"])
    c["gtd_k"] = c.guaranteed.fillna(0) * 1000
    m = u.merge(c[["k", "year_signed", "gtd_k", "team"]], left_on=["k", "cls"], right_on=["k", "year_signed"], how="inner")
    m["lg"] = np.log1p(m.gtd_k)
    m["z"] = (m.lg - m.lg.mean()) / m.lg.std()
    m["on_board"] = m.hasan_rank.notna().astype(float)
    m["grp"] = m.position_group.astype(str)
    return m


def public(uni):
    """full_pre expectation for each replication class, fit on the other board-era classes 2014-2022 (the rolling
    fit of the original can't reach 2016-17: board features start in 2014, so there are no earlier classes with
    four seasons). Training on later classes uses no outcome of the class being predicted."""
    saved = dict(pm.PARAMS); pm.PARAMS.update(mt.REG)
    out = []
    try:
        for c in CLASSES:
            tr = uni[(uni.cls != c) & uni.cls.between(2014, 2022)]
            te = uni[uni.cls == c][["player_id"]].copy()
            te["exp_pub"] = 0.0
            for t in pm.SEASONS[:NSEAS[c]]:
                m, feats = pm.fit(tr, "full_pre", t)
                te["exp_pub"] += m.predict(uni.loc[uni.cls == c, feats])
            out.append(te)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    return pd.concat(out)


def fit(d, var, extra=""):
    f = smf.glm(f"actual ~ {var} + {extra}C(grp) + lexp" + (" + C(cls)" if d.cls.nunique() > 1 else ""), d,
                family=sm.families.Poisson()).fit(cov_type="HC1")
    lo, hi = np.exp(f.conf_int().loc[var])
    return np.exp(f.params[var]), lo, hi, f.pvalues[var]


def main():
    uni = pm.load()
    m = signers(uni).merge(public(uni), on="player_id", how="inner").dropna(subset=["exp_pub"])
    m["lexp"] = np.log(m.exp_pub.clip(lower=1e-4))
    m["sign_fr"] = m.team.map(lambda t: NICK.get(str(t).split()[-1]))
    vk = visits(uni)
    vk["player_id"] = vk.player_id.astype(str)
    m["player_id"] = m.player_id.astype(str)
    met = vk.groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    pri = vk[vk.PRI > 0].groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    key = list(zip(m.player_id, m.cls))
    m["n_pri"] = [len(pri.get(k, set()) - {s}) for s, k in zip(m.sign_fr, key)]
    m["n_other"] = [len(met.get(k, set()) - {s}) for s, k in zip(m.sign_fr, key)]
    m["pri"] = (m.n_pri > 0).astype(float)
    m["l_n_pri"], m["l_n_other"] = np.log1p(m.n_pri), np.log1p(m.n_other)
    off = m[m.on_board == 0].copy()
    print(f"undrafted signers {len(m)} ({m.groupby('cls').size().to_dict()}); off board {len(off)}; signing team mapped "
          f"{off.sign_fr.notna().mean():.0%}; top-30 visit by another team {int(off.pri.sum())} "
          f"({off.groupby('cls').pri.sum().astype(int).to_dict()}); any meeting {(off.n_other > 0).mean():.0%}")
    r = lambda x: x.actual.sum() / x.exp_pub.sum()
    print(f"actual / public model: visited {r(off[off.pri == 1]):.2f}, not {r(off[off.pri == 0]):.2f}; "
          f"played at all {(off.actual[off.pri == 1] > 0).mean():.0%} vs {(off.actual[off.pri == 0] > 0).mean():.0%}")
    rows = [("R1 top-30 visit flag (PRIMARY)", off, "pri", ""),
            ("R2 same, given guarantee", off, "pri", "z + "),
            ("R3 per log1p(top-30 visits)", off, "l_n_pri", ""),
            ("R5 teams met, any meeting (original x1.03)", off, "l_n_other", "")]
    rows += [(f"R4 {c}", g, "pri", "") for c, g in off.groupby("cls")]
    print("\n(original 2018-22: flag x2.43, given guarantee x2.13, per log1p x1.88)")
    for lab, d, var, extra in rows:
        if d[var].gt(0).sum() < 5:
            print(f"    {lab:44s} too few ({int(d[var].gt(0).sum())})")
            continue
        e, lo, hi, p = fit(d, var, extra)
        print(f"    {lab:44s} n={len(d):4d} (flagged {int(d[var].gt(0).sum()):3d})  x{e:.2f} (95% {lo:.2f}-{hi:.2f}, "
              f"one-sided p {p / 2 if e > 1 else 1 - p / 2:.3f})")
    top = off[off.pri == 1].sort_values("actual", ascending=False)
    print("\nlargest visited outcomes: " + ", ".join(f"{n} ({c}) {a:.3f}" for n, c, a in
                                                    list(zip(top.player_name, top.cls, top.actual))[:5]))
    print(f"share of visited group's value from top 3: {top.actual.head(3).sum() / max(top.actual.sum(), 1e-9):.0%}")
    off.to_parquet(pm.OUT / "visits_udfa_replication.parquet")


if __name__ == "__main__":
    main()
