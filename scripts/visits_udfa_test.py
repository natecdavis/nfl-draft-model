"""Do pre-draft visit counts pick up the interest that undrafted guarantees reveal, for players off the board?

Follow-up to scripts/conviction_test.py (part B): guarantees for undrafted signers predict value beyond the public
pre-draft model, mostly for players the consensus board doesn't rank, but they're signed after the draft. Visits
(top-30 private visits, combine / pro-day meetings, private workouts; kentweyrauch.com via cfdarko) are reported
before the draft, so if they carry the same information they could feed late-round choices.

Universe: undrafted signers 2018-2022 off the consensus board (data/derived/conviction_udfa.parquet), matched to
visits by normalized name + class (unique keys; visits_test.visits). Reporting caveats: 2021 (COVID) has few
visit reports; some meetings are reported after the draft, and a meeting with the team that later signed him
could be one of those, so the main count leaves the signing team out.
Specified before looking:
  V1 guarantee: OLS log1p(guarantee $k) ~ log1p(teams met, other than the signing team) + log public-model
     expectation + position + class
  V2 value: Poisson value seasons 1-4 ~ same visit term, public model as covariate, position + class
  V3 value with both visits and guarantee z: does the guarantee effect survive / does visits add?
  V4 by meeting type (top-30 private visit, private workout) in V1 and V2; V5 without 2021.
Run: python scripts/visits_udfa_test.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import multisport_test as mt
import prospect_model as pm
from visits_test import visits

NICK = {v.split()[-1]: co.franchise(k) for k, v in co.TEAMS.items()} | {"Commanders": "WAS", "Football Team": "WAS"}


def public_expectation(uni):
    f = pm.OUT / "conviction_udfa_pub.parquet"
    if f.exists():
        return pd.read_parquet(f)
    saved = dict(pm.PARAMS); pm.PARAMS.update(mt.REG)
    try:
        F = pm.rolling(uni, versions=("full_pre",), test_classes=tuple(range(2018, 2023)))
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    F["exp_pub"] = F[[f"full_pre_p{t}" for t in pm.SEASONS]].sum(axis=1)
    F = F[["player_id", "exp_pub"]]
    F.to_parquet(f)
    return F


def show(f, var, lab, n, exp=True):
    lo, hi = f.conf_int().loc[var]
    b, p = f.params[var], f.pvalues[var]
    if exp:
        print(f"    {lab:58s} n={n:4d}  x{np.exp(b):.2f} (95% {np.exp(lo):.2f}-{np.exp(hi):.2f}, p {p:.3f})")
    else:
        print(f"    {lab:58s} n={n:4d}  {b:+.3f} (95% {lo:+.3f} to {hi:+.3f}, p {p:.3f})")


def main():
    uni = pm.load()
    m = pd.read_parquet(pm.OUT / "conviction_udfa.parquet")
    m = m.merge(public_expectation(uni), on="player_id", how="inner").dropna(subset=["exp_pub"])
    m["lexp"] = np.log(m.exp_pub.clip(lower=1e-4))
    m["sign_fr"] = m.team.map(lambda t: NICK.get(str(t).split()[-1]))
    vk = visits(uni)
    vk["player_id"] = vk.player_id.astype(str)
    m["player_id"] = m.player_id.astype(str)
    g = vk.groupby(["player_id", "cls"])
    met = g.fr.agg(set).to_dict()
    pri = vk[vk.PRI > 0].groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    wor = vk[vk.WOR > 0].groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    key = list(zip(m.player_id, m.cls))
    m["n_all"] = [len(met.get(k, set())) for k in key]
    m["met_signer"] = [s in met.get(k, set()) for s, k in zip(m.sign_fr, key)]
    m["n_other"] = [len(met.get(k, set()) - {s}) for s, k in zip(m.sign_fr, key)]
    m["n_pri"] = [len(pri.get(k, set()) - {s}) for s, k in zip(m.sign_fr, key)]
    m["n_wor"] = [len(wor.get(k, set()) - {s}) for s, k in zip(m.sign_fr, key)]
    for c in ["n_other", "n_pri", "n_wor"]:
        m[f"l_{c}"] = np.log1p(m[c])
    off = m[m.on_board == 0].copy()
    print(f"undrafted signers 2018-22 with public-model expectation {len(m)}; off board {len(off)}; signing team mapped "
          f"{off.sign_fr.notna().mean():.0%}")
    print(f"off board: any reported meeting {(off.n_all > 0).mean():.0%} (mean teams {off.n_all.mean():.2f}); met the signing team "
          f"{off.met_signer.mean():.0%} of all, {off.met_signer[off.n_all > 0].mean():.0%} of those with any meeting; by class "
          + " ".join(f"{c}:{(x.n_all > 0).mean():.0%}" for c, x in off.groupby("cls")))
    print(f"  guarantee median by teams met (others): 0 ${off.gtd_k[off.n_other == 0].median():.0f}k, "
          f"1 ${off.gtd_k[off.n_other == 1].median():.0f}k, 2+ ${off.gtd_k[off.n_other >= 2].median():.0f}k "
          f"(n {int((off.n_other == 0).sum())} / {int((off.n_other == 1).sum())} / {int((off.n_other >= 2).sum())})")

    base = "C(grp) + C(cls) + lexp"
    print("\nV1 GUARANTEE ~ teams met (other than the signer); OLS on log1p($k), coefficient per log1p(teams)")
    for d, lab in [(off, "off board"), (off[off.cls != 2021], "  without 2021"), (m[m.on_board == 1], "  on board (context)")]:
        f = smf.ols(f"lg ~ l_n_other + {base}", d).fit(cov_type="HC1")
        show(f, "l_n_other", lab, len(d), exp=False)
    f = smf.ols(f"lg ~ met_signer + {base}", off.assign(met_signer=off.met_signer.astype(float))).fit(cov_type="HC1")
    show(f, "met_signer", "  met the signing team (may be post-draft reporting)", len(off), exp=False)
    print("\nV2 VALUE ~ teams met (other than the signer); Poisson, public model as covariate; per log1p(teams)")
    for d, lab in [(off, "off board"), (off[off.cls != 2021], "  without 2021")]:
        f = smf.glm(f"actual ~ l_n_other + {base}", d, family=sm.families.Poisson()).fit(cov_type="HC1")
        show(f, "l_n_other", lab, len(d))
    print("\nV3 VALUE with both (off board)")
    f = smf.glm(f"actual ~ l_n_other + z + {base}", off, family=sm.families.Poisson()).fit(cov_type="HC1")
    show(f, "l_n_other", "teams met, given the guarantee", len(off))
    show(f, "z", "guarantee per sd, given teams met", len(off))
    f = smf.glm(f"actual ~ z + {base}", off, family=sm.families.Poisson()).fit(cov_type="HC1")
    show(f, "z", "  guarantee per sd alone (same rows)", len(off))
    print("\nV4 BY MEETING TYPE (off board; other than the signer)")
    for c, lab in [("l_n_pri", "top-30 private visits"), ("l_n_wor", "private workouts")]:
        f = smf.ols(f"lg ~ {c} + {base}", off).fit(cov_type="HC1")
        show(f, c, f"guarantee ~ {lab}", len(off), exp=False)
        f = smf.glm(f"actual ~ {c} + {base}", off, family=sm.families.Poisson()).fit(cov_type="HC1")
        show(f, c, f"value ~ {lab}", len(off))
    off.drop(columns=[], errors="ignore").to_parquet(pm.OUT / "visits_udfa.parquet")


if __name__ == "__main__":
    main()
