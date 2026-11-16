"""Robustness for scripts/fa_vs_extension.py: own vs new team with (a) age split, (b) franchise fixed effects,
(c) team-season rating next season (does FA money just arrive on bad teams?), (d) by position group."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import age_pricing as ap  # noqa: E402


def classify(ps):
    main_team = ps.sort_values("share").groupby(["pfr_player_id", "season"]).team.last()
    prev = pd.Series([main_team.get((p, int(s) - 1)) if pd.notna(s) else None
                      for p, s in zip(ps.pfr_player_id, ps.signed)], index=ps.index)
    at = pd.Series([main_team.get((p, int(s))) if pd.notna(s) else None
                    for p, s in zip(ps.pfr_player_id, ps.signed)], index=ps.index)
    ps["cls"] = np.select([~ps.vet, prev.isna(), prev == at], ["rookie_deal", "other", "own"], "new")
    return ps


def fit(ps, rat, key, label, fe=False, pairs=()):
    W = ps.pivot_table(index=["season", "team"], columns=key, values="value", aggfunc="sum", fill_value=0).reset_index()
    cols = [c for c in W.columns if c not in ("season", "team")]
    df = W.merge(rat, on=["season", "team"])
    df[cols] = df[cols] - df.groupby("season")[cols].transform("mean")
    f = "rating ~ " + " + ".join(f"Q('{c}')" for c in cols) + (" + C(team)" if fe else "")
    m = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": df.team})
    print(f"\n--- {label} ---")
    for c in cols:
        n = f"Q('{c}')"
        lo, hi = m.conf_int().loc[n]
        print(f"  {c:22s} {m.params[n]:+.3f} ({lo:+.3f} to {hi:+.3f})")
    for a, b in pairs:
        t = m.t_test(f"Q('{a}') - Q('{b}') = 0")
        print(f"  {a} - {b}: {float(np.ravel(t.effect)[0]):+.3f}, p {float(np.ravel(t.pvalue)[0]):.3f}")


def main():
    ps = classify(ap.player_seasons())
    rat = ap.ratings()
    nonqb_vet = (ps.grp != "qb") & ps.cls.isin(["own", "new"])
    base = np.where(ps.grp == "qb", "qb_" + ps.cls, np.where(ps.cls.isin(["own", "new", "other"]), "vet_" + ps.cls, ps.cls))
    v = ps[nonqb_vet]
    print("non-QB vet mean age (value-weighted): own %.1f, new %.1f" % tuple(
        np.average(v[v.cls == c].age, weights=v[v.cls == c].value) for c in ["own", "new"]))
    ps["b"] = base
    fit(ps, rat, "b", "baseline", pairs=[("vet_own", "vet_new")])
    fit(ps, rat, "b", "franchise fixed effects", fe=True, pairs=[("vet_own", "vet_new")])
    ps["ba"] = np.where(nonqb_vet, ps.b + np.where(ps.age >= 29, "_29plus", "_u29"), ps.b)
    fit(ps, rat, "ba", "age split at 29", pairs=[("vet_own_u29", "vet_new_u29"), ("vet_own_29plus", "vet_new_29plus")])
    ps["bg"] = np.where(nonqb_vet, ps.b + "_" + ps.grp, ps.b)
    fit(ps, rat, "bg", "by position group", pairs=[(f"vet_own_{g}", f"vet_new_{g}") for g in ["ol", "skill", "front", "db"]])
    # next season's rating on this season's money: if FA money goes to bad teams that improve, the gap shrinks
    r2 = rat.copy(); r2["season"] -= 1
    fit(ps, r2.rename(columns={"rating": "rating"}), "b", "next season's rating on this season's roster",
        pairs=[("vet_own", "vet_new")])


if __name__ == "__main__":
    main()
