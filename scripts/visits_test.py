"""Pre-draft team visits: does a visit give the visiting team information the board lacks?

Data: kentweyrauch.com's draft-visits dashboard as scraped by cfdarko (~/cfdarko/data/raw/draft_visits.parquet;
one row per team x player x draft year: PRI = top-30 private visit, COM = combine meeting, PRO = pro-day meeting,
WOR = private workout). Matched on normalized name + draft class, keys unique on both sides.
Already tested in cfdarko (experiments/00w, 00zh): visit COUNTS add nothing beyond board rank / pick.
New questions here (drafted 2016-2022, value vs draft slot out of sample; scripts/multisport_test.expectations):
  1. Drafted by a team that met him vs by one that didn't, at the same slot (among players with >= 1 meeting).
  2. Passes by teams that met him while he was available, weighted by the chance a team picking there takes him
     (availability model around his consensus rank), controlling for the same sum over all earlier picks;
     placebo = the same number of random teams that did not meet him.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm as N

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import prospect_model as pm
from draftday import drift, spread
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT
from public_board import _norm

VISITS = pm.CF / "data/raw/draft_visits.parquet"
NICK = {v.split()[-1]: co.franchise(k) for k, v in co.TEAMS.items()} | {"Commanders": "WAS"}


def visits(uni):
    v = pd.read_parquet(VISITS)
    v["fr"] = v.team.map(NICK)
    assert v.fr.notna().all(), v[v.fr.isna()].team.unique()
    v["k"] = v.player.map(_norm)
    v = v.rename(columns={"draft_year": "cls"})
    vk = v.drop_duplicates(["k", "cls", "fr"])
    u = uni[["player_id", "player_name", "cls"]].assign(k=uni.player_name.map(_norm))
    u = u[~u.duplicated(["k", "cls"], keep=False)]
    names = vk.groupby(["k", "cls"]).player.nunique()
    vk = vk.join(names.rename("n_names"), on=["k", "cls"])
    vk = vk[vk.n_names == 1].merge(u[["player_id", "k", "cls"]], on=["k", "cls"], how="inner")
    return vk


def poisson(d, e, cols, label):
    d = d.dropna(subset=[e] + cols)
    m = sm.GLM(d.actual, sm.add_constant(d[cols].astype(float)), family=sm.families.Poisson(),
               offset=np.log(d[e].clip(lower=1e-4))).fit(cov_type="HC1")
    k = cols[0]
    lo, hi = np.exp(m.conf_int().loc[k])
    print(f"    {label:56s} n={len(d):4d}  per unit {k}: value x{np.exp(m.params[k]):.2f} (95% {lo:.2f}-{hi:.2f}, p {m.pvalues[k]:.2f})")


def main():
    uni = pm.load()
    vk = visits(uni)
    pk = pd.read_parquet(OUT / "coaching_picks.parquet")[["season", "pick", "fr", "player_id"]].dropna(subset=["player_id"])
    pk["player_id"] = pk.player_id.astype(str)
    vk["player_id"] = vk.player_id.astype(str)
    met = vk.groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    pri = vk[vk.PRI > 0].groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    raw = pd.read_parquet(VISITS)
    print(f"visit rows {len(raw)} -> matched {len(vk)} ({len(vk) / len(raw):.0%}); players {vk.groupby(['player_id', 'cls']).ngroups}")

    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R.drop(columns=[c for c in ["pick", "fr", "season"] if c in R]).merge(pk, left_on=["player_id", "cls"], right_on=["player_id", "season"], how="inner")
    d["met"] = [met.get((p, c), set()) for p, c in zip(d.player_id, d.cls)]
    d["pri"] = [pri.get((p, c), set()) for p, c in zip(d.player_id, d.cls)]
    d["n_met"] = d.met.map(len)
    d["by_met"] = [f in m for f, m in zip(d.fr, d.met)]
    d["by_pri"] = [f in m for f, m in zip(d.fr, d.pri)]
    d = d[d.exp_slot.notna()].copy()
    k = d[d.n_met > 0].copy()
    print(f"drafted 2016-22 with outcomes {len(d)}; with >= 1 reported meeting {len(k)} (mean {k.n_met.mean():.1f} teams); "
          f"drafted by a team that met him {k.by_met.sum()} ({k.by_met.mean():.0%}), by a top-30 host {k.by_pri.sum()}")
    one = k[k.n_met == 1]
    print(f"  players with exactly one reported team: {len(one)}, that team drafted him {one.by_met.mean():.0%} "
          f"(if visits were pre-draft and random wrt the pick ~1/32 = 3%; high values suggest post-draft reporting)")

    print("\n1. DRAFTED BY A TEAM THAT MET HIM vs ONE THAT DIDN'T (same slot; players with >= 1 meeting)")
    k["log_nmet"] = np.log1p(k.n_met)
    compare(k, "by_met", "exp_slot", covs=("log_nmet",), label="any meeting")
    compare(k, "by_pri", "exp_slot", covs=("log_nmet",), label="top-30 private visit")
    compare(k[k.pick <= 100], "by_met", "exp_slot", covs=("log_nmet",), label="  any meeting, picks 1-100")
    compare(k[k.pick > 100], "by_met", "exp_slot", covs=("log_nmet",), label="  any meeting, picks 101+")
    compare(k[~k.cls.isin([2020, 2021])], "by_met", "exp_slot", covs=("log_nmet",), label="  any meeting, no 2020-21")

    # 2. weighted passes by teams that met him
    owner = pk.set_index(["season", "pick"]).fr.to_dict()
    all_picks = pk.groupby("season").pick.apply(lambda s: np.sort(s.unique())).to_dict()
    frs = sorted(pk.fr.unique())
    rng = np.random.default_rng(0)

    def hazard(rank, ps):
        mu, sd = np.log(rank + 10) + drift(rank), spread(rank)
        F = lambda x: N.cdf((np.log(np.clip(x, -9.5, None) + 10) - mu) / sd)
        return np.clip((F(ps + 0.5) - F(ps - 0.5)) / np.clip(1 - F(ps - 0.5), 1e-9, None), 0, 1)

    def weighted(r, n_plac=20):
        ps = all_picks[r.cls]
        ps = ps[ps < r.pick]
        h = hazard(r.hasan_rank, ps)
        ow = np.array([owner.get((r.cls, p)) for p in ps])
        w = lambda ts: float(h[np.isin(ow, list(ts))].sum()) if ts else 0.0
        teams = r.met - {r.fr}
        tp = r.pri - {r.fr}
        others = [t for t in frs if t not in r.met and t != r.fr]
        plac = np.mean([w(set(rng.choice(others, min(len(teams), len(others)), replace=False))) for _ in range(n_plac)]) if teams else 0.0
        return pd.Series({"w_met": w(teams), "w_pri": w(tp), "w_all": float(h.sum()), "w_placebo": plac,
                          "n_other": len(teams)})

    b = k[k.hasan_rank.notna()].copy()
    b = pd.concat([b, b.apply(weighted, axis=1)], axis=1)
    print(f"\n2. PASSES BY TEAMS THAT MET HIM, weighted by the chance a team picking there takes him "
          f"(board players with meetings {len(b)}; mean weighted passes by met teams {b.w_met.mean():.2f} "
          f"of {b.w_all.mean():.2f} by all teams)")
    poisson(b, "exp_slot", ["w_met", "w_all", "n_other"], "weighted passes by teams that met him")
    poisson(b, "exp_slot", ["w_pri", "w_all", "n_other"], "weighted passes by top-30 hosts")
    poisson(b, "exp_slot", ["w_placebo", "w_all", "n_other"], "  placebo: random teams that didn't meet him")
    f = b[b.hasan_rank < b.pick]
    poisson(f, "exp_slot", ["w_met", "w_all", "n_other"], "  fallers only (picked after board rank)")
    b.drop(columns=["met", "pri"]).to_parquet(OUT / "visits_value.parquet")


if __name__ == "__main__":
    main()
