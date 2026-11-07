"""Do passes by teams whose coaches know a player tell us anything?

For each drafted player (2016-2022, outcomes vs draft slot out of sample), the franchises connected to his
school that year (src/coaching.py; the team that took him excluded):
  passes       - connected franchises that made at least one pick before his and took someone else
  fall_passes  - connected franchises that picked during his fall (after his consensus rank, before his pick)
  n_other      - all connected franchises other than his drafter (the control: being known to many teams)
If what coaches know matters, at the same slot more connected passes = worse outcomes.
Also: undrafted board players (every team passed) - does being known to more teams predict worse
outcomes vs consensus rank?
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import prospect_model as pm
from multisport_test import CONS_CLASSES, expectations
from outlook import OUT


def poisson(d, e, cols, label):
    d = d.dropna(subset=[e] + cols)
    X = sm.add_constant(d[cols].astype(float))
    m = sm.GLM(d.actual, X, family=sm.families.Poisson(), offset=np.log(d[e].clip(lower=1e-4))).fit(cov_type="HC1")
    lg = sm.Logit((d.actual > d[e]).astype(int), sm.add_constant(pd.concat([np.log(d[e].clip(lower=1e-4)).rename("lexp"),
                                                                              d[cols].astype(float)], axis=1))).fit(disp=0, cov_type="HC1")
    k = cols[0]
    print(f"   {label:52s} n={len(d):4d}  per extra {k}: value x{np.exp(m.params[k]):.2f} (95% "
          f"{np.exp(m.conf_int().loc[k, 0]):.2f}-{np.exp(m.conf_int().loc[k, 1]):.2f}, p {m.pvalues[k]:.2f}); "
          f"odds of beating slot x{np.exp(lg.params[k]):.2f} (p {lg.pvalues[k]:.2f})")


def main():
    picks = pd.read_parquet(OUT / "coaching_picks.parquet")
    v = pd.read_parquet(OUT / "coaching_value.parquet")
    cn = co.connections()
    conn = cn.groupby(["year", "key"]).fr.agg(set).to_dict()
    first_pick = picks.groupby(["season", "fr"]).pick.apply(sorted).to_dict()  # each team's picks that year

    def counts(r):
        teams = conn.get((r.season, r.key), set()) - {r.fr}
        before = [t for t in teams if any(p < r.pick for p in first_pick.get((r.season, t), []))]
        fall = 0
        if pd.notna(r.hasan_rank) and r.hasan_rank < r.pick:
            lo = r.hasan_rank
            fall = sum(any(lo <= p < r.pick for p in first_pick.get((r.season, t), [])) for t in teams)
        return pd.Series({"n_other": len(teams), "passes": len(before), "fall_passes": fall})

    v = pd.concat([v, v.apply(counts, axis=1)], axis=1)
    v["log_pick"] = np.log(v.pick)
    k = v[v.n_other > 0].copy()
    k["pass_share"] = k.passes / k.n_other
    print(f"drafted 2016-2022 with outcomes: {len(v)}; known to at least one other team's staff: {len(k)} "
          f"(mean {k.n_other.mean():.1f} teams, passed by {k.passes.mean():.1f})")
    print("\nVALUE vs DRAFT SLOT")
    poisson(v, "exp_slot", ["passes", "n_other"], "connected passes, all picks")
    poisson(k, "exp_slot", ["pass_share", "n_other"], "share of connected teams that passed (known players)")
    poisson(k[k["round"] <= 3], "exp_slot", ["passes", "n_other"], "connected passes, rounds 1-3")
    # rounds 4-7: nearly every connected team has picked by then (passes = n_other), so no separate test
    f = v[v.hasan_rank.notna() & (v.hasan_rank < v.pick)].copy()
    f["fall"] = np.log((f.pick + 10) / (f.hasan_rank + 10))
    print(f"\n   players who fell below their board rank ({len(f)}):")
    poisson(f, "exp_slot", ["fall_passes", "n_other", "fall"], "connected teams that passed during his fall")
    f["any_fall_pass"] = (f.fall_passes > 0).astype(int)
    poisson(f, "exp_slot", ["any_fall_pass", "n_other", "fall"], "any connected team passed during his fall")

    # ---- weighted passes: each connected pass counts by the chance a team picking there takes him, given he is
    # still available (draft-day availability model around his consensus rank: log(pick+10) ~ normal).
    # Control: the same sum over every pick before his (how far he fell relative to expectations).
    # Placebo: the same number of randomly chosen unconnected teams, scored the same way (should show nothing).
    from scipy.stats import norm as N
    from draftday import drift, spread
    all_picks = picks.groupby("season").pick.apply(lambda s: np.sort(s.unique())).to_dict()
    owner = picks.set_index(["season", "pick"]).fr.to_dict()
    frs = sorted(picks.fr.unique())
    rng = np.random.default_rng(0)

    def hazard(rank, ps):
        mu, sd = np.log(rank + 10) + drift(rank), spread(rank)
        F = lambda x: N.cdf((np.log(np.clip(x, -9.5, None) + 10) - mu) / sd)
        return np.clip((F(ps + 0.5) - F(ps - 0.5)) / np.clip(1 - F(ps - 0.5), 1e-9, None), 0, 1)

    def weighted(r, placebo_sets=20):
        ps = all_picks[r.season]
        ps = ps[ps < r.pick]
        h = hazard(r.hasan_rank, ps)
        teams = conn.get((r.season, r.key), set()) - {r.fr}
        ow = np.array([owner.get((r.season, p)) for p in ps])
        w = lambda ts: float(h[np.isin(ow, list(ts))].sum()) if ts else 0.0
        others = [t for t in frs if t not in teams and t != r.fr]
        plac = np.mean([w(set(rng.choice(others, len(teams), replace=False))) for _ in range(placebo_sets)]) if teams else 0.0
        return pd.Series({"w_conn": w(teams), "w_all": float(h.sum()), "w_placebo": plac})

    wb = v[v.hasan_rank.notna()].copy()
    wb = pd.concat([wb, wb.apply(weighted, axis=1)], axis=1)
    kk = wb[wb.n_other > 0]
    print(f"\n   WEIGHTED by chance he'd be taken at that pick (board players {len(wb)}; known to another staff {len(kk)}; "
          f"mean weighted connected passes {kk.w_conn.mean():.2f} of {kk.w_all.mean():.2f} total)")
    poisson(wb, "exp_slot", ["w_conn", "w_all", "n_other"], "weighted connected passes")
    poisson(wb, "exp_slot", ["w_placebo", "w_all", "n_other"], "  placebo: random unconnected teams")
    poisson(wb, "exp_slot", ["passes", "n_other"], "  unweighted, same players")
    pl = wb.copy()
    pl["p_placebo"] = [np.mean([sum(any(p < r.pick for p in first_pick.get((r.season, t), []))
                                    for t in rng.choice([t for t in frs if t not in conn.get((r.season, r.key), set()) - {r.fr} and t != r.fr],
                                                        int(r.n_other), replace=False)) for _ in range(5)]) for r in pl.itertuples()]
    poisson(pl, "exp_slot", ["p_placebo", "n_other"], "  unweighted placebo: random unconnected teams")
    wb.to_parquet(OUT / "connected_pass_weighted.parquet")

    # undrafted board players: everyone passed; does being known to more staffs predict worse?
    uni = pm.load()
    R = expectations(uni, CONS_CLASSES)
    u = R[(~R.drafted.astype(bool)) & (R.on_board == 1) & R.exp_cons.notna()].copy()
    _, college = co.college_keys(pd.read_parquet(OUT / "coach_careers.parquet"))
    from public_board import _norm
    h = pd.read_csv(pm.CF / "hasan.csv", encoding="utf-8-sig")
    h["k"] = h.name.map(_norm)
    u["k"] = u.player_name.map(_norm)
    u = u.merge(h[["k", "year", "school"]].drop_duplicates(["k", "year"]), left_on=["k", "cls"], right_on=["k", "year"], how="left")
    u["key"] = u.school.map(lambda x: college(x) if isinstance(x, str) else None)
    u["n_conn"] = [len(conn.get((c, kk), set())) for c, kk in zip(u.cls, u.key)]
    u["log_rank"] = np.log(u.hasan_rank)
    print(f"\nUNDRAFTED BOARD PLAYERS 2018-2022: {len(u)}, school found {u.school.notna().sum()}, "
          f"known to >=1 staff {(u.n_conn > 0).sum()} (mean {u.n_conn.mean():.1f} teams)")
    poisson(u, "exp_cons", ["n_conn"], "vs consensus, per connected team that passed")
    u["any_conn"] = (u.n_conn > 0).astype(int)
    poisson(u, "exp_cons", ["any_conn"], "vs consensus, known to any staff")
    poisson(u[u.hasan_rank <= 200], "exp_cons", ["n_conn"], "  board rank <= 200, per connected team")
    v.to_parquet(OUT / "connected_pass.parquet")


if __name__ == "__main__":
    main()
