"""Stacked small signals: do the unadopted leads, combined in one shrunk multiplier, improve actual draft choices?

Single leads (captain x1.21, coaching contact, visits, RB pass-game grades, ...) each fail or can't pass a choice test
alone because a x1.1-1.3 bump on near-tied players moves realized value less than four drafts can resolve. If they
add, the combination should move more. Specified before running (Nov 22 2026).

Features (pre-draft knowable, from ledger rows with a significant residual test, not adopted):
  team-independent: rb_route (RB x z of best college PFF receiving grade), rb_pblock (RB x z of best pass-block
    grade), edge_ht (EDGE x combine height z), recruit (recruiting composite z, 0 if missing), n_met (log1p teams
    reporting a meeting / top-30 visit)
  picking-team specific: conn (a coach on the picking team's staff was at his college in the 5 seasons before),
    conn_recent (in the last 2), own_visit (picking team reported meeting him). In fitting, the actual drafter.
  Brugler (stack B only): in_guide, captain, raw (WR/DB/RB 'raw' language), track, nfl son
Multiplier: Poisson regression of actual / expected (consensus_pre prediction, 4 seasons) on standardized features,
L2 strength chosen by inner leave-one-draft-out deviance, fit leave-one-draft-out; clipped 0.5-2.
Choice test as beast_choice.py: at each pick, each version takes the best available by prediction x multiplier.
  STACK A (drafts 2018-22): baseline consensus vs consensus x stack A.
  STACK B (drafts 2019-22): baseline consensus x in-guide multiplier vs consensus x (stack A + Brugler features).
PRIMARY: stack minus baseline, Super Bowl odds per pick, all picks (each stack). SECONDARY: value; by round and picks
101+; forward-only (multiplier from earlier drafts only); 'believed' gain (what the multiplier expected) vs realized.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import coaching as co
import prospect_model as pm
from outcomes_v2 import CF_OVER_OURS, REG
from outlook import OUT
from pick_value import baseline_paths, pick_effect
from trade import Context

FAM2GRP = {"DB": "CB", "OL": "T", "DL": "IDL", "EDGE": "ED"}
A_FEATS = ["rb_route", "rb_pblock", "edge_ht", "recruit", "n_met", "conn", "conn_recent", "own_visit"]
B_FEATS = ["in_guide", "captain", "raw", "track", "son"]
ALPHAS = [1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]


def z(x):
    return (x - x.mean()) / x.std()


def static_features(uni):
    from visits_test import visits
    u = uni[["player_id", "cls", "position_group"]].copy()
    u["player_id"] = u.player_id.astype(str)
    rb = (uni.position_group == "RB").to_numpy()
    f = pd.DataFrame(index=u.player_id)
    for col, nm in [("pff_grades_pass_route_peak", "rb_route"), ("pff_grades_pass_block_peak", "rb_pblock")]:
        g = uni[col].where(rb)
        f[nm] = np.where(rb, z(g).fillna(0).to_numpy(), 0.0)
    ed = (uni.position_group == "EDGE").to_numpy()
    f["edge_ht"] = np.where(ed, uni.combine_ht_z.fillna(0).to_numpy(), 0.0)
    f["recruit"] = z(uni.recruiting_composite).fillna(0).to_numpy()
    vk = visits(uni)
    vk["player_id"] = vk.player_id.astype(str)
    met = vk.groupby("player_id").fr.agg(set)
    f["n_met"] = np.log1p(f.index.map(lambda p: len(met.get(p, ()))).to_numpy(float))
    return f, met


def brugler(uni):
    from beast_test import match
    import multisport as ms
    import re
    b = pd.read_parquet(OUT / "beast_profiles.parquet")
    txt = (b.background.fillna("") + " " + b.strengths.fillna("") + " " + b.summary.fillna("")).str.lower()
    b["captain"] = txt.str.contains("captain").astype(float)
    sw = (b.strengths.fillna("") + " " + b.weaknesses.fillna("") + " " + b.summary.fillna("")).str.lower()
    raw = r"\braw\b|unrefined|project\b|inconsistent|undisciplined|freelanc|lacks? (?:awareness|discipline|instincts|feel)"
    b["raw_any"] = sw.str.contains(raw).astype(float)
    b["track"] = ms.text_sports(b)["track"].astype(float).to_numpy()
    m = match(b, uni)
    m = m[m.player_id.notna()].drop_duplicates("player_id")
    m["player_id"] = m.player_id.astype(str)
    m["raw"] = m.raw_any * m.grp.isin(["WR", "DB", "RB"])
    fa = pd.read_parquet(OUT / "nfl_fathers.parquet")[["player_id", "father_played"]].dropna(subset=["player_id"])
    fa["player_id"] = fa.player_id.astype(str)
    m = m.merge(fa.drop_duplicates("player_id"), on="player_id", how="left")
    m["son"] = m.father_played.fillna(False).astype(float)
    m["in_guide"] = 1.0
    return m.set_index("player_id")[B_FEATS]


def team_tables():
    cp = pd.read_parquet(OUT / "coaching_picks.parquet")
    cp["player_id"] = cp.player_id.astype("string")
    key = cp.dropna(subset=["player_id", "key"]).drop_duplicates("player_id").set_index("player_id").key
    owner = cp.drop_duplicates(["season", "pick"]).set_index(["season", "pick"]).fr
    drafter = cp.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id").fr
    c = co.connections()
    links = c.groupby(["fr", "year", "key"])["last"].max()
    return key, owner, drafter, links


def team_feats(pids, cls, fr, key, links, met):
    k = pids.map(lambda p: key.get(p))
    last = np.array([links.get((fr, cls, kk), np.nan) if isinstance(kk, str) else np.nan for kk in k])
    conn = ~np.isnan(last)
    recent = conn & (np.nan_to_num(last, nan=-1) >= cls - 2)
    vis = np.array([fr in met.get(p, ()) for p in pids])
    return conn.astype(float), recent.astype(float), vis.astype(float)


def fit_mult(R, feats, train, alpha=None):
    tr = R[R.cls.isin(train) & (R.expected > 1e-4)]
    X = tr[feats].to_numpy(float); mu, sd = X.mean(0), X.std(0) + 1e-9
    if alpha is None:  # inner leave-one-draft-out
        best = None
        for a in ALPHAS:
            dev = 0.0
            for c in train:
                i, o = tr[tr.cls != c], tr[tr.cls == c]
                if len(o) == 0 or i.cls.nunique() == 0:
                    continue
                m = PoissonRegressor(alpha=a, max_iter=2000).fit((i[feats].to_numpy(float) - mu) / sd,
                                                                 i.actual / i.expected, sample_weight=i.expected)
                p = m.predict((o[feats].to_numpy(float) - mu) / sd) * o.expected
                y = o.actual.clip(lower=0)
                dev += np.sum(2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / p), 0) - (y - p)))
            if best is None or dev < best[0]:
                best = (dev, a)
        alpha = best[1]
    m = PoissonRegressor(alpha=alpha, max_iter=2000).fit((X - mu) / sd, tr.actual / tr.expected, sample_weight=tr.expected)
    return (lambda Z: np.clip(m.predict((Z - mu) / sd), 0.5, 2.0)), dict(zip(feats, m.coef_)), alpha


def build():
    uni = pm.load()
    classes = (2018, 2019, 2020, 2021, 2022)
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("consensus_pre",), test_classes=classes)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R["player_id"] = R.player_id.astype(str)
    R["actual"] = R[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    R["expected"] = R[[f"consensus_pre_p{t}" for t in pm.SEASONS]].sum(axis=1)
    S, met = static_features(uni)
    B = brugler(uni)
    key, owner, drafter, links = team_tables()
    R = R.join(S, on="player_id").join(B, on="player_id")
    R[B_FEATS] = R[B_FEATS].fillna(0.0)
    R[[c for c in S.columns]] = R[[c for c in S.columns]].fillna(0.0)
    # fitting: team features for the actual drafter (undrafted: none)
    R["conn"] = R["conn_recent"] = R["own_visit"] = 0.0
    for c in classes:
        idx = R.index[(R.cls == c) & R.player_id.isin(drafter.index)]
        for i in idx:
            p = R.at[i, "player_id"]; fr = drafter[p]
            cn, rc, vs = team_feats(pd.Series([p]), c, fr, key, links, met)
            R.at[i, "conn"], R.at[i, "conn_recent"], R.at[i, "own_visit"] = cn[0], rc[0], vs[0]
    print(f"rows {len(R)}; drafted with coaching link {int(R.conn.sum())}, own visit {int(R.own_visit.sum())}; "
          f"guide {int(R.in_guide.sum())}, captain {int(R.captain.sum())}")
    return uni, classes, R, S, B, met, key, owner, drafter, links


def main():
    uni, classes, R, S, B, met, key, owner, drafter, links = build()
    ALLB = A_FEATS + B_FEATS
    mult = {}
    for c in classes:
        oth = [x for x in classes if x != c]
        fa, ca, aa = fit_mult(R, A_FEATS, oth)
        mult[(c, "A")] = fa
        mult[(c, "A_fwd")] = fit_mult(R, A_FEATS, [x for x in classes if x < c])[0] if c >= 2020 else None
        print(f"{c} A (alpha {aa}): " + ", ".join(f"{k} {v:+.2f}" for k, v in ca.items()))
        if c >= 2019:
            ob = [x for x in oth if x >= 2019]
            fb, cb, ab = fit_mult(R, ALLB, ob)
            mult[(c, "B")] = fb
            mult[(c, "G")] = fit_mult(R, ["in_guide"], ob, alpha=1e-3)[0]
            fw = [x for x in classes if 2019 <= x < c]
            mult[(c, "B_fwd")] = fit_mult(R, ALLB, fw)[0] if len(fw) >= 2 else None
            mult[(c, "G_fwd")] = fit_mult(R, ["in_guide"], fw, alpha=1e-3)[0] if len(fw) >= 2 else None
            print(f"{c} B (alpha {ab}): " + ", ".join(f"{k} {v:+.2f}" for k, v in cb.items()))

    ctx = Context(); sit = {"start": 0.0, "shift": {}, "qb_from": 1}
    base = baseline_paths(sit, np.random.default_rng(0), 2000); b0 = ctx.sb(base).sum(axis=1).mean()
    cache = {}

    def titles(row, k):
        kk = (row.player_id, k)
        if kk not in cache:
            fam = row.grp
            if fam not in CF_OVER_OURS:
                cache[kk] = 0.0
            else:
                y = np.array([max(row[f"y{t}"], 0) for t in pm.SEASONS]) / CF_OVER_OURS[fam]
                y5 = np.r_[y, y[-1]]; price = max(y5.max(), 1e-6)
                pts = pick_effect((y5 / price)[None, :], np.array([price]), int(k), FAM2GRP.get(fam, fam), sit,
                                  ctx.cost, ctx.bar, ctx.rates)
                cache[kk] = (ctx.sb(base + pts).sum(axis=1).mean() - b0) * 100
        return cache[kk]

    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    rows = []
    try:
        for c in classes:
            te = uni[uni.cls == c].copy(); tr = uni[uni.cls < c]
            te["player_id"] = te.player_id.astype(str)
            te["y14"] = te[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
            Zs = S.reindex(te.player_id).fillna(0.0)
            Zb = B.reindex(te.player_id).fillna(0.0)
            models = [pm.fit(tr[tr.cls + t - 1 <= c - 1], "consensus_pre", t) for t in pm.SEASONS]
            vers = ["baseA", "A"] + (["A_fwd", "baseA_fwd"] if mult[(c, "A_fwd")] else []) + \
                   (["G", "B"] if c >= 2019 else []) + (["G_fwd", "B_fwd"] if c >= 2019 and mult[(c, "B_fwd")] else [])
            taken = {v: set() for v in vers}
            for k in range(1, int(te[te.drafted].pick.max()) + 1):
                fr = owner.get((c, k))
                if fr is None:
                    continue
                avail = (te.pick >= k).to_numpy()
                actual = te[te.pick == k]
                r = {"cls": c, "pick": k}
                if len(actual):
                    r["team_value"] = actual.y14.iloc[0]; r["team_titles"] = titles(actual.iloc[0], k)
                cand = te.copy(); lk = np.log(k + 10)
                fell = cand.on_board.astype(bool) & (cand.log_rank < lk)
                cand.loc[fell, "log_rank"] += pm.FALL_WEIGHT * (lk - cand.loc[fell, "log_rank"])
                pred = sum(m.predict(cand[f]) for m, f in models)
                cn, rc, vs = team_feats(te.player_id.reset_index(drop=True), c, fr, key, links, met)
                ZA = np.column_stack([Zs[f].to_numpy(float) for f in A_FEATS[:5]] + [cn, rc, vs])
                ZB = np.column_stack([ZA, Zb.to_numpy(float)])
                ml = {"baseA": np.ones(len(te)), "A": mult[(c, "A")](ZA)}
                if "A_fwd" in vers:
                    ml["A_fwd"] = mult[(c, "A_fwd")](ZA); ml["baseA_fwd"] = np.ones(len(te))
                if c >= 2019:
                    ml["G"] = mult[(c, "G")](Zb[["in_guide"]].to_numpy(float)); ml["B"] = mult[(c, "B")](ZB)
                    if "B_fwd" in vers:
                        ml["B_fwd"] = mult[(c, "B_fwd")](ZB); ml["G_fwd"] = mult[(c, "G_fwd")](Zb[["in_guide"]].to_numpy(float))
                picks = {}
                for v in vers:
                    score = np.where(avail & ~te.player_id.isin(taken[v]).to_numpy(), pred * ml[v], -np.inf)
                    i = int(np.argmax(score)); row = te.iloc[i]; picks[v] = i
                    taken[v].add(row.player_id)
                    r[f"{v}_value"] = row.y14; r[f"{v}_titles"] = titles(row, k); r[f"{v}_player"] = row.player_name
                # believed gain: under the stack's own multiplier, its pick vs the baseline's pick (value units)
                for s_, b_ in [("A", "baseA"), ("B", "G")]:
                    if s_ in picks:
                        r[f"{s_}_believed"] = pred[picks[s_]] * ml[s_][picks[s_]] - pred[picks[b_]] * ml[s_][picks[b_]]
                rows.append(r)
            print(f"{c} done", flush=True)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    ch = pd.DataFrame(rows)
    ch.to_parquet(OUT / "stacked_choice.parquet")
    report(ch)


def boot(d, rng):
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def report(ch):
    rng = np.random.default_rng(0)
    ch = ch.assign(rnd=pd.cut(ch.pick, [0, 32, 140, 262], labels=["R1", "R2-4", "R5-7"]))
    for s_, b_, lab in [("A", "baseA", "STACK A (2018-22): consensus x stack vs consensus"),
                        ("B", "G", "STACK B (2019-22): consensus x (stack + Brugler) vs consensus x in-guide")]:
        d0 = ch.dropna(subset=[f"{s_}_titles"])
        print(f"\n{lab}; picks {len(d0)}; choice changed {(d0[f'{s_}_player'] != d0[f'{b_}_player']).mean():.0%}")
        for unit, scale in [("titles", 1), ("value", 100)]:
            print(f"  {unit} ({'% of a title' if unit == 'titles' else '% of cap, 4 seasons'}) per pick, stack minus baseline")
            for name, g in [("All (PRIMARY)" if unit == "titles" else "All", d0), ("picks 101+", d0[d0.pick > 100])] + \
                    list(d0.groupby("rnd", observed=True)):
                d = ((g[f"{s_}_{unit}"] - g[f"{b_}_{unit}"]) * scale).to_numpy()
                lo, hi = boot(d, rng)
                print(f"    {name:16s} {d.mean():+.3f} (95% {lo:+.3f} to {hi:+.3f})")
            fw = d0.dropna(subset=[f"{s_}_fwd_{unit}"]) if f"{s_}_fwd_{unit}" in d0 else d0.iloc[:0]
            if len(fw):
                bf = "baseA_fwd" if s_ == "A" else "G_fwd"
                d = ((fw[f"{s_}_fwd_{unit}"] - fw[f"{bf}_{unit}"]) * scale).to_numpy()
                lo, hi = boot(d, rng)
                print(f"    forward-only     {d.mean():+.3f} (95% {lo:+.3f} to {hi:+.3f})  [drafts {sorted(fw.cls.unique())}]")
        bel = d0[f"{s_}_believed"].to_numpy() * 100
        real = ((d0[f"{s_}_value"] - d0[f"{b_}_value"]) * 100).to_numpy()
        print(f"  believed value gain per pick {bel.mean():+.3f} vs realized {real.mean():+.3f} (% of cap)")


if __name__ == "__main__":
    main()
