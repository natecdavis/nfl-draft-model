"""Trade suggestions, phases 1 and 3: what veterans cost in picks, and does the model pick good buys at that price?

Sample: offseason trades (Feb 1 - Sep 1 of season Y, 2014-2025) of exactly one player for draft picks
(data/raw/trades.parquet; a row with no pick_season is a player, every other row a pick). Player = forecast anchor
at Y (value in Y-1 or Y-2). Price (the trade market's chart, src/market.py, #1 pick = 1): picks the acquiring team
gave minus picks it got back, floored at 0. This year's picks before May 1 at their number; later picks
d^years x the round's mean chart value (d = 0.48 for a 1st, 0.63 otherwise, as market.price).
Inputs known at the trade, no look-ahead: past-only comps (value_ours_f features, no in-season snaps); contract in
force = latest one signed before Y (fa_market.history: APY share of cap at signing, deflated 7% a year; years
left = signed + years - Y, 1-5; guarantees left = guarantees - APY x seasons already played, >= 0).

Phase 1  market price model: price ~ forecast value over the remaining contract, its cost, age, years left, QB, rookie
         deal (Poisson GLM, log link: zero prices allowed). Leave-trade-year-out. 80% ranges from the training folds'
         price / predicted ratios; contract_ranges.py rule: coverage 0.77-0.83 overall and 0.72-0.88 in every segment
         with n >= 30 (age band, forecast tercile, rookie deal), else point estimate only.
Phase 3  the gate, stated before running:
         edge  = model points of the player to the acquirer over his remaining contract (comps futures, contract cost
                 charged at COST_RATE; veteran.points) minus model points of the picks paid (trade.package_points,
                 'Middle of the pack' situation), both over 5 seasons
         H     = seasons observed (min(years left, 2025 - Y + 1, 3)); realized value = value_ours with the acquiring
                 team (0 if he's gone), predicted value over the same H
         G1  realized value over H ~ predicted value over H + log price: the model's coefficient > 0 (95% CI)
             - the model knows something the price doesn't
         G2  top third of trades by edge: realized edge (edge + rate x (realized - predicted value over H)) > 0
             (bootstrap 95% CI by trade), and realized / predicted value >= 0.85 (traded veterans overall 0.94)
         Pass both -> build suggestions (phases 2, 4); else only show the market price in the evaluator.
Run: .venv/bin/python scripts/vet_trade_market.py   (writes data/derived/vet_trade_market.parquet)
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import market as mk  # noqa: E402
import roster_need as rn  # noqa: E402
import veteran as vt  # noqa: E402

YEARS = (2014, 2025)
OUT = vt.OUT / "vet_trade_market.parquet"
GROWTH = 1.07


def trades():
    t = pd.read_parquet(vt.RAW / "trades.parquet")
    t["date"] = pd.to_datetime(t.trade_date)
    t["Y"] = t.date.dt.year
    t = t[(t.date.dt.month >= 2) & (t.date < pd.to_datetime(t.Y.astype(str) + "-09-01")) & t.Y.between(*YEARS)]
    t["is_pl"] = t.pick_season.isna()
    g = t.groupby("trade_id").agg(n_pl=("is_pl", "sum"), n=("is_pl", "size"))
    one = g[(g.n_pl == 1) & (g.n > 1)].index
    t = t[t.trade_id.isin(one)]
    p = chart = mk.chart()["fit"]
    rows = []
    for tid, d in t.groupby("trade_id"):
        pl = d[d.is_pl].iloc[0]
        if pd.isna(pl.pfr_id) or not pl.pfr_id:
            continue
        frm, to, Y, date = pl.gave, pl.received, int(pl.Y), pl.date
        pre = date < pd.Timestamp(f"{Y}-05-01")
        price, picks_paid, picks_back = 0.0, [], []
        for r in d[~d.is_pl].itertuples():
            yrs = int(r.pick_season) - Y
            rnd = int(r.pick_round) if pd.notna(r.pick_round) else 7
            if yrs == 0 and pre and pd.notna(r.pick_number):
                val, pk = float(mk.v(r.pick_number, p)), ("now", int(r.pick_number))
            else:
                yrs = max(yrs, 1)
                d_ = p["d1r1"] if rnd == 1 else p["d1l"]
                val = d_ ** yrs * float(mk.v(np.arange(32 * (rnd - 1) + 1, 32 * rnd + 1), p).mean())
                pk = ("fut", rnd, yrs)
            if r.gave == to:
                price += val
                picks_paid.append(pk)
            else:
                price -= val
                picks_back.append(pk)
        rows.append({"trade_id": tid, "pfr_id": pl.pfr_id, "player": pl.pfr_name, "Y": Y, "date": date,
                     "from": frm, "to": to, "price_raw": price, "price": max(price, 0.0),
                     "picks_paid": picks_paid, "picks_back": picks_back})
    return pd.DataFrame(rows)


def contract_at(p, Y):
    h = [x for x in fm.history().get(p, []) if x[0] < Y]
    if not h:
        return None
    ys, kind, apy, yrs, gtd, team = h[-1]
    left = ys + yrs - Y
    if left < 1:
        left = 1  # traded in his final year / option or tag pending: one season
    return {"apy": apy / GROWTH ** (Y - ys), "yrs_left": int(min(left, 5)),
            "gtd_left": max(gtd - apy * (Y - ys), 0.0) / GROWTH ** (Y - ys),
            "rookie": int(kind in ("Drafted", "UDFA"))}


def build():
    import pick_value as pv
    import trade as tr
    from pick_value import COST_RATE
    T = trades()
    A = vt.anchors(vt.season_table(), feat="value_ours_f")
    C = vt.Comps(A)
    D = T.merge(A, on=["pfr_id", "Y"])
    print(f"one-player-for-picks offseason trades {YEARS[0]}-{YEARS[1]}: {len(T)}; with a forecast anchor: {len(D)}")
    rates = json.load(open(vt.OUT / "bridge.json"))["preseason"]
    ctx = tr.Context()
    sit = pv.SITUATIONS["Middle of the pack"]
    rng = np.random.default_rng(0)
    base = pv.baseline_paths(sit, rng, 4000)
    pick_cache = {}

    def pick_pts(pk):
        if pk not in pick_cache:
            P = tr.Pick(pk[1]) if pk[0] == "now" else tr.Pick(rnd=pk[1], year=min(pk[2], 2), src="typical")
            pick_cache[pk] = float(tr.package_points(ctx, sit, [P], base, rng).sum(axis=1).mean())
        return pick_cache[pk]

    played = rn._played_for()
    V = vt.season_table().set_index(["pfr_id", "season"]).value_ours.to_dict()
    rows = []
    for r in D.to_dict("records"):
        c = contract_at(r["pfr_id"], r["Y"])
        if c is None:
            continue
        yrs = c["yrs_left"]
        Vp, Sp = C.paths(r, yrs, exclude=True, skip=0)
        val, cost, on = vt.team_seasons(Vp, yrs, c["apy"], c["gtd_left"], growth=vt.CAP_GROWTH)
        pts = vt.points(val, Sp, cost, on, r["grp"] == "QB")
        rate = rates["qb"] if r["grp"] == "QB" else rates["nonqb"]
        H = int(min(yrs, vt.LAST - r["Y"] + 1, 3))
        to = rn.team(r["to"])
        real = [V.get((r["pfr_id"], r["Y"] + k), 0.0) if to in played.get((r["pfr_id"], r["Y"] + k), set()) else 0.0
                for k in range(H)]
        paid = sum(pick_pts(pk) for pk in r["picks_paid"]) - sum(pick_pts(pk) for pk in r["picks_back"])
        rows.append({**{k: r[k] for k in ["trade_id", "pfr_id", "player", "Y", "from", "to", "price", "price_raw",
                                          "grp", "age", "v1", "v2", "g1"]}, **c,
                     "fc_value": float(Vp.mean(axis=0).sum()), "fc_cost": float(c["apy"] * yrs),
                     "player_pts": float(pts.sum(axis=1).mean()), "pick_pts": paid,
                     "edge": float(pts.sum(axis=1).mean()) - paid, "H": H,
                     "pred_H": float(val[:, :H].mean(axis=0).sum()), "real_H": float(sum(real)), "rate": rate})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT)
    return out


def phase1(D):
    D = D.copy()
    D["qb"] = (D.grp == "QB").astype(int)
    D["lv"] = np.log(D.fc_value * 100 + 0.1)
    D["lc"] = np.log(D.fc_cost * 100 + 0.1)
    f = "price ~ lv + lc + age + yrs_left + qb + rookie"
    pred = pd.Series(index=D.index, dtype=float)
    lo = pd.Series(index=D.index, dtype=float)
    hi = pd.Series(index=D.index, dtype=float)
    for Y in sorted(D.Y.unique()):
        tr_, te = D[D.Y != Y], D.Y == Y
        m = smf.glm(f, tr_, family=sm.families.Poisson()).fit(scale="X2")
        # ratio distribution from the training rows' own leave-year-out-free fit (in-sample on training folds)
        ratio = tr_.price / m.predict(tr_)
        pred[te] = m.predict(D[te])
        lo[te], hi[te] = pred[te] * np.quantile(ratio, 0.1), pred[te] * np.quantile(ratio, 0.9)
    D["pred_price"], D["lo"], D["hi"] = pred, lo, hi
    D["cov"] = (D.price >= D.lo) & (D.price <= D.hi)
    full = smf.glm(f, D, family=sm.families.Poisson()).fit(scale="X2")
    dev0 = ((D.price - D.price.mean()) ** 2).sum()
    print(f"\nPHASE 1 market price ({len(D)} trades; price in #1-pick units, median {D.price.median():.3f}, "
          f"zero {(D.price == 0).mean():.0%})")
    print("  terms (z):", {k: round(v, 1) for k, v in full.tvalues.drop('Intercept').items()})
    print(f"  leave-year-out: corr(pred, price) {np.corrcoef(D.pred_price, D.price)[0, 1]:.3f}; "
          f"R2 {1 - ((D.price - D.pred_price) ** 2).sum() / dev0:.3f}; "
          f"rank corr {D.pred_price.rank().corr(D.price.rank()):.3f}")
    segs = {"age < 27": D.age < 27, "age 27-29": D.age.between(27, 30, inclusive="left"), "age 30+": D.age >= 30,
            "rookie deal": D.rookie == 1, "veteran deal": D.rookie == 0}
    terc = pd.qcut(D.fc_value, 3, labels=["low", "mid", "high"])
    for t_ in ["low", "mid", "high"]:
        segs[f"forecast {t_}"] = terc == t_
    ok = 0.77 <= D["cov"].mean() <= 0.83
    print(f"  80% range coverage overall {D['cov'].mean():.2f} (rule 0.77-0.83)")
    for k, s in segs.items():
        n, c = int(s.sum()), D.loc[s, "cov"].mean()
        flag = "" if n < 30 or 0.72 <= c <= 0.88 else "  <- outside 0.72-0.88"
        ok &= n < 30 or 0.72 <= c <= 0.88
        print(f"    {k:16s} n {n:3d} coverage {c:.2f}{flag}")
    print(f"  ranges {'PASS' if ok else 'FAIL: point estimate only'}")
    return D, full


def phase3(D):
    rng = np.random.default_rng(1)
    d = D[D.H >= 1].copy()
    d["lp"] = np.log(d.price + 0.005)
    m = smf.ols("real_H ~ pred_H + lp + C(H)", d).fit(cov_type="HC1")
    ci = m.conf_int().loc["pred_H"]
    g1 = ci[0] > 0
    print(f"\nPHASE 3 gate ({len(d)} trades with observed seasons)")
    print(f"  overall realized / predicted value over H: {d.real_H.sum() / d.pred_H.sum():.2f}")
    print(f"  G1 realized ~ predicted + log price: predicted {m.params['pred_H']:.2f} ({ci[0]:.2f} to {ci[1]:.2f}), "
          f"log price {m.params['lp']:.4f} ({m.conf_int().loc['lp'][0]:.4f} to {m.conf_int().loc['lp'][1]:.4f}) "
          f"-> {'pass' if g1 else 'FAIL'}")
    d["real_edge"] = d.edge + d.rate * (d.real_H - d.pred_H) * 100
    d["terc"] = pd.qcut(d.edge.rank(method="first"), 3, labels=["bottom", "middle", "top"])
    top = d[d.terc == "top"]
    bs = [top.iloc[rng.integers(0, len(top), len(top))].real_edge.mean() for _ in range(4000)]
    cal = top.real_H.sum() / top.pred_H.sum()
    bc = [(lambda s: s.real_H.sum() / s.pred_H.sum())(top.iloc[rng.integers(0, len(top), len(top))]) for _ in range(4000)]
    g2 = np.percentile(bs, 2.5) > 0 and cal >= 0.85
    print("  by edge third (points over 5 seasons; edge = model player points - model pick points):")
    print(d.groupby("terc", observed=True).agg(n=("edge", "size"), edge=("edge", "mean"), real_edge=("real_edge", "mean"),
                                               price=("price", "mean"), pred_H=("pred_H", "mean"),
                                               real_H=("real_H", "mean")).round(3).to_string())
    print(f"  G2 top third realized edge {top.real_edge.mean():+.2f} ({np.percentile(bs, 2.5):+.2f} to "
          f"{np.percentile(bs, 97.5):+.2f}); realized / predicted {cal:.2f} ({np.percentile(bc, 2.5):.2f} to "
          f"{np.percentile(bc, 97.5):.2f}) -> {'pass' if g2 else 'FAIL'}")
    print("  top-third trades:")
    print(top.sort_values("edge", ascending=False)[["player", "Y", "grp", "age", "price", "edge", "real_edge",
                                                    "pred_H", "real_H"]].head(20).round(3).to_string(index=False))
    print(f"\nGATE: {'PASS - build suggestions' if g1 and g2 else 'FAIL - market price in the evaluator only'}")
    return d


def main():
    pd.set_option("display.width", 220)
    D = pd.read_parquet(OUT) if "--cached" in sys.argv and OUT.exists() else build()
    D, _ = phase1(D)
    phase3(D)


if __name__ == "__main__":
    main()
