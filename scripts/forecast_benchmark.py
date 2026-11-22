"""How much better could the veteran value forecast be, and does the market know something it doesn't?

Setting = the free-agency tool's (veteran.Comps, skip 1): anchor season Y with full data through Y-1 plus weeks 1-11
of Y; forecast the player's value in seasons Y+1..Y+3 (the comps' y2..y4), % of cap, 0 if out of the league.

B  Market opinion. Past UFA signings 2015-2024 (fa_market.training(1): contract signed in March of S, data through
   S-1). Price residual r = log(actual APY / contract model's out-of-sample prediction), leave-signing-year-out.
   Does r predict the player's value in S..S+2 beyond the comps forecast (skip 0, no partial season)? Slope of
   (actual - forecast) on r, paired bootstrap by player; actual/forecast by residual tercile.
A  Headroom. Leave-one-season-out over anchor seasons 2014-2023 (test players' other anchors also dropped from
   training, like Comps' own exclusion). Models:
     comps        - current forecast (kNN + local-linear shift incl. weeks 1-11)
     gbm base     - LightGBM on the same inputs (group, age, v1, v2, games, p_share, p_gp)
     gbm history  - + quality and snap share separately (last two seasons), value 3 seasons back, career peak,
                    seasons played, experience, draft pick
     gbm market   - + current/previous contract (APY share of cap, rookie deal, years since signed, kind) and the
                    residual r of his last UFA deal (from B)
     ensemble     - mean of comps and the best GBM
   RMSE (x100), actual/forecast, MSE change vs comps with player bootstrap. All anchors and the tool's population
   (players signing a UFA deal in Y+1).
Pre-set rules:
   A: a model is a candidate to replace / blend with comps only if season-1 (y2) MSE falls vs comps with a 95%
      interval below 0 on all anchors AND on UFA signers, and calibration stays within 0.95-1.05. Candidates then
      need the bargain / realized-points check before adoption.
   B: market residual carries information if the y1-y3 pooled slope's 95% interval excludes 0 with the same sign in
      both halves (2015-19, 2020-24). Then test it as a forecast input (prior-deal residual, A's market set).
C  Why (B's follow-up): regress next-season quality q, snap share and games fraction on r, controlling for last
   season's q / share / games, age, group and the predicted log price. Quality = talent info; share / games = usage
   (team commitment, or health the team knows about).
INPUTS (Nov 21 2026 fix): the stored season values price quality with the talent model's SMOOTHED estimate
(talent_win), which uses later seasons' grades - including the seasons being forecast. `--inputs past` builds every
input (v1 / v2 for comps and the contract model, q1 / q2, v3, career peak, prior-deal residual) from the past-only
columns (value_ours_f, q_f: talent_post, grades through that season); outcomes stay on the smoothed values.
`--inputs smoothed` reproduces the first (leaky) run.
Run: .venv/bin/python scripts/forecast_benchmark.py [--inputs past|smoothed] [c]   (c: part C only)
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402

RNG = np.random.default_rng(7)
INPUTS = sys.argv[sys.argv.index("--inputs") + 1] if "--inputs" in sys.argv else "past"
VCOL, QCOL = ("value_ours_f", "q_f") if INPUTS == "past" else ("value_ours", "q")
TAG = "" if INPUTS == "past" else "_smoothed"
fm.VALUE_COL = VCOL


def make_comps():
    """veteran.comps() with the chosen input values (weeks 1-11 snaps merged the same way)."""
    A = vt.anchors(vt.season_table(), feat=VCOL)
    pp = []
    for Y in sorted(A.Y.unique()):
        Pt = vt.partial_snaps(int(Y))
        if len(Pt):
            pp.append(Pt[vt.P_FEATS].assign(Y=Y).reset_index(names="pfr_id"))
    A = A.merge(pd.concat(pp), on=["pfr_id", "Y"], how="left")
    A[vt.P_FEATS] = A[vt.P_FEATS].fillna(0.0)
    return vt.Comps(A)
N_SAMPLE = 6000
NB = 2000


def boot_ci(x, players, nb=NB):
    """Mean of x and 95% interval, resampling players."""
    df = pd.DataFrame({"x": x, "p": players})
    g = df.groupby("p").x.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(s), len(s))
        bs.append(s[i].sum() / c[i].sum())
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_slope(e, r, players, nb=NB):
    df = pd.DataFrame({"e": e, "r": r, "p": players})
    def slope(d):
        rr = d.r - d.r.mean()
        return (rr * (d.e - d.e.mean())).sum() / (rr ** 2).sum()
    ids = df.p.unique()
    by = {p: g for p, g in df.groupby("p")}
    rng = np.random.default_rng(1)
    bs = [slope(pd.concat([by[p] for p in rng.choice(ids, len(ids))])) for _ in range(nb // 4)]
    return slope(df), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ---------------------------------------------------------------- B: market residual
def price_residuals():
    """OOS price residual of every UFA signing (lag 1, leave-signing-year-out)."""
    D = fm.training(1)
    out = []
    for Y in sorted(D.Y.unique()):
        te = D[D.Y == Y].copy()
        m = fm.fit(D[D.Y != Y], feats=fm.FEATS_BASE)["log_apy"]
        te["pred"] = np.exp(m.predict(te[fm.FEATS_BASE]))
        out.append(te)
    P = pd.concat(out)
    P["r"] = np.log(P.apy / P.pred)
    return P[["pfr_id", "Y", "apy", "pred", "r", "grp", "age"]]


def part_b(C, R):
    A = C.A.set_index(["pfr_id", "Y"])
    rows = []
    for x in R[R.Y <= 2024].itertuples():
        if (x.pfr_id, x.Y) not in A.index:
            continue
        a = A.loc[(x.pfr_id, x.Y)]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        row = {**a.to_dict(), "pfr_id": x.pfr_id, "Y": x.Y, "p_share": np.nan, "p_gp": np.nan}
        for k in (1, 2, 3):
            if pd.notna(row[f"y{k}"]):
                rows.append({"pfr_id": x.pfr_id, "S": x.Y, "k": k, "r": x.r, "apy": x.apy,
                             "f": C.outcomes(row, k)[0].mean(), "y": row[f"y{k}"]})
    B = pd.DataFrame(rows)
    B["e"] = B.y - B.f
    print(f"B. Market residual -> value beyond the forecast (UFA signings 2015-24, {B.pfr_id.nunique()} players)")
    print(f"   residual sd {B.drop_duplicates(['pfr_id', 'S']).r.std():.2f} (log units)")
    out = []
    for lab, d in [("all", B), ("2015-19", B[B.S <= 2019]), ("2020-24", B[B.S >= 2020])] + \
                  [(f"season {k}", B[B.k == k]) for k in (1, 2, 3)]:
        s, lo, hi = boot_slope(d.e.to_numpy(), d.r.to_numpy(), d.pfr_id.to_numpy())
        out.append({"slice": lab, "n": len(d), "slope (x100 per log unit)": f"{s * 100:+.3f} ({lo * 100:+.3f} to {hi * 100:+.3f})",
                    "mean forecast x100": d.f.mean() * 100})
    print(pd.DataFrame(out).round(3).to_string(index=False))
    B["terc"] = pd.qcut(B.r, 3, labels=["paid below model", "middle", "paid above model"])
    t = B.groupby("terc", observed=True).agg(n=("y", "size"), r=("r", "mean"), actual=("y", "sum"), fc=("f", "sum"))
    t["actual/forecast"] = t.actual / t.fc
    print(t[["n", "r", "actual/forecast"]].round(3).to_string())
    return B


# ---------------------------------------------------------------- A: headroom
def extra_features(A, R):
    v = vt.season_table().set_index(["pfr_id", "season"])
    Q, S, V = v[QCOL].to_dict(), v.share.to_dict(), v[VCOL].to_dict()
    seasons = vt.season_table().groupby("pfr_id").season.apply(sorted).to_dict()
    pl = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    pick = dict(zip(pl.pfr_id, pl.draft_pick))
    H = fm.history()
    rres = R.set_index(["pfr_id", "Y"]).r.to_dict()
    rows = []
    for p, Y in zip(A.pfr_id, A.Y):
        ss = [s for s in seasons.get(p, []) if s < Y]
        h = [x for x in H.get(p, []) if x[0] <= Y]  # contracts signed by season Y (known in-season)
        ufa = [x for x in H.get(p, []) if x[0] <= Y and x[1] == "UFA" and (p, x[0]) in rres]
        rows.append({
            "q1": Q.get((p, Y - 1), np.nan), "q2": Q.get((p, Y - 2), np.nan),
            "sh1": S.get((p, Y - 1), 0.0), "sh2": S.get((p, Y - 2), 0.0),
            "v3": V.get((p, Y - 3), 0.0), "vmax": max([V.get((p, s), 0) for s in ss], default=0.0),
            "nseas": len(ss), "pick": pick.get(p) if pd.notna(pick.get(p)) else 300.0,
            "c_apy": h[-1][2] if h else np.nan, "c_rookie": int(bool(h) and h[-1][1] in ("Drafted", "UDFA")),
            "c_age": Y - h[-1][0] if h else np.nan, "c_ext": int(bool(h) and h[-1][1] == "Extension"),
            "c_yrs": h[-1][3] if h else np.nan,
            "prev_r": rres[(p, ufa[-1][0])] if ufa else np.nan,
        })
    return pd.concat([A.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


BASE = ["grp_c", "age", "v1", "v2f", "g1", "p_share", "p_gp"]
HIST = BASE + ["q1", "q2", "sh1", "sh2", "v3", "vmax", "nseas", "exp", "pick"]
MARKET = HIST + ["c_apy", "c_rookie", "c_age", "c_ext", "c_yrs", "prev_r"]
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def part_a(C, R):
    import lightgbm as lgb
    A = extra_features(C.A, R)
    A["grp_c"] = pd.Categorical(A.grp, categories=fm.GROUPS)
    H = fm.history()
    signers = {(p, x[0]) for p, h in H.items() for x in h if x[1] == "UFA"}
    pool = A[A.Y.between(2014, 2023) & A.y2.notna()]
    T = pool.sample(min(N_SAMPLE, len(pool)), random_state=5)
    T = T.assign(ufa_next=[(p, y + 1) in signers for p, y in zip(T.pfr_id, T.Y)])
    res = []
    for k in (2, 3, 4):
        d = T[T[f"y{k}"].notna()].copy()
        d["comps"] = [C.outcomes({**r, "v2f": r["v2f"]}, k)[0].mean() for r in d[["pfr_id", "Y", "grp", "age", "v1", "v2", "v2f", "g1", "p_share", "p_gp"]].to_dict("records")]
        for name, feats in [("gbm base", BASE), ("gbm history", HIST), ("gbm market", MARKET)]:
            d[name] = np.nan
            for Y, te in d.groupby("Y"):
                tr = A[(A.Y != Y) & ~A.pfr_id.isin(te.pfr_id) & A[f"y{k}"].notna()]
                m = lgb.LGBMRegressor(**PARAMS).fit(tr[feats], tr[f"y{k}"])
                d.loc[te.index, name] = m.predict(te[feats]).clip(min=0)
        res.append(d.assign(k=k))
    D = pd.concat(res)
    best = min(["gbm base", "gbm history", "gbm market"],
               key=lambda c: ((D[D.k == 2][c] - D[D.k == 2].y2) ** 2).mean())
    D["ensemble"] = (D.comps + D[best]) / 2
    models = ["comps", "gbm base", "gbm history", "gbm market", "ensemble"]
    print(f"\nA. Forecast headroom (leave-season-out, anchors 2014-23, skip 1); ensemble = comps + {best}")
    for pop, sel in [("all anchors", lambda x: x), ("UFA signers next March", lambda x: x[x.ufa_next])]:
        rows = []
        for k in (2, 3, 4):
            d = sel(D[D.k == k])
            a = d[f"y{k}"].to_numpy()
            e0 = (a - d.comps.to_numpy()) ** 2
            for m in models:
                e = (a - d[m].to_numpy()) ** 2
                r = {"season": k - 1, "model": m, "n": len(d), "RMSE x100": np.sqrt(e.mean()) * 100,
                     "R2": 1 - e.sum() / ((a - a.mean()) ** 2).sum(), "actual/forecast": a.sum() / d[m].sum()}
                if m != "comps":
                    s, lo, hi = boot_ci((e - e0) * 1e4, d.pfr_id.to_numpy())
                    r["MSE vs comps x1e4"] = f"{s:+.3f} ({lo:+.3f} to {hi:+.3f})"
                rows.append(r)
        print(f"  {pop}")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    D.to_parquet(vt.OUT / f"forecast_benchmark{TAG}.parquet")
    # which added inputs matter (gain importance, season-1 model on all data)
    tr = A[A.y2.notna()]
    m = lgb.LGBMRegressor(**PARAMS, importance_type="gain").fit(tr[MARKET], tr.y2)
    imp = pd.Series(m.feature_importances_, MARKET)
    print("  gain share, gbm market season 1:", (imp / imp.sum()).sort_values(ascending=False).round(3).to_dict())
    return D


def part_c(R):
    R = R[R.Y <= 2024]
    v = vt.season_table().set_index(["pfr_id", "season"])
    rows = []
    for x in R.itertuples():
        g = lambda s, c: v[c].get((x.pfr_id, s), np.nan)  # noqa: E731
        rows.append(dict(p=x.pfr_id, r=x.r, grp=x.grp, age=x.age, lp=np.log(x.pred), q0=g(x.Y - 1, QCOL),
                         sh0=g(x.Y - 1, "share"), gf0=g(x.Y - 1, "gfrac"), q=g(x.Y, "q"), sh=g(x.Y, "share"),
                         gf=g(x.Y, "gfrac")))
    D = pd.DataFrame(rows)
    X = ["r", "q0", "sh0", "gf0", "age", "lp"]
    print("\nC. What the price residual predicts in the first season (OLS + group dummies, player bootstrap)")
    for y, lab in [("q", "quality"), ("sh", "snap share"), ("gf", "games fraction")]:
        d = D.dropna(subset=[y] + X).reset_index(drop=True)
        dummies = pd.get_dummies(d.grp, drop_first=True).astype(float)
        M = np.column_stack([np.ones(len(d)), d[X].to_numpy(), dummies.to_numpy()])
        b = np.linalg.lstsq(M, d[y], rcond=None)[0][1]
        by = d.groupby("p").indices
        ids = list(by)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(300):
            i = np.concatenate([by[p] for p in rng.choice(ids, len(ids))])
            bs.append(np.linalg.lstsq(M[i], d[y].to_numpy()[i], rcond=None)[0][1])
        print(f"  {lab}: n {len(d)}, per log unit {b:+.3f} ({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}),"
              f" per 1 sd of r = {b * R.r.std() / d[y].std():+.2f} sd of the outcome")


def main():
    pd.set_option("display.width", 220)
    print(f"inputs: {INPUTS} ({VCOL}, {QCOL})")
    if "c" in sys.argv[1:]:
        part_c(pd.read_parquet(vt.OUT / f"ufa_price_residuals{TAG}.parquet"))
        return
    C = make_comps()
    R = price_residuals()
    R.to_parquet(vt.OUT / f"ufa_price_residuals{TAG}.parquet")
    part_b(C, R)
    part_a(C, R)
    part_c(R)


if __name__ == "__main__":
    main()
