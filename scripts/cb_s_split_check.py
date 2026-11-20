"""Should the free-agency / trade tools split DB into CB and S? (they price, forecast and predict contracts for one DB group)

Checks, all with the current pooled-DB models, by the player's OTC contract position (CB / S):
  1 contract model: out-of-sample actual / predicted APY for DB signings (scripts/contract_predict.py, both lags);
    a split matters if CBs and safeties are mispriced in opposite directions
  2 forecast: actual / forecast value, random anchors (other players' comps) and UFA signings with the team
  3 team strength per 1% of curve value, CB vs S (scripts/vet_currency_check.py design: rating ~ value by group,
    season means removed, team-clustered); equal slopes = the pooled curve is fine as a currency
Draft side (already tested, README "Pay within OL and DB"): CB x1.11 / S x0.89 pay factors failed the choice test.
Run: .venv/bin/python scripts/cb_s_split_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["scripts", "src"]
import veteran as vt  # noqa: E402

rng = np.random.default_rng(0)


def pos_map():
    c = pd.read_parquet(vt.RAW / "contracts.parquet", columns=["gsis_id", "position", "year_signed"])
    c = c[c.position.isin(["CB", "S"])].sort_values("year_signed")
    pl = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    # most common OTC position over his contracts (ties -> latest)
    m = c.assign(pfr=c.gsis_id.map(g2p)).dropna(subset=["pfr"]).groupby("pfr").position.agg(lambda x: x.value_counts().index[0])
    return m.to_dict()


def ci_ratio(d, a, f):
    r = d[a].sum() / d[f].sum()
    bs = [(lambda s: s[a].sum() / s[f].sum())(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(1000)]
    return f"{r:.2f} ({np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}) n={len(d)}"


def main():
    pos = pos_map()
    print("1. Contract model, DB signings, actual / predicted APY (out of sample):")
    for lag in (1, 2):
        P = pd.read_parquet(vt.OUT / f"contract_predict_lag{lag}.parquet")
        P = P[P.grp == "DB"].assign(pos=lambda d: d.pfr_id.map(pos)).dropna(subset=["pos"])
        P["apy_pred"] = P.apy_pred * {1: 1.128, 2: 1.185}[lag]  # saved unscaled
        print(f"  lag {lag}: " + "; ".join(f"{p} {ci_ratio(P[P.pos == p], 'apy', 'apy_pred')}" for p in ("CB", "S")))
        big = P[P.apy_pred >= 0.03]
        print(f"     predicted >= 3% of cap: " + "; ".join(f"{p} {ci_ratio(big[big.pos == p], 'apy', 'apy_pred')}" for p in ("CB", "S")))
    print("\n2. Forecast (comps from all DBs), actual / forecast value:")
    A = pd.read_parquet(vt.OUT / "vet_anchors.parquet")
    S = A[(A.grp == "DB")].assign(pos=lambda d: d.pfr_id.map(pos)).dropna(subset=["pos"]).sample(2500, random_state=3)
    C = vt.comps()
    F = pd.DataFrame([C.forecast(r) for r in S.to_dict("records")], index=S.index)
    S = S.join(F)
    for k in (1, 3):
        d = S[S[f"y{k}"].notna()]
        print(f"  random DB anchors, season {k}: " + "; ".join(f"{p} {ci_ratio(d[d.pos == p], f'y{k}', f'f{k}')}" for p in ("CB", "S")))
    U = pd.read_parquet(vt.OUT / "vet_signings_ufa.parquet")
    U = U[U.grp == "DB"].assign(pos=lambda d: d.pfr_id.map(pos)).dropna(subset=["pos"])
    print("  UFA signings, season 1 with team: " + "; ".join(f"{p} {ci_ratio(U[U.pos == p], 'w1', 'f1')}" for p in ("CB", "S")))
    print("\n3. Team strength per 1% of curve value:")
    import age_pricing as ap
    ps = ap.player_seasons()
    v = vt.season_table()
    tot = ps.groupby(["pfr_player_id", "season"]).share.transform("sum")
    ps["w"] = np.where(tot > 0, ps.share / tot, 0)
    ps = ps.merge(v[["pfr_id", "season", "value_ours", "group"]].rename(columns={"pfr_id": "pfr_player_id"}),
                  on=["pfr_player_id", "season"], how="left")
    ps["curve"] = ps.value_ours.fillna(0) * ps.w * 100
    ps["b"] = np.where(ps.group == "QB", "qb", np.where(ps.group == "DB", ps.pfr_player_id.map(pos).fillna("DBx"), "other"))
    ps.loc[ps.b == "DBx", "b"] = "other"
    rat = ap.ratings()
    ap.fit(ps, rat, "curve", "b", "curve value: CB vs S vs other non-QB vs QB", test=["CB", "S"])


if __name__ == "__main__":
    main()
