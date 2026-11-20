"""Does the team-needs factor improve the free-agent ranking? (adoption check for the room adjustment)

Factor (scripts/fa_room_test.py, signees' own value over seasons 1-2): own value per 1.0 of forecast falls with room
fullness: f = 1 + slope_g x (ret - mean ret of rooms that signed), slope = interaction / main effect, OL and LB
their own, other groups pooled; clipped to [0.6, 1.25]. Fit leaving the signing year out.
Check, same contracts as scripts/fa_bargain_check.py (UFA 2015-23 finished by 2025, out-of-sample predicted APY):
  - bargain score = forecast value per season (x f with the factor) / predicted APY
  - realized value per $1 of actual cost by score tercile, high - low spread, and per-contract correlation, with and
    without the factor; and realized / forecast value by fullness tercile (does the factor remove the bias?)
  - fullness for a re-signing excludes the player himself (he counts as a returner in ret).
Pre-set: adopt as the default if the spread and the correlation both improve and the full-room bias shrinks;
otherwise an optional setting.
Run: .venv/bin/python scripts/fa_need_check.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import roster_need as rn  # noqa: E402
import veteran as vt  # noqa: E402

GROUP_SPLIT = lambda g: g if g in ("OL", "LB") else "other"  # noqa: E731


def slopes(U, exclude_year=None, n=2, last=2024):
    d = U[(U.year <= last) & (U.year != exclude_year)].copy()
    m0 = d[d[f"Efa_{n}"] > 0].ret.mean()
    d["rc"] = d.ret - m0
    d["sp"] = d.g.map(GROUP_SPLIT)
    for sp in ("OL", "LB", "other"):
        d[f"x_{sp}"] = d[f"Efa_{n}"] * d.rc * (d.sp == sp)
        d[f"E_{sp}"] = d[f"Efa_{n}"] * (d.sp == sp)
    m = smf.ols(f"ownfa_{n} ~ E_OL + E_LB + E_other + x_OL + x_LB + x_other + ret + v_last + Eall_{n} + C(g):C(year)",
                d).fit()
    return {sp: m.params[f"x_{sp}"] / m.params[f"E_{sp}"] for sp in ("OL", "LB", "other")}, m0


def factor(g, ret, sl, m0):
    return float(np.clip(1 + sl[GROUP_SPLIT(g)] * (ret - m0), 0.6, 1.25))


def main():
    pd.set_option("display.width", 200)
    U = pd.read_parquet(vt.OUT / "fa_room_test.parquet")
    sl_all, m0_all = slopes(U)
    print("slopes (share of value per 1.0 of fullness), all years:", {k: round(v, 3) for k, v in sl_all.items()},
          "mean ret", round(m0_all, 3))
    F = pd.read_parquet(vt.OUT / "vet_signings_ufa.parquet")
    P = pd.read_parquet(vt.OUT / "contract_predict_lag1.parquet")[["pfr_id", "Y", "apy_pred"]]
    P["apy_pred"] *= fm.SMEAR[1]
    F = F.merge(P, on=["pfr_id", "Y"])
    # fullness of the signing team's room, excluding the player himself if he was a returner
    _, ps, _, _ = rn.load()
    se_prev = ps.groupby(["season", "team", "pfr_player_id"]).se.sum()
    ret_se = U.set_index(["year", "team", "g"]).ret_se
    gmean = U.groupby("g").ret_se.mean()
    F["ret"] = [(float(ret_se.get((y, t, g), np.nan)) - float(se_prev.get((y - 1, t, p), 0.0))) / gmean.get(g, np.nan)
                for y, t, g, p in zip(F.Y, F.team, F.grp, F.pfr_id)]
    F = F[F.ret.notna() & F.grp.isin(gmean.index)].copy()
    fits = {y: slopes(U, exclude_year=y) for y in F.Y.unique()}
    F["f"] = [factor(g, r, *fits[y]) for g, r, y in zip(F.grp, F.ret, F.Y)]
    C = vt.comps()
    base, adj = [], []
    for r in F.itertuples():
        V, S = C.paths(r._asdict(), int(r.n))
        for out, mult in ((base, 1.0), (adj, r.f)):
            val, _, _ = vt.team_seasons(V * mult, int(r.n), r.apy, r.gtd)
            out.append(val.sum(axis=1).mean() / r.n)
    F["fc"], F["fc_need"] = base, adj
    F["real_w"] = sum(np.where(F.n >= k, F[f"w{k}"].fillna(0), 0) for k in range(1, 6))
    F["cost"] = F.apy * F.kept + F.dead
    F["real_yr"] = F.real_w / F.n
    F["terc_ret"] = pd.qcut(F.ret.rank(method="first"), 3, labels=["thin", "typical", "full"])
    print(f"\n{len(F)} contracts; factor range {F.f.min():.2f}-{F.f.max():.2f}, sd {F.f.std():.3f}")
    print("realized / forecast value per season, by fullness of the signing room:")
    print(F.groupby("terc_ret", observed=True).apply(lambda d: pd.Series({
        "n": len(d), "without factor": d.real_yr.sum() / d.fc.sum(), "with factor": d.real_yr.sum() / d.fc_need.sum()}),
        include_groups=False).round(3).to_string())
    print("by group x fullness (OL, LB):")
    for g in ("OL", "LB"):
        d0 = F[F.grp == g]
        print(" ", g, d0.groupby("terc_ret", observed=True).apply(lambda d: (round(d.real_yr.sum() / d.fc.sum(), 2),
              round(d.real_yr.sum() / d.fc_need.sum(), 2), len(d)), include_groups=False).to_dict())
    rng = np.random.default_rng(0)
    G = F[F.cost > 0]
    for col, lab in (("fc", "without factor"), ("fc_need", "with factor")):
        F["score"] = F[col] / F.apy_pred
        F["t"] = pd.qcut(F.score.rank(method="first"), 3, labels=["low", "mid", "high"])
        v = F.groupby("t", observed=True).apply(lambda d: d.real_w.sum() / d.cost.sum(), include_groups=False)
        hi, lo = F[F.t == "high"], F[F.t == "low"]
        bs = []
        for _ in range(1000):
            a, b = hi.sample(len(hi), replace=True, random_state=rng.integers(1e9)), lo.sample(len(lo), replace=True, random_state=rng.integers(1e9))
            bs.append(a.real_w.sum() / a.cost.sum() - b.real_w.sum() / b.cost.sum())
        c = np.corrcoef(G[col] / G.apy_pred, (G.real_w / G.cost).clip(upper=5))[0, 1]
        cr = np.corrcoef(F[col], F.real_yr)[0, 1]
        print(f"{lab:15s} value per $1 by tercile {v.round(3).tolist()}  high - low {v.iloc[-1] - v.iloc[0]:+.3f} "
              f"({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}); corr(score, value per $) {c:.3f}; "
              f"corr(forecast, realized value) {cr:.3f}")
    # paired: does the factor improve the per-contract forecast of realized value? (squared error)
    e0, e1 = (F.real_yr - F.fc) ** 2, (F.real_yr - F.fc_need) ** 2
    dd = e1 - e0
    bs = [dd.sample(len(dd), replace=True, random_state=k).mean() for k in range(1000)]
    print(f"MSE change with factor: {dd.mean() * 1e4:+.4f} (x1e-4; 95% {np.percentile(bs, 2.5) * 1e4:+.4f} to "
          f"{np.percentile(bs, 97.5) * 1e4:+.4f}); RMSE {np.sqrt(e0.mean()) * 100:.3f} -> {np.sqrt(e1.mean()) * 100:.3f} % of cap")
    json_out = {"slopes": sl_all, "mean_ret": m0_all, "group_mean_se": gmean.to_dict()}
    pd.Series(json_out).to_json(vt.OUT / "fa_need_factor.json")


if __name__ == "__main__":
    main()
