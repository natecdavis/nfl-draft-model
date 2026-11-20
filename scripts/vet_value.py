"""Free agency / trades, step 1: value one veteran over a contract, on the same yardstick as draft picks.

Season value (scripts/vet_season_values.py): snap share x veteran pay curve(share, darko quality) x games
fraction, this project's units (share of cap) - the same pricing as the draft label's seasons 1-4.

A  Forecast. Anchor = a player entering season Y with value in Y-1 or Y-2 (2014-2025). Features known in the
   offseason before Y: pricing group, age on Sep 1 of Y, value Y-1 (v1), value Y-2 (v2), games fraction Y-1.
   Forecast of seasons Y..Y+4 (any team; 0 = no snaps: cut and unsigned, retired, injured) = mean of the K
   nearest comparable anchors in the same group (distance on age, v1, v2, games). Out of sample: comps come from
   other players and other anchor seasons. Checked against actual values by horizon and age, vs carrying v1 forward.
B  Signings. Unrestricted free-agent contracts (OTC contract_history 'UFA'), 2015-2023, APY >= 1% of cap, matched to
   an anchor. Per contract year: forecast value (any team), realized value with the signing team (0 once he's gone),
   realized value anywhere, APY. Early exit = no snaps for the team that season. Dead money proxy: guaranteed money
   not covered by APY x seasons with the team. Extensions and franchise tags are shown for reference.

Run: .venv/bin/python scripts/vet_value.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import roster_need as rn  # noqa: E402

RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
import veteran as vt  # noqa: E402
from veteran import H, K, LAST, Comps, anchors, season_table  # noqa: E402


def part_a(A, C, rng):
    # out-of-sample forecasts on a random 4000 anchors (comps exclude the same player and season)
    S = A.sample(4000, random_state=1)
    F = pd.DataFrame([C.forecast(r) for r in S.to_dict("records")], index=S.index)
    S = S.join(F)
    print(f"A. Forecast check: {len(S)} random anchors 2014-2025, comps from other players and seasons (K={K})")
    rows = []
    for k in range(1, H + 1):
        m = S[f"y{k}"].notna()
        a, f, naive = S.loc[m, f"y{k}"], S.loc[m, f"f{k}"], S.loc[m, "v1"]
        rows.append({"season": k, "n": int(m.sum()), "actual": a.mean() * 100, "forecast": f.mean() * 100,
                     "v1": naive.mean() * 100, "rmse_fc": np.sqrt(((a - f) ** 2).mean()) * 100,
                     "rmse_v1": np.sqrt(((a - naive) ** 2).mean()) * 100, "corr_fc": np.corrcoef(a, f)[0, 1]})
    print("  mean value, % of cap (actual vs forecast vs last season carried forward):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    S["ab"] = pd.cut(S.age, [0, 25, 28, 31, 34, 99], labels=["<25", "25-27", "28-30", "31-33", "34+"])
    t = S.groupby("ab", observed=True).apply(lambda d: pd.Series(
        {"n": len(d), **{f"act{k}/fc{k}": d[f"y{k}"].sum() / d.loc[d[f"y{k}"].notna(), f"f{k}"].sum()
                         for k in (1, 3, 5)}}), include_groups=False)
    print("  actual / forecast by age band (1.00 = calibrated):")
    print(t.round(2).to_string())
    t = S.groupby("grp").apply(lambda d: pd.Series(
        {"n": len(d), **{f"act{k}/fc{k}": d[f"y{k}"].sum() / d.loc[d[f"y{k}"].notna(), f"f{k}"].sum()
                         for k in (1, 3)}}), include_groups=False)
    print("  by group:")
    print(t.round(2).to_string())
    # aging: value kept relative to last season, starters (v1 >= 2% of cap)
    T = A[A.v1 >= 0.02]
    T = T.assign(ab=pd.cut(T.age, [0, 25, 28, 31, 34, 99], labels=["<25", "25-27", "28-30", "31-33", "34+"]))
    print("\n  Value kept vs last season, players worth >= 2% of cap last season (sum y_k / sum v1):")
    print(T.groupby("ab", observed=True).apply(lambda d: pd.Series(
        {"n": len(d), **{f"y{k}": d[f"y{k}"].sum() / d.loc[d[f"y{k}"].notna(), "v1"].sum() for k in range(1, H + 1)}}),
        include_groups=False).round(2).to_string())
    return S


def signings(kind="UFA"):
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    c = pd.read_parquet(RAW / "contracts.parquet")
    cap = (c.apy / c.apy_cap_pct).replace([np.inf], np.nan).groupby(c.year_signed).median()
    c = c[c.gsis_id.map(g2p).notna() & c.contract_history.notna()]
    c = c.assign(n=c.contract_history.map(len)).sort_values("n").drop_duplicates("gsis_id", keep="last")
    nick = dict(zip(*pd.read_parquet(RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Oilers": "TEN"})
    rows = []
    for gs, hist in zip(c.gsis_id, c.contract_history):
        for x in hist:
            if x.get("contract_type") != kind or x.get("year_signed") is None or x.get("apy") is None:
                continue
            y = int(x["year_signed"])
            if y not in cap.index:
                continue
            rows.append({"pfr_id": g2p[gs], "Y": y, "team": rn.team(nick.get(x["team"], x["team"])),
                         "yrs": int(x["yrs"] or 1), "apy": x["apy"] / cap[y], "gtd": (x.get("guarantees") or 0) / cap[y],
                         "status": x.get("status")})
    return pd.DataFrame(rows)


def part_b(A, C, kind="UFA", years=(2015, 2023), min_apy=0.01):
    played = rn._played_for()
    D = signings(kind)
    D = D[D.Y.between(*years) & (D.apy >= min_apy)].drop_duplicates(["pfr_id", "Y", "team"])
    D = D.merge(A, on=["pfr_id", "Y"], how="inner")
    F = pd.DataFrame([C.forecast(r) for r in D.to_dict("records")], index=D.index)
    D = D.join(F)
    V = season_table().set_index(["pfr_id", "season"]).value_ours.to_dict()
    for k in range(1, H + 1):
        y = D.Y + k - 1
        on = np.array([tm in played.get((p, yy), set()) for p, yy, tm in zip(D.pfr_id, y, D.team)])
        D[f"on{k}"] = np.where(y <= LAST, on, np.nan)
        D[f"w{k}"] = np.where(y <= LAST, np.where(on, D[f"y{k}"], 0.0), np.nan)  # value with the team
    # seasons with the team before the first gone season, within the contract
    D["kept"] = [next((k - 1 for k in range(1, min(r.yrs, H) + 1) if r[f"on{k}"] != 1), min(r.yrs, H))
                 for _, r in D.iterrows()]
    D["full"] = D.Y + np.minimum(D.yrs, H) - 1 <= LAST
    D["dead"] = np.where(D.kept < np.minimum(D.yrs, H), (D.gtd - D.apy * D.kept).clip(lower=0), 0.0)
    print(f"\nB. {kind} contracts {years[0]}-{years[1]}, APY >= {min_apy:.0%} of cap: {len(D)} matched "
          f"(median {D.yrs.median():.0f} yrs, APY {D.apy.median() * 100:.1f}% of cap)")
    rows = []
    for k in range(1, H + 1):
        m = (D.yrs >= k) & D[f"y{k}"].notna()
        d = D[m]
        rows.append({"contract yr": k, "n": len(d), "APY": d.apy.mean() * 100, "forecast": d[f"f{k}"].mean() * 100,
                     "realized anywhere": d[f"y{k}"].mean() * 100, "realized w/ team": d[f"w{k}"].mean() * 100,
                     "still on team": d[f"on{k}"].mean()})
    print("  per contract season, mean % of cap:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    Fu = D[D.full].copy()
    yrs = np.minimum(Fu.yrs, H)
    Fu["pay"] = Fu.apy * yrs
    Fu["fc"] = sum(np.where(yrs >= k, Fu[f"f{k}"], 0) for k in range(1, H + 1))
    Fu["real_w"] = sum(np.where(yrs >= k, Fu[f"w{k}"], 0) for k in range(1, H + 1))
    Fu["cost"] = Fu.apy * Fu.kept + Fu.dead  # paid while with the team + dead money proxy
    print(f"  whole contracts finished by {LAST} (n {len(Fu)}), totals % of cap: APY x yrs {Fu.pay.mean() * 100:.2f}, "
          f"forecast {Fu.fc.mean() * 100:.2f}, realized w/ team {Fu.real_w.mean() * 100:.2f}, "
          f"actual cost {Fu.cost.mean() * 100:.2f} (dead proxy {Fu.dead.mean() * 100:.2f})")
    Fu["s_fc"], Fu["s_real"] = Fu.fc - Fu.pay, Fu.real_w - Fu.cost
    print(f"  forecast surplus (fc - APY x yrs) vs realized (w/ team - cost): corr {np.corrcoef(Fu.s_fc, Fu.s_real)[0, 1]:.2f}")
    Fu["tert"] = pd.qcut(Fu.s_fc.rank(method="first"), 3, labels=["low", "mid", "high"])
    print("  by forecast surplus tercile (mean % of cap over the contract):")
    print(Fu.groupby("tert", observed=True)[["apy", "s_fc", "s_real", "fc", "real_w", "cost"]].mean().mul(100)
          .round(2).to_string())
    Fu["ab"] = pd.cut(Fu.age, [0, 27, 30, 99], labels=["<=27", "28-30", "31+"])
    Fu["qb"] = np.where(Fu.grp == "QB", "QB", "non-QB")
    print("  by age at signing / QB:")
    print(Fu.groupby(["qb", "ab"], observed=True).apply(lambda d: pd.Series({
        "n": len(d), "apy": d.apy.mean() * 100, "fc/pay": d.fc.sum() / d.pay.sum(),
        "real_w/cost": d.real_w.sum() / d.cost.sum(), "real_w/fc": d.real_w.sum() / d.fc.sum()}),
        include_groups=False).round(2).to_string())
    return D


def part_c(A, C, D, kind):
    """Fit the cut rule (CUT_THETA) to how long signees stay, then score each finished contract in points:
    predicted (comps' futures through the contract) vs realized (value with the team, APY while there, dead money
    proxy in the season he left)."""
    F = D[D.Y + np.minimum(D.yrs, H) - 1 <= LAST].copy()
    F["n"] = np.minimum(F.yrs, H).astype(int)
    paths = {i: C.paths(r, r["n"]) for i, r in zip(F.index, F.to_dict("records"))}
    on_real = {k: F.loc[F.n >= k, f"on{k}"].mean() for k in range(2, H + 1)}
    print(f"\nC. {kind}: finished contracts {len(F)}; share still on the team by contract season:",
          {k: round(v, 2) for k, v in on_real.items()})
    best = None
    for th in ([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] if kind == "UFA" else [vt.CUT_THETA]):
        on_m = {k: [] for k in range(2, H + 1)}
        for i, r in F.iterrows():
            V, S = paths[i]
            _, _, on = vt.team_seasons(V, r.n, r.apy, r.gtd, th)
            for k in range(2, r.n + 1):
                on_m[k].append(on[:, k - 1].mean())
        on_m = {k: np.mean(v) for k, v in on_m.items() if v}
        err = sum((on_m[k] - on_real[k]) ** 2 * (F.n >= k).sum() for k in on_m)
        print(f"  theta {th:4.2f}: modeled on-team {dict((k, round(v, 2)) for k, v in on_m.items())}  err {err:.2f}")
        if best is None or err < best[1]:
            best = (th, err)
    th = best[0]
    print(f"  chosen theta {th}")
    pred, real = [], []
    for i, r in F.iterrows():
        V, S = paths[i]
        val, cost, on = vt.team_seasons(V, r.n, r.apy, r.gtd, th)
        pred.append(vt.points(val, S, cost, on, r.grp == "QB").sum(axis=1).mean())
        w = np.array([[r[f"w{k}"] for k in range(1, r.n + 1)]])
        sh = np.array([[r[f"s{k}"] if r[f"on{k}"] == 1 else 0.0 for k in range(1, r.n + 1)]])
        o = np.array([[r[f"on{k}"] == 1 and k <= r.kept for k in range(1, r.n + 1)]])
        c = np.where(o, r.apy, 0.0)
        if r.kept < r.n:
            c[0, int(r.kept)] = r.dead
        real.append(vt.points(np.where(o, w, 0.0), sh, c, o, r.grp == "QB").sum())
    F["pred_pts"], F["real_pts"] = pred, real
    print(f"  points over the contract (strength-points x seasons): predicted {F.pred_pts.mean():+.3f}, "
          f"realized {F.real_pts.mean():+.3f}; corr {np.corrcoef(F.pred_pts, F.real_pts)[0, 1]:.2f}")
    rng = np.random.default_rng(0)
    bs = [F.real_pts.to_numpy()[k].mean() - F.pred_pts.to_numpy()[k].mean()
          for k in (rng.integers(0, len(F), len(F)) for _ in range(2000))]
    print(f"  realized - predicted {F.real_pts.mean() - F.pred_pts.mean():+.3f} (95% {np.percentile(bs, 2.5):+.3f} "
          f"to {np.percentile(bs, 97.5):+.3f})")
    F["tert"] = pd.qcut(F.pred_pts.rank(method="first"), 3, labels=["low", "mid", "high"])
    F["qb"] = np.where(F.grp == "QB", "QB", "non-QB")
    F["ab"] = pd.cut(F.age, [0, 27, 30, 99], labels=["<=27", "28-30", "31+"])
    F["size"] = pd.cut(F.apy, [0, 0.02, 0.05, 1], labels=["1-2%", "2-5%", "5%+"])
    for col in ["tert", "qb", "ab", "size"]:
        print(F.groupby(col, observed=True).agg(n=("apy", "size"), apy=("apy", lambda x: x.mean() * 100),
                                                 pred=("pred_pts", "mean"), real=("real_pts", "mean"))
              .round(3).to_string())
    return F, th


def part_e():
    """Example contracts, titles added over 5 seasons by situation (% of one Super Bowl)."""
    import pick_value as pv
    ex = [("EDGE, 27, star (6% last 2 yrs)", dict(grp="EDGE", age=27, v1=0.06, v2=0.06, g1=1.0), 4, 0.10, 0.20),
          ("EDGE, 27, star, cheaper", dict(grp="EDGE", age=27, v1=0.06, v2=0.06, g1=1.0), 4, 0.07, 0.14),
          ("WR, 30, good (4%)", dict(grp="WR", age=30, v1=0.04, v2=0.045, g1=1.0), 3, 0.06, 0.08),
          ("OL, 26, solid starter (3%)", dict(grp="OL", age=26, v1=0.03, v2=0.03, g1=1.0), 4, 0.04, 0.06),
          ("CB, 32, aging (2.5%)", dict(grp="DB", age=32, v1=0.025, v2=0.03, g1=0.9), 2, 0.03, 0.03),
          ("QB, 29, good starter (12%)", dict(grp="QB", age=29, v1=0.12, v2=0.12, g1=1.0), 4, 0.17, 0.35),
          ("QB, 35, aging (10%)", dict(grp="QB", age=35, v1=0.10, v2=0.12, g1=1.0), 2, 0.12, 0.15)]
    rows = []
    for name, pl_, yrs, apy, gtd in ex:
        r = {"player": name, "yrs": yrs, "APY %": apy * 100}
        for sname in pv.SITUATIONS:
            r[sname] = vt.evaluate(sname, pl_, yrs, apy, gtd)["titles_added"] * 100
        e = vt.evaluate("Middle of the pack", pl_, yrs, apy, gtd)
        r["value/yr %"] = e["value"].mean() * 100
        rows.append(r)
    print("\nE. Example signings: titles added over 5 seasons (% of one Super Bowl), by situation")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    if "--rebuild" in sys.argv or not (OUT / "vet_anchors.parquet").exists():
        anchors(season_table()).to_parquet(OUT / "vet_anchors.parquet")
    A = pd.read_parquet(OUT / "vet_anchors.parquet")
    C = Comps(A)
    if "--skip-a" not in sys.argv:
        part_a(A, C, np.random.default_rng(0))
    D = part_b(A, C, "UFA")
    F, th = part_c(A, C, D, "UFA")
    F.to_parquet(OUT / "vet_signings_ufa.parquet")
    for kind in ("Extension", "Franchise"):
        part_c(A, C, part_b(A, C, kind), kind)
    part_e()


if __name__ == "__main__":
    main()
