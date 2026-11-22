"""Do free agents who re-sign with their own team beat their forecast more than those who switch teams?

The old team has watched him for years and knows his medicals; letting him walk (or him leaving) may be a signal.
UFA signings 2015-24 (OTC contract history, APY >= 0.5% of cap). Last team = his team on the final weekly roster of
season S-1 (matched by gsis id), else the team of his last snap in S-1 (known before free agency); signing team = OTC contract team. Re-signed = same team.
Forecast = veteran.Comps on past-only inputs (scripts/forecast_benchmark.py make_comps), skip 0: features through
S-1, outcomes seasons S..S+2 at any team (0 if out of the league). Price residual r = log(actual APY / contract
model's out-of-sample prediction) (data/derived/ufa_price_residuals.parquet, past-only).
A  actual / forecast by class, seasons 1-3; gap (actual - forecast, % of cap x100) re-signed minus switched,
   raw and controlling for r (teams play who they pay: scripts/forecast_benchmark.py part C), player bootstrap.
B  what differs: first-season quality q, snap share, games fraction on the re-sign dummy, controlling for last
   season's q / share / games, age, group, predicted log price and r.
C  splits: big deals (>= 5% of cap) vs smaller, age < 29 vs 29+, halves 2015-19 / 2020-24.
Pre-set rule: re-signing carries information if A's pooled gap controlling for r has a 95% interval excluding 0
with the same sign in both halves; B then says whether it is talent (q) or usage (share / games).
Run: .venv/bin/python scripts/resign_vs_switch.py
"""
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1] + ["--inputs", "past"]
sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import forecast_benchmark as fb  # noqa: E402
import roster_need as rn  # noqa: E402
import veteran as vt  # noqa: E402

NB = 1000


def classify(R):
    nick = dict(zip(*pd.read_parquet(vt.RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS"})
    H = fm.history()
    sign_team = {(p, y): rn.team(nick.get(t, t)) if t else None for p, h in H.items() for (y, k, a, yr, g, t) in h
                 if k == "UFA"}
    # last team: final weekly roster (by gsis id - older rosters often lack pfr ids), else the team of his last snap
    pl = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    p2g = dict(zip(pl.pfr_id, pl.gsis_id))
    ro = pd.read_parquet(vt.RAW / "rosters_weekly.parquet", columns=["season", "week", "team", "gsis_id"])
    lr = ro.dropna(subset=["gsis_id"]).sort_values("week").groupby(["gsis_id", "season"]).team.last().map(rn.team).to_dict()
    sc = pd.read_parquet(vt.RAW / "snap_counts.parquet", columns=["season", "week", "team", "pfr_player_id"])
    ls = sc.sort_values("week").groupby(["pfr_player_id", "season"]).team.last().map(rn.team).to_dict()
    last = {}
    for p, y in zip(R.pfr_id, R.Y):
        t = lr.get((p2g.get(p), y - 1)) or ls.get((p, y - 1))
        if t:
            last[(p, y - 1)] = t
    R = R.assign(new_team=[sign_team.get((p, y)) for p, y in zip(R.pfr_id, R.Y)],
                 old_team=[last.get((p, y - 1)) for p, y in zip(R.pfr_id, R.Y)])
    R["cls"] = np.where(R.new_team.isna() | R.old_team.isna(), "unknown",
                        np.where(R.new_team == R.old_team, "re-signed", "switched"))
    return R


def forecasts(C, R):
    A = C.A.set_index(["pfr_id", "Y"])
    rows = []
    for x in R.itertuples():
        if (x.pfr_id, x.Y) not in A.index:
            continue
        a = A.loc[(x.pfr_id, x.Y)]
        a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
        row = {**a.to_dict(), "pfr_id": x.pfr_id, "Y": x.Y, "p_share": np.nan, "p_gp": np.nan}
        for k in (1, 2, 3):
            if pd.notna(row[f"y{k}"]):
                rows.append({"pfr_id": x.pfr_id, "S": x.Y, "k": k, "cls": x.cls, "r": x.r, "apy": x.apy, "age": x.age,
                             "grp": x.grp, "f": C.outcomes(row, k)[0].mean(), "y": row[f"y{k}"]})
    B = pd.DataFrame(rows)
    B["e"] = B.y - B.f
    return B


def gap(d, control):
    """Re-signed minus switched gap in (actual - forecast), x100, optionally controlling for r; player bootstrap."""
    d = d.reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), (d.cls == "re-signed").astype(float)] + ([d.r.to_numpy()] if control else []))
    y = d.e.to_numpy() * 100
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    by = d.groupby("pfr_id").indices
    ids = list(by)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(NB):
        i = np.concatenate([by[p] for p in rng.choice(ids, len(ids))])
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    return b, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def part_b(R):
    R = R[R.cls != "unknown"]
    v = vt.season_table().set_index(["pfr_id", "season"])
    rows = []
    for x in R.itertuples():
        g = lambda s, c: v[c].get((x.pfr_id, s), np.nan)  # noqa: E731
        rows.append(dict(p=x.pfr_id, rs=float(x.cls == "re-signed"), r=x.r, grp=x.grp, age=x.age, lp=np.log(x.pred),
                         q0=g(x.Y - 1, "q_f"), sh0=g(x.Y - 1, "share"), gf0=g(x.Y - 1, "gfrac"),
                         q=g(x.Y, "q"), sh=g(x.Y, "share"), gf=g(x.Y, "gfrac")))
    D = pd.DataFrame(rows)
    X = ["rs", "r", "q0", "sh0", "gf0", "age", "lp"]
    print("\nB. First season, re-signed minus switched (OLS + group dummies, player bootstrap)")
    for y, lab in [("q", "quality"), ("sh", "snap share"), ("gf", "games fraction")]:
        d = D.dropna(subset=[y] + X).reset_index(drop=True)
        M = np.column_stack([np.ones(len(d)), d[X].to_numpy(), pd.get_dummies(d.grp, drop_first=True).astype(float).to_numpy()])
        b = np.linalg.lstsq(M, d[y], rcond=None)[0][1]
        by = d.groupby("p").indices
        ids = list(by)
        rng = np.random.default_rng(0)
        bs = [np.linalg.lstsq(M[i], d[y].to_numpy()[i], rcond=None)[0][1]
              for i in (np.concatenate([by[p] for p in rng.choice(ids, len(ids))]) for _ in range(300))]
        print(f"  {lab}: n {len(d)}, {b:+.3f} ({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}) = "
              f"{b / d[y].std():+.2f} sd of the outcome")


def main():
    pd.set_option("display.width", 200)
    R = pd.read_parquet(vt.OUT / "ufa_price_residuals.parquet")
    R = classify(R[R.Y <= 2024])
    print("classes:", R.cls.value_counts().to_dict())
    print("price residual r by class:", R.groupby("cls").r.mean().round(3).to_dict(),
          "| predicted APY % cap:", (R.groupby("cls").pred.mean() * 100).round(2).to_dict())
    B = forecasts(fb.make_comps(), R)
    B = B[B.cls != "unknown"]
    t = B.groupby(["cls", "k"]).agg(n=("y", "size"), actual=("y", "sum"), fc=("f", "sum"))
    t["actual/forecast"] = t.actual / t.fc
    print("\nA. actual / forecast by class and season")
    print(t[["n", "actual/forecast"]].unstack("cls").round(3).to_string())
    rows = []
    for lab, d in [("all seasons 1-3", B), ("season 1", B[B.k == 1]), ("2015-19", B[B.S <= 2019]),
                   ("2020-24", B[B.S >= 2020]), ("big deals >= 5% cap", B[B.apy >= 0.05]),
                   ("smaller deals", B[B.apy < 0.05]), ("age < 29", B[B.age < 29]), ("age 29+", B[B.age >= 29])]:
        g0, g1 = gap(d, False), gap(d, True)
        rows.append({"slice": lab, "n": len(d), "re-signed share": f"{(d.cls == 're-signed').mean():.2f}",
                     "gap raw": f"{g0[0]:+.3f} ({g0[1]:+.3f} to {g0[2]:+.3f})",
                     "gap | price residual": f"{g1[0]:+.3f} ({g1[1]:+.3f} to {g1[2]:+.3f})"})
    print("\n   re-signed minus switched, (actual - forecast) % of cap x100 per season; mean forecast "
          f"{B.f.mean() * 100:.2f}")
    print(pd.DataFrame(rows).to_string(index=False))
    part_b(R)


if __name__ == "__main__":
    main()
