"""Comparison-career correction (outcomes_v2.COMP_CORRECT = "ll") vs today's engine ("off").

Background (scripts/misses.py): the engine samples careers of the nearest drafted players by expected value;
at the top of each family all neighbours are below the player, so consensus top-10 non-QBs got careers worth
~0.71x what such players actually did. "ll" rescales the sample to a local-linear fit of comparison value on
expected value at the player's own expectation.

  engine   leave-one-draft-out check, 2014-2022: for each class, an engine fit without it; its drafted board
           players' actual 4-season value vs the mean of the comparison careers, off and ll
  review   leave-one-draft-out past-drafts review, off (holdout/review_picks_lodo_cost0.1.parquet = production)
           vs ll (holdout/review_picks_lodo_ccll.parquet, from COMP_CORRECT=ll scripts/holdout_review.py):
           scorecard on common yardsticks, calibration by consensus band, recommendation mix
  draft    whole-draft replay (simulator tables, both built leave-one-draft-out): the model drafts each team's
           whole draft at its real picks, greedy, its own choices setting needs; ll minus off per team-draft
           (needs COMP_CORRECT=ll python src/simulator.py build first)
  values   pick values (titles added over 5 seasons, % of one) by situation, position and pick, off vs ll

Run:  python scripts/comp_correct.py engine|review|draft|values
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT, RAW  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
YEARS = range(2014, 2023)
HD = OUT / "holdout"
FAMS = ["QB", "WR", "EDGE", "DL", "LB", "OL", "DB", "RB", "TE"]
rng = np.random.default_rng(0)


def boot(x, by=None):
    """Mean and 95% range; resampling draft years if `by` (array of years) is given."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return "n/a"
    if by is None:
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    else:
        by = np.asarray(by)
        yrs = np.unique(by)
        parts = {y: x[by == y] for y in yrs}
        b = [np.concatenate([parts[y] for y in rng.choice(yrs, len(yrs))]).mean() for _ in range(2000)]
    return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


def engine():
    import outcomes_v2 as ov
    import prospect_model as pm
    full = pm.load()
    rows = []
    for y in YEARS:
        ov.HOLDOUT, ov._MODEL = ("lodo", y), None
        M = ov.model()
        d = full[full.drafted & (full.cls == y) & full.grp.isin(FAMS)].copy()
        d["r"] = d.hasan_rank.fillna(pm.OFF_BOARD).clip(1, pm.OFF_BOARD).astype(int)
        d["actual"] = d[[f"y{t}" for t in pm.SEASONS]].sum(axis=1) / ov.unit_factor("")
        c = [(lambda pw: float((pw[0].sum(1) * pw[1]).sum()))(M._near(g, r)) for g, r in zip(d.grp, d.r)]
        d["off"] = c
        d["ll"] = [v * M._scale(g, r) for v, g, r in zip(c, d.grp, d.r)]
        rows.append(d[["cls", "grp", "hasan_rank", "actual", "off", "ll"]])
        print(f"{y} done", flush=True)
    ov.HOLDOUT, ov._MODEL = None, None
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(HD / "comp_correct_engine.parquet")
    d["band"] = pd.cut(d.hasan_rank.fillna(999), [0, 10, 32, 64, 120, 200, 300, 1e9],
                       labels=["1-10", "11-32", "33-64", "65-120", "121-200", "201-300", "off board"])
    d["kind"] = np.where(d.grp == "QB", "QB", "non-QB")
    print("\n== Leave-one-draft-out: drafted players 2014-2022, 4-season value (cap share) ==")
    for keys, lab in [(["kind", "band"], ""), (None, "top 10 by family")]:
        g = d if keys else d[d.hasan_rank <= 10]
        g = g.groupby(keys or "grp", observed=True).agg(n=("actual", "size"), actual=("actual", "mean"),
                                                         off=("off", "mean"), ll=("ll", "mean"))
        g["off/actual"], g["ll/actual"] = g.off / g.actual, g.ll / g.actual
        print(lab)
        print(g.round(3).to_string())
    # errors per player (board players): mean absolute and squared error, off vs ll
    b = d[d.hasan_rank.notna()]
    for k in ["off", "ll"]:
        print(f"board players, {k}: mean error {np.mean(b[k] - b.actual):+.4f}, mean abs error "
              f"{np.mean(np.abs(b[k] - b.actual)):.4f}, rmse {np.sqrt(np.mean((b[k] - b.actual) ** 2)):.4f}")


def review():
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    look = {"gs": {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)},
            "av": {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)},
            "pay": {(y, p): a * 100 for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}}
    get = lambda d, y, a: 0.0 if pd.isna(a) else float(np.nan_to_num(d.get((y, int(a)), 0.0)))
    V = {"off": pd.read_parquet(HD / "review_picks_lodo_cost0.1.parquet"),
         "ll": pd.read_parquet(HD / "review_picks_lodo_ccll.parquet")}
    V = {k: d[d.year <= 2022].reset_index(drop=True) for k, d in V.items()}
    out = []
    for v, d in V.items():
        d = d[d.real_team.notna()]
        o = d[d.player != d.rec_player].drop_duplicates(["year", "rec_player"]).copy()
        for m, dd in look.items():
            o[m] = [get(dd, y, a) - get(dd, y, k) for y, a, k in zip(o.year, o.rec_drafted_at, o.pick)]
        o["t"] = o.real_model - o.real_team
        print(f"\n=== {v}: model minus team where they disagree, each rec once (titles realized; same yardstick) ===")
        for g, f in [("R1", o["round"] == 1), ("R2-4", o["round"].between(2, 4)), ("R5-7", o["round"] >= 5)]:
            x = o[f]
            print(f"  {g:5s} n={len(x):3d} | titles {boot(x.t)} | starts {boot(x.gs)} | AV {x.av.mean():+.1f} | "
                  f"2nd-contract pay {boot(x.pay)}")
        r1 = d[d["round"] == 1]
        print(f"  round-1 recs: QB {(r1.rec_grp == 'QB').mean():.0%}, position mix round 1: "
              f"{r1.rec_grp.value_counts(normalize=True).head(6).round(2).to_dict()}")
        r24 = d[d["round"].between(2, 4)]
        print(f"  rounds 2-4 mix: {r24.rec_grp.value_counts(normalize=True).head(6).round(2).to_dict()}")
        # calibration: recs each once, and team picks, by consensus band
        recs = d.drop_duplicates(["year", "rec_player"]).copy()
        recs["band"] = pd.cut(recs.rec_rank, [0, 10, 32, 64, 120, 1e9], labels=["1-10", "11-32", "33-64", "65-120", "121+"])
        recs["kind"] = np.where(recs.rec_grp == "QB", "QB", "non-QB")
        g = recs.groupby(["kind", "band"], observed=True).agg(n=("value_rec", "size"), pred=("value_rec", "sum"),
                                                             real=("real_model", "sum"))
        g["delivered/pred"] = g.real / g.pred
        print("  recommendations, delivered / predicted:")
        print(g.round(2).to_string())
        tp = d[(d.grp != "QB") & d.board_rank.notna()].copy()
        tp["band"] = pd.cut(tp.board_rank, [0, 5, 10, 20, 32, 64, 120, 1e9], labels=["1-5", "6-10", "11-20", "21-32", "33-64", "65-120", "121+"])
        tp["miss"] = tp.real_team - tp.value_actual
        print("  team non-QB picks, delivered minus predicted (95% over draft years):")
        for b, x in tp.groupby("band", observed=True):
            print(f"    {b:>7}: n {len(x):4d}  predicted {x.value_actual.mean():.2f}  delivered {x.real_team.mean():.2f}  "
                  f"miss {boot(x.miss, x.year)}")
        out.append(o.assign(version=v))
    a, b = (V[k].set_index(["year", "pick"]).rec_player for k in ("off", "ll"))
    same = a == b.reindex(a.index)
    rd = V["off"].set_index(["year", "pick"])["round"]
    print(f"\nsame recommendation: {same.mean():.0%} (R1 {same[rd == 1].mean():.0%}, R2-4 {same[rd.between(2, 4)].mean():.0%}, "
          f"R5-7 {same[rd >= 5].mean():.0%})")
    m = V["off"].merge(V["ll"], on=["year", "pick"], suffixes=("_off", "_ll"))
    ch = m[m.rec_player_off != m.rec_player_ll]
    print("changed round-1 recommendations:")
    print(ch[ch.round_off == 1][["year", "pick", "team_off", "rec_player_off", "rec_grp_off", "real_model_off",
                                 "rec_player_ll", "rec_grp_ll", "real_model_ll"]].round(2).to_string(index=False))
    pd.concat(out).to_parquet(HD / "comp_correct_review.parquet")


def draft():
    import outcomes_v2 as ov
    import simulator as sm
    from export_simulator_artifact import realized_arrays
    from positions import weights
    from review import p_need  # noqa: F401 (p_need column already in states)
    S = pd.read_parquet(OUT / "review_states.parquet")
    R = pd.read_parquet(OUT / "review_picks.parquet")
    real = realized_arrays(S, R)
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    pay = {(y, p): a * 100 for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}
    get = lambda d, y, a: 0.0 if pd.isna(a) else float(np.nan_to_num(d.get((y, int(a)), 0.0)))
    rows = []
    for y in YEARS:
        P, W = sm.pool(y), weights(y)
        for _, s in S[S.year == y].iterrows():
            ks = sorted(int(k) for k in R[(R.year == y) & (R.team == s.team)].pick)
            if not ks or f"{y}_{s.team}" not in real:
                continue
            arr = real[f"{y}_{s.team}"]
            res = {}
            for v in ("off", "ll"):
                ov.COMP_CORRECT = v
                table = sm.get_table(sm.situation(s), year=y)
                taken, counts, picks = set(), {}, []
                for j, k in enumerate(ks):
                    o = sm.options(P, table, k, taken, counts, float(s.p_need), W)
                    t = o.iloc[0]
                    taken.add(int(t.pid))
                    counts[t.grp] = counts.get(t.grp, 0) + 1
                    picks.append((j, k, int(t.pid), t["name"], t.grp, float(t.titles), t.drafted_at))
                res[v] = picks
            for v, picks in res.items():
                for j, k, pid, nm, g, ti, at in picks:
                    rows.append({"year": y, "team": s.team, "version": v, "pick": k, "name": nm, "grp": g, "pred": ti,
                                 "real": float(arr[j, pid]), "gs": get(gs, y, at), "av": get(av, y, at), "pay": get(pay, y, at)})
        print(f"{y} done", flush=True)
    ov.COMP_CORRECT = "off"
    D = pd.DataFrame(rows)
    D.to_parquet(HD / "comp_correct_draft.parquet")
    D["rnd"] = np.select([D.pick <= 32, D.pick <= 128], ["R1", "R2-4"], "R5-7")  # approx. round bands
    tot = D.groupby(["year", "team", "version"])[["real", "gs", "av", "pay"]].sum().unstack("version")
    print("\n== Whole-draft replay, ll minus off, per team-draft (95% over team-drafts) ==")
    for m in ["real", "gs", "av", "pay"]:
        print(f"  {m:5s}: {boot(tot[(m, 'll')] - tot[(m, 'off')])}   (off total mean {tot[(m, 'off')].mean():.2f})")
    a = D[D.version == "off"].reset_index(drop=True)
    b = D[D.version == "ll"].reset_index(drop=True)
    print(f"  same player at the same pick: {(a.name == b.name).mean():.0%}; by band: "
          f"{a.assign(s=a.name == b.name).groupby('rnd').s.mean().round(2).to_dict()}")
    for band in ["R1", "R2-4", "R5-7"]:
        f = a.rnd == band
        print(f"  {band}: titles ll-off per pick {boot(b.real[f] - a.real[f])}; mix off "
              f"{a[f].grp.value_counts(normalize=True).head(5).round(2).to_dict()} | ll "
              f"{b[f].grp.value_counts(normalize=True).head(5).round(2).to_dict()}")


def values():
    import json
    import polars as pl
    import outcomes_v2 as ov
    import pick_value as pv
    from outlook import star_thresholds
    from value import rookie_cost_curve
    sched = pv.load_schedules()
    teams_df = pd.read_parquet(RAW / "teams.parquet")
    curve = pv.strength_curve(sched, teams_df, pv.league_params(sched))
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)
    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    cost, bar = rookie_cost_curve(c), star_thresholds(c).xs(2026, level="season").to_dict()
    rates = json.load(open(OUT / "bridge.json"))["preseason"]
    picks = [1, 3, 5, 10, 16, 20, 32, 48, 64, 96, 160]
    rows = []
    for v in ("off", "ll"):
        ov.COMP_CORRECT = v
        for sname, sit in pv.SITUATIONS.items():
            base = pv.baseline_paths(sit, np.random.default_rng(11))
            bt = sb(base)
            for grp in ["QB", "WR", "ED", "IDL", "LB", "T", "CB", "S", "RB", "TE"]:
                for k in picks:
                    sh, pr = pv.sample_outcomes(None, None, grp, k, np.random.default_rng(k))
                    pts = pv.pick_effect(sh, pr, k, grp, sit, cost, bar, rates)
                    if pts is not None:
                        rows.append({"v": v, "situation": sname, "grp": grp, "pick": k,
                                     "titles": (sb(base + pts) - bt).sum(axis=1).mean() * 100})
    ov.COMP_CORRECT = "off"
    d = pd.DataFrame(rows)
    d.to_parquet(HD / "comp_correct_values.parquet")
    for sname in ["Middle of the pack", "Rebuilding"]:
        t = d[d.situation == sname].pivot_table(index=["grp", "v"], columns="pick", values="titles")
        print(f"\n{sname}: titles added over 5 seasons (% of one), off vs ll")
        print(t.round(2).to_string())
    nq = d[d.grp != "QB"].groupby(["v", "situation", "pick"]).titles.mean().unstack("pick")
    print("\nAverage non-QB position (unweighted), by situation:")
    print(nq.round(2).to_string())


if __name__ == "__main__":
    {"engine": engine, "review": review, "draft": draft, "values": values}[sys.argv[1]]()
