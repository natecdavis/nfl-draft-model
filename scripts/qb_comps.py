"""QB comparison careers by consensus rank (outcomes_v2.QB_COMPS = "rank") vs today's engine ("ev").

Background (README 'QB valuation'): the model's QB choices in rounds 2-3 delivered 0.56x / 0.28x the titles
predicted (round 1 1.20x). The engine samples careers of the drafted QBs nearest in expected value, but the
regression barely separates QB ranks, so QBs ranked 33-100 draw careers from QBs ranked 11-32: too many
starter-level outcomes (fat tail), which title odds reward more than linearly. "rank" picks the nearest
drafted QBs by consensus rank instead (same weights, local-linear correction and monotone cap).

  engine   leave-one-draft-out, 2014-2022: each class's board QBs valued by an engine fit without it;
           mean 4-season value and P(4-season value > 0.30 of cap, a starter-level hit) of the sampled
           careers vs what the players did, ev and rank
  draft    whole-draft replay (simulator tables, both leave-one-draft-out): the model drafts each team's whole
           draft at its real picks, greedy; rank minus ev per team-draft, on titles, AV, seasons started,
           2nd-contract pay (build first:  QB_COMPS=rank python -c "import sys; sys.path.insert(0,'src');
           import simulator; simulator.build_all(6)")
  values   QB pick values (titles, % of one) by situation and pick, ev vs rank

Run:  python scripts/qb_comps.py engine|draft|values
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
YEARS = range(2014, 2023)
HD = OUT / "holdout"
HIT = 0.30
VERSIONS = ("ev", "rank")


def engine():
    import outcomes_v2 as ov
    import prospect_model as pm
    full = pm.load()
    rows = []
    for y in YEARS:
        d = full[(full.cls == y) & (full.grp == "QB") & full.hasan_rank.notna()].copy()
        d["r"] = d.hasan_rank.clip(1, pm.OFF_BOARD).astype(int)
        d["actual"] = d[[f"y{t}" for t in pm.SEASONS]].sum(axis=1) / ov.unit_factor("QB")
        for v in VERSIONS:
            ov.QB_COMPS, ov.HOLDOUT, ov._MODEL = v, ("lodo", y), None
            M = ov.model()
            mean, hit = [], []
            for r in d.r:
                paths, w = M._near("QB", int(r))
                val = paths.sum(1) * M.scale("QB", int(r))
                mean.append(float((val * w).sum())); hit.append(float(w[val > HIT].sum()))
            d[v], d[f"{v}_hit"] = mean, hit
        rows.append(d[["cls", "player_name", "hasan_rank", "drafted", "actual"] + [c for v in VERSIONS for c in (v, f"{v}_hit")]])
        print(f"{y} done", flush=True)
    ov.QB_COMPS, ov.HOLDOUT, ov._MODEL = "ev", None, None
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(HD / "qb_comps_engine.parquet")
    report_engine(d)


def report_engine(d):
    d["band"] = pd.cut(d.hasan_rank, [0, 10, 20, 32, 50, 75, 100, 150, 330])
    d["hit"] = (d.actual > HIT).astype(float)
    g = d.groupby("band", observed=True).agg(n=("actual", "size"), actual=("actual", "mean"), ev=("ev", "mean"),
                                             rank=("rank", "mean"), hit=("hit", "mean"), ev_hit=("ev_hit", "mean"),
                                             rank_hit=("rank_hit", "mean"))
    g["ev/actual"], g["rank/actual"] = g.ev / g.actual, g["rank"] / g.actual
    print(f"\n== Leave-one-draft-out: board QBs 2014-2022, 4-season value (cap share); hit = value > {HIT} ==")
    print(g.round(3).to_string())
    for v in VERSIONS:
        e = d[v] - d.actual
        # Brier score of the hit probability
        print(f"  {v:4s}: mean error {e.mean():+.4f}, mean abs error {e.abs().mean():.4f}, rmse {np.sqrt((e ** 2).mean()):.4f}, "
              f"hit Brier {((d[f'{v}_hit'] - d.hit) ** 2).mean():.4f}, ranks 33-100 mean error "
              f"{e[d.hasan_rank.between(33, 100)].mean():+.4f}")


def draft():
    import outcomes_v2 as ov
    import simulator as sm
    from comp_correct import boot
    from export_simulator_artifact import realized_arrays
    from positions import weights
    from review import Careers
    S = pd.read_parquet(OUT / "review_states.parquet")
    R = pd.read_parquet(OUT / "review_picks.parquet")
    real = realized_arrays(S, R)
    car = Careers()
    rows = []
    for y in YEARS:
        P, W = sm.pool(y), weights(y)
        for _, s in S[S.year == y].iterrows():
            ks = sorted(int(k) for k in R[(R.year == y) & (R.team == s.team)].pick)
            if not ks or f"{y}_{s.team}" not in real:
                continue
            arr = real[f"{y}_{s.team}"]
            for v in VERSIONS:
                ov.QB_COMPS = v
                table = sm.get_table(sm.situation(s), year=y)
                taken, counts = set(), {}
                for j, k in enumerate(ks):
                    o = sm.options(P, table, k, taken, counts, float(s.p_need), W)
                    t = o.iloc[0]
                    taken.add(int(t.pid))
                    counts[t.grp] = counts.get(t.grp, 0) + 1
                    at = t.drafted_at
                    rows.append({"year": y, "team": s.team, "version": v, "pick": k, "name": t["name"], "grp": t.grp,
                                 "rank": float(t["rank"]) if "rank" in t else np.nan, "pred": float(t.titles),
                                 "real": float(arr[j, int(t.pid)]), "gs": car.get("gs", y, at, t["name"]),
                                 "av": car.get("av", y, at, t["name"]), "pay": car.get("pay", y, at, t["name"])})
        print(f"{y} done", flush=True)
    ov.QB_COMPS = "ev"
    D = pd.DataFrame(rows)
    D.to_parquet(HD / "qb_comps_draft.parquet")
    report_draft(D, boot)


def report_draft(D, boot=None):
    if boot is None:
        from comp_correct import boot
    D["rnd"] = np.select([D.pick <= 32, D.pick <= 64, D.pick <= 128], ["R1", "R2", "R3-4"], "R5-7")
    tot = D.groupby(["year", "team", "version"])[["real", "gs", "av", "pay"]].sum().unstack("version")
    print("\n== Whole-draft replay, rank minus ev, per team-draft (95% over team-drafts) ==")
    for m, lab in [("real", "titles"), ("gs", "seasons started"), ("av", "AV"), ("pay", "2nd-contract pay")]:
        print(f"  {lab:17s}: {boot(tot[(m, 'rank')] - tot[(m, 'ev')])}   (ev mean {tot[(m, 'ev')].mean():.2f})")
    a = D[D.version == "ev"].reset_index(drop=True)
    b = D[D.version == "rank"].reset_index(drop=True)
    same = a.name == b.name
    print(f"  same player at the same pick: {same.mean():.0%}")
    for band in ["R1", "R2", "R3-4", "R5-7"]:
        f = a.rnd == band
        print(f"  {band:4s}: titles rank-ev per pick {boot(b.real[f] - a.real[f])}; QB share ev {(a.grp[f] == 'QB').mean():.2f} "
              f"rank {(b.grp[f] == 'QB').mean():.2f}; changed {(~same[f]).sum()}")
    for v, x in [("ev", a), ("rank", b)]:
        q = x[x.grp == "QB"]
        print(f"  {v:4s} QB choices: n {len(q)}, predicted {q.pred.sum():.1f}, delivered {q.real.sum():.1f} "
              f"(x{q.real.sum() / max(q.pred.sum(), 1e-9):.2f}); by round "
              + ", ".join(f"{r} {g.real.sum() / max(g.pred.sum(), 1e-9):.2f} (n {len(g)})" for r, g in q.groupby("rnd")))
    c = pd.DataFrame({"year": a.year, "team": a.team, "pick": a.pick, "ev": a.name, "ev_grp": a.grp, "ev_real": a.real,
                      "rank": b.name, "rank_grp": b.grp, "rank_real": b.real})[~same]
    c = c[(c.ev_grp == "QB") | (c.rank_grp == "QB")]
    print(f"\nChanged choices involving a QB ({len(c)}), round 1-4:")
    print(c[c.pick <= 128].round(2).to_string(index=False))


def values():
    import outcomes_v2 as ov
    from pick_value import SITUATIONS
    from trade import Context, Pick, evaluate
    ctx = Context()
    rows = []
    for v in VERSIONS:
        ov.QB_COMPS, ov._MODEL = v, None
        for name, sit in SITUATIONS.items():
            if sit["qb_from"] is None:
                continue
            for k in (1, 5, 10, 16, 24, 32, 48, 64, 80, 100, 128):
                rows.append({"version": v, "situation": name, "pick": k,
                             "titles": evaluate(ctx, sit, [], [Pick(k, pos="QB")], n=8000, seed=k)["trade"] * 100})
    ov.QB_COMPS, ov._MODEL = "ev", None
    T = pd.DataFrame(rows).pivot_table(index=["situation", "pick"], columns="version", values="titles")
    print("\nQB at pick k (consensus rank k), titles added over 5 seasons, % of one:")
    print(T.round(2).to_string())


if __name__ == "__main__":
    {"engine": engine, "draft": draft, "values": values,
     "report": lambda: (report_engine(pd.read_parquet(HD / "qb_comps_engine.parquet")),
                        report_draft(pd.read_parquet(HD / "qb_comps_draft.parquet")))}[sys.argv[1]]()
