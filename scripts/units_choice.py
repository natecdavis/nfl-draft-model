"""Units fix: value seasons with cfdarko's within-position pricing (one overall conversion factor)
instead of one factor per position family, then compare choices against teams' picks, 2014-2022.

  family   today's model: per-family factors matching snap share x own later pay, which
           discounts rotational positions (RB, TE, EDGE, DL) and boosts OL
  single   one factor (1.10) for all positions: a role is priced at what veterans at that
           position earn for that snap share and grade

Both use the current need rule and fall shading applied once (scripts/need_choice.py tables:
data/derived/need_tables for family, data/derived/units_tables for single), so only the units
differ. Family-unit choices are read from data/derived/need_choice.parquet ("current" rule).

Yardsticks (each rec counted once per draft where it differs from the team's pick):
  titles, family units / titles, single units   realized value (review.RealizedV2) in each units;
                                                each favors its own model, so read them together
  AV (PFR weighted career approximate value), seasons started (PFR), second-contract pay
  (best later contract, % of cap; 0 if none) - outside both value measures; drafted players only

Run:  python scripts/units_choice.py [workers=6]     (report only: ... report)
"""
import os
import sys

os.environ["OUTCOME_UNITS"] = "single"
os.environ["NEED_TABLE_DIR"] = "units_tables"
os.environ["NEED_LEVELS"] = "current"
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from need_choice import TDIR, YEARS, NTable, _key, build_table  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from review import OFF_BOARD_RANK, board, draft_night  # noqa: E402
from review import team as _team  # noqa: E402

RULES = ["family", "single"]


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    from simulator import situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(YEARS)]
    sits = {_key(situation(r)): situation(r) for _, r in S.iterrows()}
    todo = [s for k, s in sits.items() if not (TDIR / f"{k}.npz").exists()]
    print(f"{len(sits)} situations, building {len(todo)} tables in {TDIR.name}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))

    rows = []
    for y in YEARS:
        picks, _ = draft_night(y)
        b = board(y, picks)
        st = {r.team: situation(r) for _, r in S[S.year == y].iterrows()}
        tabs, counts = {}, {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            if tm not in st:
                continue
            tab = tabs.setdefault(_key(st[tm]), NTable(st[tm]))
            c = counts.setdefault(tm, {})
            avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp != "SPEC")]
            v = [tab.current(g, c.get(g, 0), r, k) for g, r in zip(avail.grp, avail["rank"].clip(upper=OFF_BOARD_RANK))]
            rec = avail.iloc[int(np.argmax(v))]
            rows.append({"year": y, "pick": k, "single_player": rec["name"], "single_grp": rec.grp, "single_at": rec.drafted_at})
            if p.grp != "SPEC":
                c[p.grp] = c.get(p.grp, 0) + 1
        print(f"{y} choices done", flush=True)
    N = pd.read_parquet(OUT / "need_choice.parquet")
    D = N[["year", "pick", "round", "team", "player", "grp", "real_team", "current_player", "current_grp", "current_at"]] \
        .rename(columns={"current_player": "family_player", "current_grp": "family_grp", "current_at": "family_at"}) \
        .merge(pd.DataFrame(rows), on=["year", "pick"], how="left")
    D["team_at"] = D.pick.astype(float)
    D["team_player"], D["team_grp"] = D.player, D.grp

    from review import RealizedV2
    from trade import Context
    ctx = Context()
    sit = {(r.year, r.team): situation(r) for _, r in S.iterrows()}
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    pay = {(y, p): a for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}
    for units in RULES:
        R = RealizedV2(ctx, units=units)
        for who in ["team"] + RULES:
            D[f"t_{units}_{who}"] = [R.titles(sit[(y, tm)], g, y, at, k, nm) for y, tm, g, at, k, nm in
                                     zip(D.year, D.team, D[f"{who}_grp"], D[f"{who}_at"], D.pick, D[f"{who}_player"])]
        print(f"realized titles in {units} units done", flush=True)
    for who in ["team"] + RULES:
        at = D[f"{who}_at"]
        D[f"av_{who}"] = [av.get((y, int(a)), 0) if pd.notna(a) else np.nan for y, a in zip(D.year, at)]
        D[f"gs_{who}"] = [gs.get((y, int(a)), 0) if pd.notna(a) else np.nan for y, a in zip(D.year, at)]
        D[f"pay_{who}"] = [(pay.get((y, int(a))) if pd.notna(pay.get((y, int(a)))) else 0.0) * 100 if pd.notna(a) else np.nan
                           for y, a in zip(D.year, at)]
    D.to_parquet(OUT / "units_choice.parquet")
    report(D)


def report(D):
    rng = np.random.default_rng(0)

    def boot(x, cl):
        x, cl = np.asarray(x, float), np.asarray(cl)
        ok = ~np.isnan(x); x, cl = x[ok], cl[ok]
        ids = np.unique(cl); groups = [x[cl == i] for i in ids]
        means = [np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(means, 2.5):+.2f} to {np.percentile(means, 97.5):+.2f}) n={len(x)}"

    D = D.copy()
    D["rnd"] = pd.cut(D["round"], [0, 1, 4, 7], labels=["Round 1", "Rounds 2-4", "Rounds 5-7"])
    D["cl"] = D.year.astype(str) + D.team
    print("\nPosition mix of recommendations (% of picks):")
    for who in RULES + ["team"]:
        t = (pd.crosstab(D.rnd, D[f"{who}_grp"], normalize="index") * 100).round(0)
        print(f"-- {who}\n{t.to_string()}")
    ys = [("t_family", "titles, family units"), ("t_single", "titles, single units"), ("av", "AV (drafted recs)"),
          ("gs", "seasons started (drafted recs)"), ("pay", "2nd-contract pay %cap (drafted recs)")]
    print("\nModel's player minus team's player where they disagree, each rec once per draft:")
    for who in RULES:
        dis = D[D[f"{who}_player"] != D.player].drop_duplicates(["year", f"{who}_player"])
        print(f"== {who} units model")
        for col, lab in ys:
            a = f"{col}_{who}" if not col.startswith("t_") else f"{col}_{who}"
            tcol = f"{col}_team"
            parts = [f"{rn}: {boot(g[a] - g[tcol], g.cl)}" for rn, g in [("All", dis)] + list(dis.groupby("rnd", observed=True))]
            print(f"  {lab:38s} " + " | ".join(parts))
    print("\nSingle minus family model, paired at every pick (repeat recs included):")
    for col, lab in ys:
        parts = [f"{rn}: {boot(g[f'{col}_single'] - g[f'{col}_family'], g.cl)}"
                 for rn, g in [("All", D)] + list(D.groupby("rnd", observed=True))]
        print(f"  {lab:38s} " + " | ".join(parts))
    print(f"\nSame choice, single vs family: {(D.single_player == D.family_player).mean():.0%}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "units_choice.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
