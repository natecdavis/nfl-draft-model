"""Whole-draft choice test for the rookie-QB timing fix (README 'Rookie QBs: sitting behind an incumbent').

Fix: draft-day QB statuses "rookie deal (later round)" and "..., below-average play" get qb_from = 1 (the need
starts now) instead of 2; p_need (the chance the need is real) is unchanged. Everything else is today's model.

Both versions draft each team's whole draft greedily at its real picks, 2014-2022 (other teams as they really
drafted), with value tables rebuilt on the current engine in data/derived/qbsit_tables (single units, current
need levels; the older units_tables predate the narrow/darko labels).

Scoring per team-draft, model minus the team's real draft:
  titles, model rule   review.RealizedV2 with the base statuses: seasons before qb_from count 0 (today's yardstick)
  titles, as played    same, but every flagged QB need starts now: a rookie's real seasons count
  AV, seasons started, 2nd-contract pay   outside yardsticks
Paired fix - base, 95% bootstrap over team-drafts.

Run:  python scripts/qb_sit_choice.py [workers=6]      report only:  ... report
"""
import os
import sys
import zlib

os.environ["OUTCOME_UNITS"] = "single"
os.environ["NEED_TABLE_DIR"] = "qbsit_tables"
os.environ["NEED_LEVELS"] = "current"
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plan_choice as pc  # noqa: E402
from need_choice import TDIR, _key, build_table  # noqa: E402
from outlook import GROUPS, OUT, RAW  # noqa: E402
from review import OFF_BOARD_RANK, board, draft_night  # noqa: E402

YEARS = list(range(2014, 2023))
pc.TABLE_DIR["single"] = TDIR


def weak(status):
    s = str(status)
    return s.startswith("rookie deal (later") or "below-average play" in s


def states(version):
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(YEARS)].copy()
    if version == "fix":
        S.loc[S.status.map(weak) & S.qb_from.notna(), "qb_from"] = 1.0
    if version == "played":  # scorer: any flagged need starts now
        S.loc[S.qb_from.notna(), "qb_from"] = 1.0
    return S


def one_year(y):
    from simulator import situation
    picks_df, _ = draft_night(y)
    b = board(y, picks_df)
    B = b[b.grp.isin(GROUPS)].reset_index(drop=True)
    B["rank"] = B["rank"].clip(upper=OFF_BOARD_RANK).astype(float)
    gnames = sorted(B.grp.unique())
    gidx = B.grp.map({g: i for i, g in enumerate(gnames)}).to_numpy()
    rows = []
    for v in ("base", "fix"):
        S = states(v)
        for _, srow in S[S.year == y].iterrows():
            tm, sit = srow.team, situation(srow)
            picks = sorted(int(k) for k in picks_df[picks_df.team == tm].pick)
            if not picks:
                continue
            before = [(B.drafted_at.notna() & (B.drafted_at < k)).to_numpy() for k in picks]
            p = pc.p_need(srow.status, "prob") if sit["qb_from"] is not None else 1.0
            V = pc.value_matrix(pc.Tab(sit, "single"), B, picks, p)
            ch = pc.team_draft(B, V, gidx, picks, before, "greedy", seed=y * 1000 + zlib.crc32(tm.encode()) % 997)
            for k, i in zip(picks, ch):
                if i is None:
                    continue
                r = B.iloc[i]
                rows.append({"version": v, "year": y, "team": tm, "pick": k, "player": r["name"], "grp": r.grp,
                             "rank": r["rank"], "drafted_at": r.drafted_at, "weak": weak(srow.status)})
    # the team's real picks
    for _, p_ in picks_df.iterrows():
        if p_.grp in GROUPS:
            S = states("base")
            st = S[(S.year == y) & (S.team == p_.team)]
            rows.append({"version": "team", "year": y, "team": p_.team, "pick": int(p_.pick), "player": p_.pfr_player_name,
                         "grp": p_.grp, "rank": np.nan, "drafted_at": float(p_.pick),
                         "weak": bool(len(st) and weak(st.status.iloc[0]))})
    print(f"{y}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


def score(P):
    from review import RealizedV2
    from simulator import situation
    from trade import Context
    ctx = Context()
    R = RealizedV2(ctx, units="single")
    for lab, v in [("t_rule", "base"), ("t_played", "played")]:
        S = states(v)
        sit = {(r.year, r.team): situation(r) for _, r in S.iterrows()}
        P[lab] = [R.titles(sit[(y, tm)], g, y, at, k, nm) if (y, tm) in sit else np.nan
                  for y, tm, g, at, k, nm in zip(P.year, P.team, P.grp, P.drafted_at, P.pick, P.player)]
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    pay = {(y, p): a for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}
    get = lambda d, y, a: (0.0 if pd.isna(a) else (d.get((y, int(a))) if pd.notna(d.get((y, int(a)))) else 0.0))
    P["av"] = [get(av, y, a) for y, a in zip(P.year, P.drafted_at)]
    P["gs"] = [get(gs, y, a) for y, a in zip(P.year, P.drafted_at)]
    P["pay"] = [get(pay, y, a) * 100 for y, a in zip(P.year, P.drafted_at)]
    return P


METRICS = [("t_rule", "titles, model rule"), ("t_played", "titles, as played"), ("av", "AV"), ("gs", "seasons started"),
           ("pay", "2nd-contract pay")]


def report(P):
    for m, _ in METRICS:
        P[m] = P[m].fillna(0.0)
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"

    base = P[P.version == "base"].set_index(["year", "team", "pick"])
    fix = P[P.version == "fix"].set_index(["year", "team", "pick"])
    ch = fix.player != base.player.reindex(fix.index)
    wk = fix.weak
    print(f"Choices changed by the fix: {ch.sum()} of {len(fix)} picks; at weak-incumbent teams {ch[wk].sum()} of {wk.sum()} "
          f"({fix[wk].reset_index()[['year', 'team']].drop_duplicates().shape[0]} team-drafts)")
    print("QBs taken per team-draft (weak-incumbent teams):",
          {v: round((P[(P.version == v) & P.weak].grp == "QB").sum() / max(1, P[(P.version == v) & P.weak][['year', 'team']].drop_duplicates().shape[0]), 2)
           for v in ("base", "fix", "team")})
    c = pd.DataFrame({"base": base.player.reindex(fix.index)[ch], "fix": fix.player[ch],
                      "base_grp": base.grp.reindex(fix.index)[ch], "fix_grp": fix.grp[ch]}).reset_index()
    print("\nChanged choices, round 1-2 picks:")
    print(c[c.pick <= 64].to_string(index=False))
    print("\nPer team-draft (sum over the team's picks), 95% bootstrap:")
    for scope, msk in [("all team-drafts", lambda d: d.index == d.index), ("weak-incumbent teams", lambda d: d.weak)]:
        print(f"-- {scope}")
        for m, lab in METRICS:
            x = {v: P[(P.version == v)][msk(P[P.version == v])].groupby(["year", "team"])[m].sum() for v in ("base", "fix", "team")}
            idx = x["team"].index
            f_b = x["fix"].reindex(idx, fill_value=0) - x["base"].reindex(idx, fill_value=0)
            b_t = x["base"].reindex(idx, fill_value=0) - x["team"]
            f_t = x["fix"].reindex(idx, fill_value=0) - x["team"]
            print(f"   {lab:20s} fix - base {boot(f_b)} | base - team {boot(b_t)} | fix - team {boot(f_t)}")


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    from simulator import situation
    sits = {}
    for v in ("base", "fix"):
        for _, r in states(v).iterrows():
            s = situation(r)
            sits[_key(s)] = s
    todo = [s for k, s in sits.items() if not (TDIR / f"{k}.npz").exists()]
    print(f"{len(sits)} situations, building {len(todo)} tables in {TDIR.name}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))
    with ProcessPoolExecutor(workers) as ex:
        P = pd.concat(list(ex.map(one_year, YEARS)), ignore_index=True)
    P = score(P)
    P.to_parquet(OUT / "qb_sit_choice.parquet")
    report(P)


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "qb_sit_choice.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
