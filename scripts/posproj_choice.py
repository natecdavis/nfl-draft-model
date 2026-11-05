"""Projecting position changes: does valuing board players across the NFL families their board label
historically moved to improve whole-draft choices? Drafts 2014-2022.

  cur  today's model: a drafted player's position comes from his draft record (the position his
       team listed: known only once he's drafted, so for players not yet taken it is hindsight)
  L0   the consensus board's own label only (what was known before the draft)
  L1   board label, valued as a blend over NFL families: P(family he was priced at in the NFL | board
       label) from drafted board players who played, leaving out the draft being scored, with 5
       pseudo-players kept in the label's own family. E.g. a 5-technique (DL5T) ~ 60% DL / 40% EDGE.

Everything else as scripts/plan_choice.py version B (greedy, single units, QB need probability).
Realized value for every version uses the draft-record position, so the yardstick is identical.
A blended player's need level is that of his board-label position.

Run:  python scripts/posproj_choice.py [workers=6]      report only: ... report
"""
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plan_choice as pc  # noqa: E402
from outlook import GROUPS, OUT  # noqa: E402
from review import HASAN_POS, OFF_BOARD_RANK, board, draft_night  # noqa: E402

FAM = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "ED": "EDGE", "IDL": "DL",
       "LB": "LB", "CB": "DB", "S": "DB"}
TARGET = {"EDGE": "ED", "DL": "IDL", "LB": "LB", "DB": "S", "OL": "G", "RB": "RB", "WR": "WR", "TE": "TE", "QB": "QB"}
PRIOR = 5.0
SCRATCH = OUT / "posproj_moves.parquet"


def moves() -> pd.DataFrame:
    """Drafted board players 2014-2024 who played: year, board label, NFL family priced at."""
    if SCRATCH.exists():
        return pd.read_parquet(SCRATCH)
    L = pd.read_parquet(Path.home() / "cfdarko/data/processed/rookie_value_labels.parquet").dropna(subset=["pfr_id"])
    L = L.drop_duplicates("pfr_id").set_index("pfr_id")
    dp = pd.read_parquet("data/raw/draft_picks.parquet")
    pid = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    rows = []
    for y in range(2014, 2025):
        pk, _ = draft_night(y)
        b = board(y, pk)
        b = b[b.drafted_at.notna()]
        for pos, at in zip(b.pos, b.drafted_at):
            p_ = pid.get((y, int(at)))
            if p_ in L.index and (np.nan_to_num(L.at[p_, "value_y1"]) + np.nan_to_num(L.at[p_, "value_y2"])) > 0:
                rows.append({"year": y, "pos": pos, "nfl": L.at[p_, "group"]})
    M = pd.DataFrame(rows)
    M.to_parquet(SCRATCH)
    return M


def blend_weights(M, leave_out):
    """{board label: {target group: weight}} for labels that ever moved."""
    d = M[M.year != leave_out]
    out = {}
    for pos, g in d.groupby("pos"):
        own = FAM.get(HASAN_POS.get(pos, "SPEC"))
        if own is None:
            continue
        c = g.nfl.value_counts().to_dict()
        c[own] = c.get(own, 0) + PRIOR
        tot = sum(c.values())
        w = {}
        for f, n in c.items():
            tgt = HASAN_POS[pos] if f == own else TARGET.get(f)
            if tgt in GROUPS:
                w[tgt] = w.get(tgt, 0) + n / tot
        s = sum(w.values())
        out[pos] = {k: v / s for k, v in w.items()}
    return out


def one_year(y):
    from simulator import situation
    t0 = time.time()
    M = moves()
    W = blend_weights(M, y)
    picks_df, _ = draft_night(y)
    b = board(y, picks_df)
    b["grp_true"] = b.grp
    b["grp_label"] = b.pos.map(HASAN_POS).fillna("SPEC")
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year == y]
    rows = []
    for v in ("cur", "L0", "L1"):
        gcol = "grp_true" if v == "cur" else "grp_label"
        B = b[b[gcol].isin(GROUPS)].reset_index(drop=True).copy()
        B["grp"] = B[gcol]
        B["rank"] = B["rank"].clip(upper=OFF_BOARD_RANK).astype(float)
        gnames = sorted(B.grp.unique())
        gidx = B.grp.map({g: i for i, g in enumerate(gnames)}).to_numpy()
        for _, srow in S.iterrows():
            tm, sit = srow.team, situation(srow)
            picks = sorted(int(k) for k in picks_df[picks_df.team == tm].pick)
            if not picks:
                continue
            p = pc.p_need(srow.status, "prob") if sit["qb_from"] is not None else 1.0
            tab = pc.Tab(sit, "single")
            V = pc.value_matrix(tab, B, picks, p)
            if v == "L1":  # blend over the families each label moved to
                Vb = np.zeros_like(V)
                wsum = np.zeros(len(B))
                alts = {}
                for i, (pos, g) in enumerate(zip(B.pos, B.grp)):
                    for tgt, w in W.get(pos, {g: 1.0}).items():
                        alts.setdefault(tgt, []).append((i, w))
                for tgt, lst in alts.items():
                    idx = [i for i, _ in lst]
                    Bt = B.iloc[idx].copy(); Bt["grp"] = tgt
                    Vt = V[idx] if (Bt.grp.to_numpy() == B.grp.to_numpy()[idx]).all() else pc.value_matrix(tab, Bt, picks, p)
                    for r, (i, w) in enumerate(lst):
                        Vb[i] += w * Vt[r]; wsum[i] += w
                V = Vb / wsum[:, None, None]
            before = [(B.drafted_at.notna() & (B.drafted_at < k)).to_numpy() for k in picks]
            ch = pc.team_draft(B, V, gidx, picks, before, "greedy", seed=y * 1000 + zlib.crc32(tm.encode()) % 997)
            for k, i in zip(picks, ch):
                if i is None:
                    continue
                r = B.iloc[i]
                rows.append({"version": v, "year": y, "team": tm, "pick": k, "player": r["name"], "label": r.pos,
                             "grp_used": r.grp, "grp": r.grp_true if r.grp_true in GROUPS else r.grp,
                             "rank": r["rank"], "drafted_at": r.drafted_at})
    print(f"{y}: {len(rows)} choices, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    moves()  # build once before the workers start
    with ProcessPoolExecutor(workers) as ex:
        P = pd.concat(list(ex.map(one_year, pc.YEARS)), ignore_index=True)
    import outcomes_v2
    outcomes_v2.POS_PAY = "off"
    P = pc.score(P)
    P.to_parquet(OUT / "posproj_choice.parquet")
    report(P)


def report(P):
    U = pd.read_parquet(OUT / "units_choice.parquet")
    rnd = dict(zip(zip(U.year, U.pick), U["round"]))
    P["round"] = [rnd.get((y, k)) for y, k in zip(P.year, P.pick)]
    M = [("t_single", "titles"), ("av", "PFR AV"), ("gs", "seasons started"), ("pay", "2nd-contract pay, % cap")]
    for m, _ in M:
        P[m] = P[m].fillna(0.0)
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"

    G = [("All", lambda r: r > 0), ("R1", lambda r: r == 1), ("R2-4", lambda r: (r >= 2) & (r <= 4)), ("R5-7", lambda r: r >= 5)]
    for a_, b_ in [("L1", "L0"), ("cur", "L0")]:
        print(f"\n{a_} - {b_} per team-draft (288):")
        for m, lab in M:
            parts = []
            for gl, f in G:
                x = {v: P[(P.version == v) & f(P["round"])].groupby(["year", "team"])[m].sum() for v in (a_, b_)}
                idx = x[a_].index.union(x[b_].index)
                parts.append(f"{gl} {boot(x[a_].reindex(idx, fill_value=0) - x[b_].reindex(idx, fill_value=0))}")
            print(f"  {lab:24s} " + " | ".join(parts))
        A = P[P.version == a_].set_index(["year", "team", "pick"]); B_ = P[P.version == b_].set_index(["year", "team", "pick"])
        print(f"  same choice: {(A.player == B_.player.reindex(A.index)).mean():.0%}")
    L1 = P[P.version == "L1"]
    print("\nL1 picks by board label (top):", L1.label.value_counts().head(10).to_dict())


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "posproj_choice.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
