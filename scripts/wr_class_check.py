"""Does the model's round-1 WR edge rest on one class (2020) or a few hits?

Round-1 picks where the Past drafts review recommends a WR (review_picks.parquet, current model) and the team
took someone else. Three comparisons per pick, realized titles for this team (review.RealizedV2, single units):
  WR rec - team's pick       the scorecard comparison
  WR rec - best non-WR       the model's best available non-QB, non-WR player at that pick (same value tables
                             as scripts/stack_choice.py version B, level = team's earlier picks at his group):
                             is preferring the WR what earns the edge?
  WR rec - best board        consensus board's best available player
Unit: each recommended WR once per draft (mean over the picks where he was the rec), as review.once_per_draft.
Classes 2014-22 fully scored; 2023-25 "so far" (titles_many partial=True: seasons played only, value and cost)
as a fresh check. Also AV / seasons started / 2nd-contract pay for 2014-22.

Run:  python scripts/wr_class_check.py
"""
import os
import sys

os.environ["OUTCOME_UNITS"] = "single"
os.environ["POS_PAY"] = "off"
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import GROUPS, OUT, RAW  # noqa: E402

FW = 0.85
pd.set_option("display.width", 220)


def alternatives(R):
    from review import K_GRID, OFF_BOARD_RANK, T_GRID, board, draft_night
    from simulator import situation
    from stack_choice import TDIR, _key
    St = pd.read_parquet(OUT / "review_states.parquet")
    sit = {(r.year, r.team): situation(r) for _, r in St.iterrows()}
    x, yk = np.log(T_GRID + 10), np.log(K_GRID + 10)

    def bilin(A, rank, pick):
        t = rank if pick <= rank else np.exp(np.log(rank + 10) + FW * (np.log(pick + 10) - np.log(rank + 10))) - 10
        lt, lk = np.log(t + 10), np.log(pick + 10)
        i = int(np.clip(np.searchsorted(x, lt) - 1, 0, len(x) - 2)); j = int(np.clip(np.searchsorted(yk, lk) - 1, 0, len(yk) - 2))
        fx = np.clip((lt - x[i]) / (x[i + 1] - x[i]), 0, 1); fy = np.clip((lk - yk[j]) / (yk[j + 1] - yk[j]), 0, 1)
        return (1 - fx) * (1 - fy) * A[..., i, j] + fx * (1 - fy) * A[..., i + 1, j] + (1 - fx) * fy * A[..., i, j + 1] + fx * fy * A[..., i + 1, j + 1]

    out = []
    for y, d in R.groupby("year"):
        picks_df, _ = draft_night(y)
        b = board(y, picks_df)
        B = b[b.grp.isin(GROUPS) & ~b.grp.isin(["QB", "WR"])].reset_index(drop=True)
        B["rank"] = B["rank"].clip(upper=OFF_BOARD_RANK).astype(float)
        dat = B.drafted_at.to_numpy(float)
        for idx, tm, k in zip(d.index, d.team, d.pick):
            s = sit[(y, tm)]
            V = np.load(TDIR / f"{_key(s)}.npz")
            earlier = picks_df[(picks_df.team == tm) & (picks_df.pick < k)].grp.value_counts().to_dict()
            avail = np.where(~(np.nan_to_num(dat, nan=1e9) < k))[0]
            vals = [bilin(V[B.grp[i]][min(earlier.get(B.grp[i], 0), 2)], B["rank"][i], k) for i in avail]
            i = avail[int(np.argmax(vals))]
            out.append((idx, B.loc[i, "name"], B.loc[i, "grp"], dat[i]))
    A = pd.DataFrame(out, columns=["idx", "alt_player", "alt_grp", "alt_drafted_at"]).set_index("idx")
    return R.join(A)


def realized(R):
    from review import RealizedV2
    from simulator import situation
    from trade import Context
    St = pd.read_parquet(OUT / "review_states.parquet")
    sit = {(r.year, r.team): situation(r) for _, r in St.iterrows()}
    RV = RealizedV2(Context(), units="single")
    for col, g, at, nm in [("t_team", "grp", "pick", "player"), ("t_wr", "rec_grp", "rec_drafted_at", "rec_player"),
                           ("t_alt", "alt_grp", "alt_drafted_at", "alt_player"),
                           ("t_board", "best_available_grp", "best_available_drafted_at", "best_available")]:
        R[col] = [float(RV.titles_many(sit[(y, tm)], y, k, [gg], [a], [n_], partial=y > 2022)[0])
                  for y, tm, gg, a, k, n_ in zip(R.year, R.team, R[g], R[at], R.pick, R[nm])]
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    po = pd.read_parquet(OUT / "pick_outcomes.parquet")
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    pay = {(y, p): a * 100 for y, p, a in zip(po.draft_year, po.pick, po.best_pay)}
    get = lambda dct, y, a: 0.0 if pd.isna(a) or pd.isna(dct.get((y, int(a)))) else dct.get((y, int(a)))
    for lab, dct in [("av", av), ("gs", gs), ("pay", pay)]:
        for who, at in [("team", "pick"), ("wr", "rec_drafted_at"), ("alt", "alt_drafted_at")]:
            R[f"{lab}_{who}"] = [get(dct, y, a) for y, a in zip(R.year, R[at])]
    return R


def main():
    import stack_choice as sc
    sc.YEARS = list(range(2014, 2026))  # tables for the fresh classes too
    sc.build(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[(R["round"] == 1) & (R.rec_grp == "WR") & (R.player != R.rec_player) & R.year.between(2014, 2025)].copy()
    R = alternatives(R)
    R = realized(R)
    R["d_team"] = R.t_wr - R.t_team
    R["d_alt"] = R.t_wr - R.t_alt
    R["d_board"] = R.t_wr - R.t_board
    for lab in ["av", "gs", "pay"]:
        R[f"{lab}_d_team"] = R[f"{lab}_wr"] - R[f"{lab}_team"]
        R[f"{lab}_d_alt"] = R[f"{lab}_wr"] - R[f"{lab}_alt"]
    R.to_parquet(OUT / "wr_class_check.parquet")
    num = [c for c in R.select_dtypes("number").columns if c not in ("year",)]
    U = R.groupby(["year", "rec_player"], as_index=False).agg({**{c: "mean" for c in num}, "alt_player": "first",
                                                               "player": "first"})
    U["n_picks"] = R.groupby(["year", "rec_player"]).size().values
    full = U[U.year <= 2022]

    print("Round-1 WR recommendations the team passed on, each WR once per draft (titles x100 = % of a title)")
    by = U.groupby("year").agg(n_wr=("rec_player", "size"), n_picks=("n_picks", "sum"), wr=("t_wr", "mean"),
                               team=("t_team", "mean"), alt=("t_alt", "mean"), board=("t_board", "mean"),
                               d_team=("d_team", "mean"), d_alt=("d_alt", "mean"), d_board=("d_board", "mean"))
    print(by.round(2).to_string())

    rng = np.random.default_rng(0)

    def cboot(d, col, B=4000):
        """mean over WRs, bootstrap resampling whole classes"""
        ys = d.year.unique()
        g = {y: d[d.year == y][col].to_numpy() for y in ys}
        st = []
        for _ in range(B):
            s = rng.choice(ys, len(ys))
            v = np.concatenate([g[y] for y in s])
            st.append(v.mean())
        return f"{d[col].mean():+.2f} ({np.percentile(st, 2.5):+.2f} to {np.percentile(st, 97.5):+.2f})"

    print(f"\n2014-22 pooled ({len(full)} WRs; 95% CI resampling classes):")
    for col, lab in [("d_team", "WR rec - team's pick"), ("d_alt", "WR rec - best non-WR"), ("d_board", "WR rec - board best"),
                     ("av_d_team", "AV vs team"), ("av_d_alt", "AV vs non-WR"), ("gs_d_team", "starts vs team"),
                     ("gs_d_alt", "starts vs non-WR"), ("pay_d_team", "pay vs team"), ("pay_d_alt", "pay vs non-WR")]:
        print(f"  {lab:24s} {cboot(full, col)}")
    print("\nLeave one class out (2014-22), mean per WR:")
    rows = []
    for y in range(2014, 2023):
        d = full[full.year != y]
        rows.append({"without": y, "d_team": d.d_team.mean(), "d_alt": d.d_alt.mean(), "d_board": d.d_board.mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nDrop the top-k WRs by realized titles (2014-22):")
    srt = full.sort_values("t_wr", ascending=False)
    print("  top 8: " + ", ".join(f"{p} {y} ({t:.1f})" for p, y, t in zip(srt.rec_player[:8], srt.year[:8], srt.t_wr[:8])))
    for k in [0, 1, 3, 5, 10]:
        d = srt.iloc[k:]
        print(f"  drop {k:2d}: vs team {d.d_team.mean():+.2f}  vs non-WR {d.d_alt.mean():+.2f}  vs board {d.d_board.mean():+.2f}"
              f"  (n {len(d)}); median vs non-WR {d.d_alt.median():+.2f}, share WR better {np.mean(d.d_alt > 0):.0%}")
    print("\nWhat the non-WR alternatives were (2014-22):", full.merge(R[["year", "rec_player", "alt_grp"]].drop_duplicates(
        ["year", "rec_player"]), on=["year", "rec_player"]).alt_grp.value_counts().to_dict())
    fresh = U[U.year >= 2023]
    print(f"\nFresh classes 2023-25, seasons so far ({len(fresh)} WRs):")
    for col, lab in [("d_team", "WR rec - team's pick"), ("d_alt", "WR rec - best non-WR"), ("d_board", "WR rec - board best")]:
        print(f"  {lab:24s} {cboot(fresh, col)}")
    print(fresh.groupby("year")[["t_wr", "t_team", "t_alt", "d_team", "d_alt"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
