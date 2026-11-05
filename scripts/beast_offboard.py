"""Can Brugler's grades sort the players Hasan's consensus board leaves off (rank > ~300)?

The model treats every off-board player alike (rank ~330, same expectation by position).
Brugler's guide (2019-22) grades ~400 players a year, ~100 of them off Hasan's board.
  1. Descriptive: among off-board players, actual 4-season value by Brugler grade.
  2. Choice test, picks 100+: the consensus model vs the same model with off-board players'
     expectations multiplied by a grade-based factor (Poisson, fit on the other drafts;
     on-board players untouched). Scored in value and in titles for a middle team.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import prospect_model as pm
from beast_choice import CLASSES, FAM2GRP
from outcomes_v2 import CF_OVER_OURS, REG
from outlook import OUT
from pick_value import baseline_paths, pick_effect
from trade import Context

BINS = [("Rd 3-4", 0, 4.5), ("Rd 5", 4.5, 5.5), ("Rd 6", 5.5, 6.5), ("Rd 7", 6.5, 7.5), ("Free agent", 7.5, 9)]


def grade_bin(g):
    for name, lo, hi in BINS:
        if lo <= g < hi:
            return name
    return "Not in guide"


def features(df):
    """One-hot grade bins for off-board players (0 for everyone else)."""
    X = pd.DataFrame(index=df.index)
    for name, *_ in BINS + [("Not in guide",)]:
        X[name] = ((df.on_board == 0) & (df.gbin == name)).astype(float)
    return X


def main():
    uni = pm.load()
    t = pd.read_parquet(OUT / "beast_test.parquet")[["player_id", "grade_round", "pos_rank"]].drop_duplicates("player_id")
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("consensus_pre",), test_classes=CLASSES)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R = R.merge(uni[["player_id", "cls", "on_board"]], on=["player_id", "cls"], how="left").merge(t, on="player_id", how="left")
    R["gbin"] = [grade_bin(g) if pd.notna(g) else "Not in guide" for g in R.grade_round]
    R["actual"] = R[[f"y{s}" for s in pm.SEASONS]].sum(axis=1)
    R["expected"] = R[[f"consensus_pre_p{s}" for s in pm.SEASONS]].sum(axis=1)
    off = R[R.on_board == 0]

    print("1. Off-board players (not on Hasan's board), 2019-22: actual 4-season value by Brugler grade")
    order = [b[0] for b in BINS] + ["Not in guide"]
    g = off.groupby("gbin").agg(players=("actual", "size"), drafted=("drafted", "mean"),
                                any_value=("actual", lambda x: (x > 0.005).mean()),
                                avg_value=("actual", "mean"), model_expected=("expected", "mean")).reindex(order)
    g["avg_value"] *= 100; g["model_expected"] *= 100
    print(g.round(3).to_string())
    from scipy.stats import spearmanr
    gi = off[off.grade_round.notna()]
    print(f"   rank correlation, grade vs actual (graded off-board players, n={len(gi)}): {spearmanr(-gi.grade_round, gi.actual)[0]:.3f}")
    print(f"   for comparison, consensus rank vs actual among board players ranked 200-300: "
          f"{spearmanr(-uni[(uni.cls.isin(CLASSES)) & uni.hasan_rank.between(200, 300)].hasan_rank, uni[(uni.cls.isin(CLASSES)) & uni.hasan_rank.between(200, 300)][[f'y{s}' for s in pm.SEASONS]].sum(axis=1))[0]:.3f}")

    # ---- 2. choice test
    from sklearn.linear_model import PoissonRegressor
    uni = uni.merge(t, on="player_id", how="left")
    uni["gbin"] = [grade_bin(g) if pd.notna(g) else "Not in guide" for g in uni.grade_round]
    mult_fn = {}
    for c in CLASSES:
        tr = off[(off.cls != c) & (off.expected > 1e-4)]
        X = features(tr).to_numpy()
        m = PoissonRegressor(alpha=1e-3, max_iter=1000, fit_intercept=False).fit(X, tr.actual / tr.expected, sample_weight=tr.expected)
        mult_fn[c] = m
        print(f"   {c}: off-board multipliers from other drafts: " +
              ", ".join(f"{k} x{np.exp(b):.2f}" for k, b in zip(features(tr).columns, m.coef_)))

    ctx = Context(); sit = {"start": 0.0, "shift": {}, "qb_from": 1}
    base = baseline_paths(sit, np.random.default_rng(0), 2000); b0 = ctx.sb(base).sum(axis=1).mean()
    cache = {}

    def titles(row, k):
        key = (row.player_id, k)
        if key not in cache:
            if row.grp not in CF_OVER_OURS:
                cache[key] = 0.0
            else:
                y = np.array([max(row[f"y{s}"], 0) for s in pm.SEASONS]) / CF_OVER_OURS[row.grp]
                y5 = np.r_[y, y[-1]]; price = max(y5.max(), 1e-6)
                pts = pick_effect((y5 / price)[None, :], np.array([price]), int(k), FAM2GRP.get(row.grp, row.grp), sit,
                                  ctx.cost, ctx.bar, ctx.rates)
                cache[key] = (ctx.sb(base + pts).sum(axis=1).mean() - b0) * 100
        return cache[key]

    rows = []
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        for c in CLASSES:
            te = uni[uni.cls == c].copy().reset_index(drop=True); tr = uni[uni.cls < c]
            te["y14"] = te[[f"y{s}" for s in pm.SEASONS]].sum(axis=1)
            Xte = features(te).to_numpy()
            mult = np.where(te.on_board == 0, np.exp(Xte @ mult_fn[c].coef_), 1.0)
            models = [pm.fit(tr[tr.cls + s - 1 <= c - 1], "consensus_pre", s) for s in pm.SEASONS]
            in_guide = te.grade_round.notna().to_numpy()
            onb = (te.on_board == 1).to_numpy()
            versions = {"consensus": np.ones(len(te)), "brugler": mult,
                        # fair versions: candidates = Hasan's board (like the app), or board + Brugler-graded players
                        "board_only": np.where(onb, 1.0, 0.0),
                        "board_plus_brugler": np.where(onb, 1.0, np.where(in_guide, mult, 0.0))}
            taken = {v: set() for v in versions}
            last = int(te[te.drafted].pick.max())
            for k in range(1, last + 1):
                avail = (te.pick >= k).to_numpy()
                cand = te.copy(); lk = np.log(k + 10)
                fell = cand.on_board.astype(bool) & (cand.log_rank < lk)
                cand.loc[fell, "log_rank"] += pm.FALL_WEIGHT * (lk - cand.loc[fell, "log_rank"])
                pred = sum(m.predict(cand[f]) for m, f in models)
                r = {"cls": c, "pick": k}
                for v, mlt in versions.items():
                    score = np.where(avail & ~te.player_id.isin(taken[v]).to_numpy(), pred * mlt, -np.inf)
                    i = int(np.argmax(score)); row = te.iloc[i]; taken[v].add(row.player_id)
                    r[f"{v}_value"] = row.y14; r[f"{v}_titles"] = titles(row, k)
                    r[f"{v}_player"] = row.player_name; r[f"{v}_offboard"] = bool(row.on_board == 0)
                rows.append(r)
            print(f"   {c} done", flush=True)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    ch = pd.DataFrame(rows)
    ch.to_parquet(OUT / "beast_offboard_choice.parquet")
    late = ch[ch.pick >= 100]
    print(f"\n2. Choice test, picks 100+ of 2019-22 ({len(late)} picks)")
    print(f"   model picked an off-board player: consensus {late.consensus_offboard.mean():.0%}, with Brugler {late.brugler_offboard.mean():.0%}; "
          f"choice changed at {(late.consensus_player != late.brugler_player).mean():.0%} of picks")
    rng = np.random.default_rng(0)
    for unit, scale, label in [("value", 100, "4-season value, % of cap"), ("titles", 1, "titles for a middle team, % of a title")]:
        d = ((late[f"brugler_{unit}"] - late[f"consensus_{unit}"]) * scale).to_numpy()
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"   {label}: consensus {late[f'consensus_{unit}'].mean() * scale:.3f}, with Brugler {late[f'brugler_{unit}'].mean() * scale:.3f}; "
              f"difference {d.mean():+.3f} per pick (95% {np.percentile(b, 2.5):+.3f} to {np.percentile(b, 97.5):+.3f})")
    print(f"   whole draft, titles: difference {((ch.brugler_titles - ch.consensus_titles)).mean():+.4f} per pick")
    print(f"\n3. Fair version, picks 100+: board-only model vs board + Brugler-graded off-board players")
    print(f"   picked a Brugler-graded off-board player at {late.board_plus_brugler_offboard.mean():.0%} of picks; "
          f"choice changed at {(late.board_only_player != late.board_plus_brugler_player).mean():.0%}")
    for unit, scale, label in [("value", 100, "4-season value, % of cap"), ("titles", 1, "titles for a middle team, % of a title")]:
        d = ((late[f"board_plus_brugler_{unit}"] - late[f"board_only_{unit}"]) * scale).to_numpy()
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"   {label}: board only {late[f'board_only_{unit}'].mean() * scale:.3f}, + Brugler {late[f'board_plus_brugler_{unit}'].mean() * scale:.3f}; "
              f"difference {d.mean():+.3f} per pick (95% {np.percentile(b, 2.5):+.3f} to {np.percentile(b, 97.5):+.3f})")
    print(f"   (for reference, the 'any college player' consensus model: {late.consensus_titles.mean():.3f} titles per pick)")


if __name__ == "__main__":
    main()
