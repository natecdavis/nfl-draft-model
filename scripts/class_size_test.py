"""Does a late pick return less when the team makes more picks in that draft? (README "Class-size test")

The model values each pick independently, but a team keeps only so many rookies: a strict follower who trades
down into 11-12 picks may be credited for players who can't all make the roster.

A. Model units, drafts 2014-22 (review_picks): realized titles minus the model's predicted titles for the team's
   own pick, rounds 5-7 (and 4-7), by the team's class size and number of late picks; realized / predicted by bin.
B. Raw outcomes, 2014-22: played as a rookie (any snaps in the draft season), seasons 1-4 value (prospect_model
   labels, % of cap) minus the average at the slot.
C. Long history, drafts 2000-2021: career weighted AV minus the average at the slot (draft-year effects), and
   ever played a game, on class size.

Slot average: OLS on a cubic in log(pick + 10) with draft-year effects, fit on the same rounds. Intervals:
bootstrap over team-drafts. Control: the team's prior strength (A, B).

Run:  python scripts/class_size_test.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prospect_model as pm  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from review import team as team_code  # noqa: E402

RNG = np.random.default_rng(0)


def slot_resid(d, y):
    """y minus its average at the slot (cubic in log pick, draft-year effects), within d."""
    lp = np.log(d.pick + 10)
    X = np.column_stack([lp, lp ** 2, lp ** 3, pd.get_dummies(d.year).to_numpy(float)])
    b = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)[0]
    return d[y].to_numpy(float) - X @ b


def slope(d, y, x, controls=()):
    """OLS slope of y on x (+ controls), 95% interval from a bootstrap over team-drafts."""
    def fit(q):
        X = np.column_stack([np.ones(len(q)), q[x]] + [q[c] for c in controls]).astype(float)
        return np.linalg.lstsq(X, q[y].to_numpy(float), rcond=None)[0][1]
    keys = d.groupby(["year", "team"]).indices
    ks = list(keys)
    bs = []
    for _ in range(1000):
        idx = np.concatenate([keys[ks[i]] for i in RNG.integers(0, len(ks), len(ks))])
        bs.append(fit(d.iloc[idx]))
    return f"{fit(d):+.4f} ({np.percentile(bs, 2.5):+.4f} to {np.percentile(bs, 97.5):+.4f})"


def bins(d, col, cuts, labels, cols):
    d = d.assign(bin=pd.cut(d[col], cuts, labels=labels))
    return d.groupby("bin", observed=True).agg(picks=("pick", "size"), teams=("tkey", "nunique"), **cols).round(3)


def add_sizes(d):
    d = d.sort_values(["year", "team", "pick"]).copy()
    d["tkey"] = d.year.astype(str) + d.team
    g = d.groupby(["year", "team"])
    d["n_class"] = g.pick.transform("size")
    d["n_late"] = g["round"].transform(lambda r: (r >= 5).sum())
    d["kth"] = g.cumcount() + 1  # his order among the team's picks
    d["kth_late"] = d[d["round"] >= 5].groupby(["year", "team"]).cumcount() + 1
    return d


def main():
    pd.set_option("display.width", 200)
    # ---------- A. model units ----------
    R = add_sizes(pd.read_parquet(OUT / "review_picks.parquet"))
    S = pd.read_parquet(OUT / "review_states.parquet")[["year", "team", "strength_prior"]]
    R = R.merge(S, on=["year", "team"], how="left")
    A = R[R.real_team.notna() & R.value_actual.notna()].copy()
    A["resid"] = A.real_team - A.value_actual
    print("Class sizes 2014-22 (team-drafts):", A.drop_duplicates("tkey").n_class.value_counts().sort_index().to_dict())
    print("corr(class size, prior strength):", round(A.drop_duplicates("tkey")[["n_class", "strength_prior"]].corr().iloc[0, 1], 2))
    for lab, lo in [("rounds 5-7", 5), ("rounds 4-7", 4)]:
        L = A[A["round"] >= lo]
        print(f"\nA. Model units, {lab} ({len(L)} picks): realized - predicted titles (% of one) per pick")
        agg = dict(pred=("value_actual", "mean"), real=("real_team", "mean"), resid=("resid", "mean"))
        print(bins(L, "n_class", [0, 7, 9, 20], ["<=7", "8-9", "10+"], agg).assign(ratio=lambda t: (t.real / t.pred).round(2)).to_string())
        print(bins(L, "n_late", [0, 3, 4, 20], ["<=3 late", "4 late", "5+ late"], agg).assign(ratio=lambda t: (t.real / t.pred).round(2)).to_string())
        print("  slope of resid per extra pick in the class:      ", slope(L, "resid", "n_class"))
        print("    controlling for prior strength:                ", slope(L, "resid", "n_class", ["strength_prior"]))
        print("  slope of resid per extra late pick (rounds 5-7): ", slope(L, "resid", "n_late", ["strength_prior"]))
    L = A[A["round"] >= 5]
    print("\n  by his order among the team's rounds 5-7 picks:")
    print(bins(L, "kth_late", [0, 1, 2, 3, 20], ["1st", "2nd", "3rd", "4th+"], dict(pred=("value_actual", "mean"), real=("real_team", "mean"), resid=("resid", "mean"))).to_string())
    T = A.groupby("tkey").agg(n_class=("n_class", "first"), pred=("value_actual", "sum"), real=("real_team", "sum"))
    T["bin"] = pd.cut(T.n_class, [0, 7, 9, 20], labels=["<=7", "8-9", "10+"])
    print("\n  whole class, per team-draft: predicted and realized titles (sum over picks)")
    print(T.groupby("bin", observed=True).agg(teams=("pred", "size"), pred=("pred", "mean"), real=("real", "mean")).round(2).to_string())

    # ---------- B. raw outcomes 2014-22 ----------
    lab = pm.load()
    lab = lab[lab.drafted & lab.overall_pick.notna()][["draft_year", "overall_pick", "y1", "y2", "y3", "y4", "pfr_id"]]
    lab["sum4"] = lab[["y1", "y2", "y3", "y4"]].sum(axis=1, min_count=4) * 100
    B = R[R.year <= 2022].merge(lab.rename(columns={"draft_year": "year", "overall_pick": "pick"}), on=["year", "pick"], how="left")
    sc = pd.read_parquet(RAW / "snap_counts.parquet", columns=["season", "pfr_player_id"]).drop_duplicates()
    played = set(zip(sc.season, sc.pfr_player_id))
    B["rookie_played"] = [float((y, i) in played) for y, i in zip(B.year, B.pfr_id)]
    for lab_, lo in [("rounds 5-7", 5), ("rounds 4-7", 4)]:
        L = B[(B["round"] >= lo) & B.sum4.notna()].copy()
        L["v_resid"] = slot_resid(L, "sum4")
        L["p_resid"] = slot_resid(L, "rookie_played")
        print(f"\nB. Raw outcomes 2014-22, {lab_} ({len(L)} picks)")
        agg = dict(played=("rookie_played", "mean"), played_vs_slot=("p_resid", "mean"), value4=("sum4", "mean"), value_vs_slot=("v_resid", "mean"))
        print(bins(L, "n_class", [0, 7, 9, 20], ["<=7", "8-9", "10+"], agg).to_string())
        print(bins(L, "n_late", [0, 3, 4, 20], ["<=3 late", "4 late", "5+ late"], agg).to_string())
        print("  slope per extra pick in the class, played as a rookie (vs slot):", slope(L, "p_resid", "n_class", ["strength_prior"]))
        print("  slope per extra pick in the class, seasons 1-4 value (vs slot): ", slope(L, "v_resid", "n_class", ["strength_prior"]))

    # ---------- C. long history ----------
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dp = dp[dp.season.between(2000, 2021)].rename(columns={"season": "year"}).copy()
    dp["team"] = dp.team.map(team_code)
    dp = add_sizes(dp)
    dp["av"] = dp.w_av.fillna(0.0)
    dp["ever_played"] = (dp.games.fillna(0) > 0).astype(float)
    for lab_, lo in [("rounds 5-7", 5), ("rounds 4-7", 4), ("all rounds", 1)]:
        L = dp[dp["round"] >= lo].copy()
        L["av_resid"] = slot_resid(L, "av")
        L["pl_resid"] = slot_resid(L, "ever_played")
        print(f"\nC. Drafts 2000-2021, {lab_} ({len(L)} picks): career weighted AV and ever played, vs slot")
        agg = dict(av=("av", "mean"), av_vs_slot=("av_resid", "mean"), ever_played=("ever_played", "mean"), played_vs_slot=("pl_resid", "mean"))
        print(bins(L, "n_class", [0, 6, 7, 8, 9, 10, 30], ["<=6", "7", "8", "9", "10", "11+"], agg).to_string())
        print("  slope of AV vs slot per extra pick in the class:  ", slope(L, "av_resid", "n_class"))
        print("  slope of ever played vs slot per extra pick:      ", slope(L, "pl_resid", "n_class"))
    L = dp[dp["round"] >= 5].copy()
    L["av_resid"] = slot_resid(L, "av")
    L["pl_resid"] = slot_resid(L, "ever_played")
    print("\n  rounds 5-7 by his order among the team's rounds 5-7 picks:")
    print(bins(L, "kth_late", [0, 1, 2, 3, 4, 30], ["1st", "2nd", "3rd", "4th", "5th+"], dict(av_vs_slot=("av_resid", "mean"), played_vs_slot=("pl_resid", "mean"))).to_string())
    print("  slope of AV vs slot per extra late pick:", slope(L, "av_resid", "n_late"))
    out = L[["year", "team", "pick", "round", "n_class", "n_late", "kth_late", "av", "av_resid", "ever_played"]]
    out.to_parquet(OUT / "class_size_test.parquet")


if __name__ == "__main__":
    main()
