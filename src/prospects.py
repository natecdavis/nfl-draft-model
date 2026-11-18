"""Better player outlook, stage 1: does pre-draft information beat draft slot?

Draft slot already contains everything teams knew. So the test is whether
athletic testing, size and age predict which players beat or miss their slot.

Same outcome tiers as outlook.py (star / starter / bust), drafts 2012-2021.
Inputs added on top of the slot-only model (log pick + position):
  age        age on draft day
  size/speed height, weight, 40, vertical, broad jump, 3-cone, shuttle, bench,
             and speed score (weight x 200 / 40^4), each as a z-score within the
             player's position group, so "fast" means fast for a guard
  no-test    did not run the 40 / was not at the combine (often a tell)

Every version is scored the honest way: each draft 2016-2021 predicted from
earlier drafts only. The slot-only model from outlook.py is the bar to beat.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from college_features import FEATURES as CF, LOWER_IS_BETTER, features
from outlook import GROUPS, OUT, PICK_OFFSET, RAW

TESTS = ["ht_in", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench", "speed_score"]
FAST_IS_LOW = {"forty", "cone", "shuttle"}  # lower time = better; flipped so + is good
FAMILY = {"QB": "qb", "RB": "skill", "WR": "skill", "TE": "skill", "CB": "skill", "S": "skill",
          "T": "big", "G": "big", "C": "big", "IDL": "big", "ED": "front", "LB": "front"}


def build() -> pd.DataFrame:
    out = pd.read_parquet(OUT / "pick_outcomes.parquet")
    out = out[out.grp.isin(GROUPS)].copy()
    comb = pd.read_parquet(RAW / "combine.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    ft, inch = comb.ht.str.split("-", expand=True).astype(float).T.values
    comb["ht_in"] = ft * 12 + inch
    comb["speed_score"] = comb.wt * 200 / comb.forty ** 4
    df = out.merge(comb[["pfr_id"] + TESTS], left_on="pfr_id", right_on="pfr_id", how="left")
    df["at_combine"] = df.wt.notna().astype(int)
    df["no_forty"] = (df.at_combine.astype(bool) & df.forty.isna()).astype(int)

    players = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    birth = pd.to_datetime(players.set_index("pfr_id").birth_date, errors="coerce")
    draft_day = pd.to_datetime(df.draft_year.astype(str) + "-04-28")
    df["age"] = (draft_day - df.pfr_id.map(birth)).dt.days / 365.25

    # z-scores within position group; missing -> 0 (average), flagged separately
    for t in TESTS + ["age"]:
        z = df.groupby("grp")[t].transform(lambda s: (s - s.mean()) / s.std())
        df[f"z_{t}"] = (-z if t in FAST_IS_LOW else z).fillna(0)
    df["lp"] = np.log(df.pick + PICK_OFFSET)
    df["family"] = df.grp.map(FAMILY)
    return add_board(add_college(df))


def add_board(df: pd.DataFrame) -> pd.DataFrame:
    """ESPN's pre-draft big board (via github.com/JackLich10/nfl-draft-data; no license
    stated, so used for analysis only, not redistributed). Joined by draft year + pick.
      board_gap  log(pick+10) - log(board rank+10): + = fell below where ESPN ranked him
      grade_z    ESPN grade, z-scored within draft year
      unranked   drafted but not on ESPN's board (treated as ranked just past the board)"""
    b = pd.read_csv(RAW / "lich" / "nfl_draft_prospects.csv", low_memory=False)
    b = b.dropna(subset=["overall"]).drop(columns=["pick"]).rename(columns={"overall": "pick"})
    b = b[["draft_year", "pick", "ovr_rk", "grade"]].drop_duplicates(["draft_year", "pick"])
    df = df.merge(b, on=["draft_year", "pick"], how="left")
    last = df.groupby("draft_year").ovr_rk.transform("max")
    df["unranked"] = df.ovr_rk.isna().astype(int)
    rank = df.ovr_rk.fillna(last + 50)
    df["board_gap"] = np.log(df.pick + PICK_OFFSET) - np.log(rank + PICK_OFFSET)
    df["grade_z"] = df.groupby("draft_year").grade.transform(lambda x: (x - x.mean()) / x.std()).fillna(0)
    return df


def add_college(df: pd.DataFrame) -> pd.DataFrame:
    """College production features (college_features.py), as z-scores within position."""
    dp = pd.read_parquet(RAW / "draft_picks.parquet")[["season", "pick", "college"]]
    df = df.merge(dp.rename(columns={"season": "draft_year"}), on=["draft_year", "pick"], how="left")
    cf = features(df)
    df = df.merge(cf, on=["draft_year", "pick"], how="left")
    has_stats = df.grp.isin(list(CF))
    df["cfb_missing"] = (has_stats & df.cfb_seasons.isna()).astype(int)
    cols = sorted({c for v in CF.values() for c in v})
    for c in cols:
        z = df.groupby("grp")[c].transform(lambda x: (x - x.mean()) / x.std())
        df[f"c_{c}"] = (-z if c in LOWER_IS_BETTER else z).fillna(0)
    return df


def design(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Columns for each model version. All versions include log pick + position.
    Versions: slot only | age | age+athletic | age+athletic by family | age+athletic+college"""
    X = pd.get_dummies(df.grp, prefix="g", drop_first=True, dtype=float)
    X.insert(0, "lp", df.lp)
    if version in ("slot + board", "age+athletic+board"):
        X["board_gap"] = df.board_gap
        X["unranked"] = df.unranked
        X["grade_z"] = df.grade_z
    if version in ("slot only", "slot + board"):
        return sm.add_constant(X, has_constant="add")
    X["z_age"] = df.z_age
    if version == "age":
        return sm.add_constant(X, has_constant="add")
    X["at_combine"] = df.at_combine
    X["no_forty"] = df.no_forty
    tests = [f"z_{t}" for t in TESTS]
    if version == "age+athletic by family":
        # athletic traits may matter differently for big men, front seven, skill players
        for fam in ["skill", "big", "front"]:
            for t in tests:
                X[f"{t}:{fam}"] = df[t] * (df.family == fam)
    else:
        X[tests] = df[tests]
    if version == "age+athletic+college":
        X["cfb_missing"] = df.cfb_missing
        for c in [c for c in df.columns if c.startswith("c_")]:
            X[c] = df[c]
    return sm.add_constant(X, has_constant="add")


def fit_predict(train, test, version, alpha):
    """Two ridge logistic models (star; starter or better). Returns [bust, starter, star]."""
    Xtr, Xte = design(train, version), design(test, version)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    # no penalty on intercept, slot and position terms; ridge on the new inputs
    pen = np.array([0.0 if (c == "const" or c == "lp" or c.startswith("g_")) else alpha for c in Xtr.columns])
    probs = []
    for y in [(train.score == 2), (train.score >= 1)]:
        m = sm.GLM(y.astype(float), Xtr, family=sm.families.Binomial()).fit_regularized(
            alpha=pen, L1_wt=0.0, maxiter=300)
        probs.append(np.clip(1 / (1 + np.exp(-(Xte.to_numpy() @ m.params.to_numpy()))), 1e-4, 1 - 1e-4))
    p_star, p_plus = probs[0], np.maximum(probs[1], probs[0])
    pr = np.clip(np.column_stack([1 - p_plus, p_plus - p_star, p_star]), 1e-4, 1)
    return pr / pr.sum(axis=1, keepdims=True), m


def rolling(df, version, alpha):
    ll, rows = [], []
    for yr in range(2016, 2022):
        tr, te = df[df.draft_year < yr], df[df.draft_year == yr]
        pr, _ = fit_predict(tr, te, version, alpha)
        ll += list(-np.log(pr[np.arange(len(te)), te.score.to_numpy()]))
        rows.append(te.assign(p_star=pr[:, 2], p_bust=pr[:, 0]))
    return np.mean(ll), pd.concat(rows)


def main():
    df = build()
    df.to_parquet(OUT / "prospect_features.parquet")
    print(f"Picks 2012-2021: {len(df)}; at combine: {df.at_combine.mean():.0%}; age known: {df.age.notna().mean():.0%}")
    st = df[df.grp.isin(list(CF))]
    print("Drafted players on ESPN's board:", f"{(1 - df.unranked).mean():.0%}")
    print("College stats matched, by position:", st.groupby("grp").cfb_seasons.apply(lambda x: f"{x.notna().mean():.0%}").to_dict(), "\n")

    print("Rolling test, drafts 2016-2021 predicted from earlier drafts (avg log loss, lower is better):")
    results = {}
    for version in ["slot only", "slot + board", "age", "age+athletic", "age+athletic by family",
                    "age+athletic+college", "age+athletic+board"]:
        for alpha in ([0.0] if version == "slot only" else [0.01, 0.03, 0.1, 0.3]):
            ll, rows = rolling(df, version, alpha)
            results[(version, alpha)] = (ll, rows)
            print(f"   {version:24s} penalty {alpha:<6} {ll:.4f}")
    base = results[("slot only", 0.0)][0]
    best = min(results, key=lambda k: results[k][0])
    print(f"\nBest: {best[0]} (penalty {best[1]}): {results[best][0]:.4f} vs slot only {base:.4f} "
          f"({(base - results[best][0]) / base:.1%} better)")

    # Paired check: is the gain bigger than chance? Bootstrap over draft classes x players.
    a = results[("slot only", 0.0)][1]; b = results[best][1]
    def ll3(r):
        p = np.column_stack([r.p_bust, 1 - r.p_bust - r.p_star, r.p_star])
        return -np.log(np.clip(p[np.arange(len(r)), r.score.to_numpy()], 1e-4, 1))
    diff = ll3(a) - ll3(b)
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    print(f"Gain per player: {diff.mean():.4f} (95% range {np.percentile(boots, 2.5):.4f} to {np.percentile(boots, 97.5):.4f})")

    print("\nGain over slot only by position (avg log loss reduction per player, + = better):")
    by = {}
    for v in ["slot + board", "age+athletic", "age+athletic+college", "age+athletic+board"]:
        r = results[(v, 0.1)][1]
        by[v] = pd.Series(ll3(a) - ll3(r), index=a.index).groupby(a.grp.values).mean()
    print(pd.DataFrame(by).round(3).T.to_string())

    # What the best model learned, fit on all drafts
    _, m = fit_predict(df, df.iloc[:5], best[0], best[1])
    coef = m.params[[c for c in m.params.index if c.startswith(("z_", "c_")) or c in ("at_combine", "no_forty", "cfb_missing", "board_gap", "unranked", "grade_z")]]
    print("\nWhat moves a player's star chance beyond his slot (log-odds per 1 sd; + = better):")
    print(coef[coef.abs() > 0.05].sort_values().round(2).to_string())


if __name__ == "__main__":
    main()
