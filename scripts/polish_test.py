"""Is the word-length finding the board rewarding polish that doesn't matter on the field?

The pricing test (transcripts_test.reach) only compares teams with the board; if both reward polish, it reads ~1.
Two direct checks, word_len standardized as in transcripts_test (>= 200 words):
  1. RANK: does word_len buy a better board rank than the player's measurables / production justify?
     Stats-only expected log(board rank): LightGBM on PLAYER features + position (prospect_model.PLAYER), 5-fold
     cross-fitted over board players 2014-25 (whole board, not only transcript players). Residual (actual minus
     expected log rank; < 0 = ranked better than stats) regressed on word_len, log words, position + class FE.
  2. VALUE vs a stats-only outcome model ('player_pre' = PLAYER features, no rank or pick), rolling out of
     sample, board players 2018-22. If the board overrates polish, word_len ~ 1 here but < 1 vs consensus;
     if long words flag something bad the stats miss too, < 1 in both.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from outcomes_v2 import REG
from outlook import OUT
from transcripts_test import features

X_COLS = pm.PLAYER + ["grp_code"]


def expected_rank(uni):
    b = uni[(uni.on_board == 1) & uni.hasan_rank.notna() & (uni.cls >= pm.FIRST_BOARD_CLASS)].copy()
    b["lr"] = np.log(b.hasan_rank)
    b["exp_lr"] = np.nan
    for tr, te in KFold(5, shuffle=True, random_state=0).split(b):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              colsample_bytree=0.6, verbose=-1, random_state=0)
        m.fit(b.iloc[tr][X_COLS], b.iloc[tr].lr, categorical_feature=["grp_code"])
        b.iloc[te, b.columns.get_loc("exp_lr")] = m.predict(b.iloc[te][X_COLS])
    print(f"stats-only rank model: board players {len(b)}, corr(expected, actual log rank) {b.exp_lr.corr(b.lr):.2f}")
    return b[["player_id", "cls", "lr", "exp_lr"]]


def z(s):
    return (s - s.mean()) / s.std()


def main():
    uni = pm.load()
    uni["player_id"] = uni.player_id.astype(str)
    a = pd.read_parquet(OUT / "asap_answers_2011_22.parquet")
    a["player_id"] = a.player_id.astype(str)
    f = features(a)
    f = f[f.n_words >= 200]

    # 1. rank vs stats
    r = expected_rank(uni).merge(f, on=["player_id", "cls"]).merge(uni[["player_id", "cls", "grp"]], on=["player_id", "cls"])
    r["z"], r["log_words"], r["resid"] = z(r.word_len), np.log(r.n_words), r.lr - r.exp_lr
    fe = pd.get_dummies(r[["grp"]].assign(cls=r.cls.astype(str)), drop_first=True, dtype=float)
    m = sm.OLS(r.resid, sm.add_constant(pd.concat([r[["z", "log_words"]], fe], axis=1))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc["z"])
    print(f"\n1. RANK (board players with transcripts, classes {r.cls.min()}-{r.cls.max()}, n={len(r)})")
    print(f"   per 1 sd word length: board rank x{np.exp(m.params.z):.2f} of stats-expected (95% {lo:.2f}-{hi:.2f}, "
          f"p {m.pvalues.z:.2f}); < 1 = ranked better than stats justify")
    m2 = sm.OLS(r.lr, sm.add_constant(pd.concat([r[["z", "log_words"]], fe], axis=1))).fit(cov_type="HC1")
    print(f"   (raw, no stats control: board rank x{np.exp(m2.params.z):.2f}, p {m2.pvalues.z:.2f})")

    # 2. value vs stats-only outcome model
    pm.FEATURES["player_pre"] = list(pm.PLAYER)
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("player_pre", "consensus_pre"), test_classes=(2018, 2019, 2020, 2021, 2022))
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R["player_id"] = R.player_id.astype(str)
    R = R.merge(uni[["player_id", "cls", "on_board"]], on=["player_id", "cls"])
    R["actual"] = R[[f"y{s}" for s in pm.SEASONS]].sum(axis=1)
    for v in ("player_pre", "consensus_pre"):
        R[f"exp_{v}"] = R[[f"{v}_p{s}" for s in pm.SEASONS]].sum(axis=1)
    d = R[R.on_board == 1].merge(f, on=["player_id", "cls"])
    d["z"], d["log_words"] = z(d.word_len), np.log(d.n_words)
    print(f"\n2. VALUE, board players 2018-22 with transcripts (n={len(d)})")
    for v, lab in [("exp_player_pre", "vs stats-only model"), ("exp_consensus_pre", "vs consensus")]:
        fe = pd.get_dummies(d.cls, prefix="c", drop_first=True, dtype=float)
        g = sm.GLM(d.actual, sm.add_constant(pd.concat([d[["z", "log_words"]], fe], axis=1)), family=sm.families.Poisson(),
                   offset=np.log(d[v].clip(lower=1e-4))).fit(cov_type="HC1")
        lo, hi = np.exp(g.conf_int().loc["z"])
        print(f"   {lab:22s} per 1 sd word length: value x{np.exp(g.params.z):.2f} (95% {lo:.2f}-{hi:.2f}, p {g.pvalues.z:.2f})")
    print(f"   (stats-only vs consensus model, all board players: rank corr with outcome "
          f"{R[R.on_board == 1].groupby('cls').apply(lambda g: g.exp_player_pre.corr(g.actual, method='spearman'), include_groups=False).mean():.2f} vs "
          f"{R[R.on_board == 1].groupby('cls').apply(lambda g: g.exp_consensus_pre.corr(g.actual, method='spearman'), include_groups=False).mean():.2f})")


if __name__ == "__main__":
    main()
