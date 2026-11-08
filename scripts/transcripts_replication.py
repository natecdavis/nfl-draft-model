"""Fresh replication of the press-conference word-length finding on classes 2023-25 (fixed before outcomes were seen).

Discovery (scripts/transcripts_test.py, classes 2016-22): mean word length x0.85 per sd vs slot, x0.83 vs consensus.
Specified in advance (Nov 2026), single feature, nothing else tested:
  feature    word_len from transcripts_test.features, players with >= 200 words, standardized with the DISCOVERY
             sample's mean and sd (players in classes 2016-22) so 'per 1 sd' means the same thing
  outcome    value over the seasons played so far (2023: 3, 2024: 2, 2025: 1), against out-of-sample expectations
             for the same seasons (prospect_model.rolling, REG parameters, training only on seasons before each draft)
  primary    drafted 2023-25 vs draft slot; Poisson, offset log(expected), log words + class fixed effects, HC1
  secondary  board players 2023-25 vs consensus
  prediction value < 1 (one-sided p reported); early seasons are noisy, so power is lower than discovery
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from outcomes_v2 import REG
from outlook import OUT
from transcripts_test import features

CLASSES = (2023, 2024, 2025)


def main():
    a = pd.read_parquet(OUT / "asap_answers.parquet")
    a["player_id"] = a.player_id.astype(str)
    f = features(a)
    f = f[f.n_words >= 200]
    disc = f[f.cls.between(2016, 2022)].word_len
    f["z"] = (f.word_len - disc.mean()) / disc.std()
    f["log_words"] = np.log(f.n_words)

    uni = pm.load()
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("slot", "consensus_pre"), test_classes=CLASSES)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R["player_id"] = R.player_id.astype(str)
    R = R.merge(uni.assign(player_id=uni.player_id.astype(str))[["player_id", "cls", "on_board"]], on=["player_id", "cls"])
    n_seen = {c: pm.LAST_SEASON_PLAYED - c + 1 for c in CLASSES}
    for col, v in [("actual", "y"), ("exp_slot", "slot_p"), ("exp_cons", "consensus_pre_p")]:
        R[col] = [sum(r[f"{v}{t}"] for t in pm.SEASONS[:n_seen[r.cls]]) for _, r in R.iterrows()]
    d = R.merge(f, on=["player_id", "cls"])
    print(f"transcript players (>= 200 words) in classes 2023-25: {len(d)}; by class {d.groupby('cls').size().to_dict()}; "
          f"mean z {d.z.mean():+.2f}")
    for e, s, nm in [("exp_slot", d[d.drafted.astype(bool)], "PRIMARY: drafted vs slot"),
                     ("exp_cons", d[d.on_board == 1], "secondary: board vs consensus")]:
        fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
        m = sm.GLM(s.actual, sm.add_constant(pd.concat([s[["z", "log_words"]], fe], axis=1)), family=sm.families.Poisson(),
                   offset=np.log(s[e].clip(lower=1e-4))).fit(cov_type="HC1")
        lo, hi = np.exp(m.conf_int().loc["z"])
        one = stats.norm.cdf(m.params.z / m.bse.z)
        print(f"  {nm:32s} n={len(s):3d}  per 1 sd: value x{np.exp(m.params.z):.2f} (95% {lo:.2f}-{hi:.2f}; "
              f"one-sided p {one:.3f})")
        hi_lo = s.assign(top=s.z > s.z.median()).groupby("top").apply(lambda g: g.actual.sum() / g[e].sum(), include_groups=False)
        print(f"  {'':32s} actual / expected: longer-word half {hi_lo[True]:.2f}, shorter-word half {hi_lo[False]:.2f}")
        for c in CLASSES:
            t = s[s.cls == c]
            if len(t) >= 15:
                mc = sm.GLM(t.actual, sm.add_constant(t[["z", "log_words"]]), family=sm.families.Poisson(),
                            offset=np.log(t[e].clip(lower=1e-4))).fit(cov_type="HC1")
                print(f"  {'':32s}   class {c}: n={len(t):3d} x{np.exp(mc.params.z):.2f} (p {mc.pvalues.z:.2f})")


if __name__ == "__main__":
    main()
