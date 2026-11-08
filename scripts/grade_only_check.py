"""Does draft slot's edge over the consensus board survive a label without team-chosen playing time?

Labels (scripts/grade_only_label.py): 'stored' = cfdarko's v4 label (snap share x PFF quality); 'grade-only t20'
(rookie-deal shares >= 0.2 replaced by the share veterans of the same quality play); 't35' as a sensitivity check.
Expectations are refit under each label (prospect_model.rolling, REG parameters, out of sample), so each label
is judged by models trained on that label.

Pre-specified (before running):
  PRIMARY  team information beyond the board: board players drafted 2018-22, Poisson actual ~ reach + class FE,
           offset log(consensus expectation), reach = log((board rank + 10) / (pick + 10)) (> 0 = taken earlier
           than the board). b > 0 = players teams liked more than the board turned out better than the board
           expected. If part of that is teams playing their own high picks, b shrinks on the grade-only label.
           Paired bootstrap (players resampled) for b_grade - b_stored. Prediction: shrinks, not to zero.
  SECONDARY (a) Spearman within class of slot vs consensus expectations against actual, same players;
            (b) mean 4-year value by pick band (does the top-pick premium shrink?);
            (c) fall shading: same b, split into rises (reach > 0) and falls (reach < 0).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, expectations

LABELS = {"stored": pm.CF / "data/processed/rookie_value_labels.parquet", "grade-only t20": pm.OUT / "labels_gradeonly_t20.parquet",
          "grade-only t35": pm.OUT / "labels_gradeonly_t35.parquet"}
BANDS = [0, 10, 32, 64, 100, 160, 262]


def run(path):
    pm.LABEL_FILE = path
    uni = pm.load()
    cache = pm.OUT / f"grade_only_check_{Path(path).stem}.parquet"
    if cache.exists() and cache.stat().st_mtime > Path(path).stat().st_mtime:
        return uni, pd.read_parquet(cache)
    R = expectations(uni, SLOT_CLASSES)
    R = R.merge(uni[["player_id", "cls", "hasan_rank", "pick", "drafted"]], on=["player_id", "cls"], how="left",
                suffixes=("", "_u"))
    R.to_parquet(cache)
    return uni, R


def design(d):
    d = d.copy()
    d["reach"] = np.log((d.hasan_rank + 10) / (d.pick + 10))
    d["rise"] = d.reach.clip(lower=0)
    d["fall"] = d.reach.clip(upper=0)
    fe = pd.get_dummies(d.cls, prefix="c", drop_first=True, dtype=float)
    return d, fe


def fit_b(d, fe, cols=("reach",)):
    m = sm.GLM(d.actual, sm.add_constant(pd.concat([d[list(cols)], fe], axis=1)), family=sm.families.Poisson(),
               offset=np.log(d.exp_cons.clip(lower=1e-4))).fit()
    return m


def main():
    res, unis = {}, {}
    for nm, path in LABELS.items():
        uni, R = run(path)
        unis[nm] = uni
        res[nm] = R[(R.on_board == 1) & R.drafted.astype(bool) & R.exp_cons.notna() & R.hasan_rank.notna()]
    pm.LABEL_FILE = LABELS["stored"]  # these tests were run with the stored label as the default

    print("PRIMARY: team information beyond the board (board players drafted 2018-22)")
    keys = None
    for nm, d in res.items():
        k = set(zip(d.player_id, d.cls))
        keys = k if keys is None else keys & k
    D = {nm: design(d[[(p, c) in keys for p, c in zip(d.player_id, d.cls)]].sort_values(["cls", "player_id"])
                    .reset_index(drop=True)) for nm, d in res.items()}
    B = {}
    for nm, (d, fe) in D.items():
        mh = sm.GLM(d.actual, sm.add_constant(pd.concat([d[["reach"]], fe], axis=1)), family=sm.families.Poisson(),
                    offset=np.log(d.exp_cons.clip(lower=1e-4))).fit(cov_type="HC1")
        B[nm] = mh.params.reach
        print(f"  {nm:16s} n={len(d)}  b(reach) {mh.params.reach:+.3f} (se {mh.bse.reach:.3f}): taken 2x earlier than "
              f"board -> value x{2 ** mh.params.reach:.2f} vs consensus expectation")
    rng = np.random.default_rng(0)
    n = len(D["stored"][0])
    for nm in ["grade-only t20", "grade-only t35"]:
        diffs = []
        for _ in range(500):
            ix = rng.integers(0, n, n)
            bb = {}
            for k in ("stored", nm):
                d, fe = D[k]
                bb[k] = fit_b(d.iloc[ix].reset_index(drop=True), fe.iloc[ix].reset_index(drop=True)).params["reach"]
            diffs.append(bb[nm] - bb["stored"])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {nm} minus stored: {B[nm] - B['stored']:+.3f} (95% {lo:+.3f} to {hi:+.3f}); "
              f"share of stored b remaining {B[nm] / B['stored']:.0%}")

    print("\n(c) rises vs falls")
    for nm, (d, fe) in D.items():
        m = sm.GLM(d.actual, sm.add_constant(pd.concat([d[["rise", "fall"]], fe], axis=1)), family=sm.families.Poisson(),
                   offset=np.log(d.exp_cons.clip(lower=1e-4))).fit(cov_type="HC1")
        print(f"  {nm:16s} rise b {m.params.rise:+.3f} (se {m.bse.rise:.3f})   fall b {m.params.fall:+.3f} (se {m.bse.fall:.3f})")

    print("\n(a) Spearman with actual 4-year value, mean within class (same players)")
    for nm, (d, fe) in D.items():
        rs = d.groupby("cls").apply(lambda g: pd.Series({"slot": g.exp_slot.corr(g.actual, method="spearman"),
                                                         "cons": g.exp_cons.corr(g.actual, method="spearman")}),
                                    include_groups=False).mean()
        print(f"  {nm:16s} slot {rs.slot:.3f}  consensus {rs.cons:.3f}  slot edge {rs.slot - rs.cons:+.3f}")

    print("\n(b) mean 4-year value (% of cap) by pick, drafted 2012-22")
    rows = {}
    for nm, uni in unis.items():
        u = uni[uni.drafted.astype(bool) & uni.cls.between(2012, 2022)].copy()
        u["v"] = u[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
        rows[nm] = u.groupby(pd.cut(u.pick, BANDS), observed=True).v.mean()
    T = pd.DataFrame(rows)
    T.loc["picks 1-10 / 33-64"] = T.iloc[0] / T.iloc[2]
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
