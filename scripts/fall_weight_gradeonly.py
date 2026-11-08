"""Fall weight on the stored vs grade-only labels (follow-up to scripts/grade_only_check.py).

The app and review shade a falling player's log board rank toward the log of the pick where he went:
log_rank += w * (log_pick - log_rank), w = 0.85 (prospect_model.FALL_WEIGHT; first estimated in
scripts/fall_weight.py / fall_weight_all.py on star and starter tiers). Here w is re-estimated on each label's
4-year value, the currency the engine uses.

Pre-specified (before running): board players 2014-22 who fell (log(pick+10) > log(rank+10)), undrafted counted as
pick 300; Poisson value(y1-4) ~ a*log(pick+10) + b*log(rank+10) + position group + class, w = a / (a + b).
Paired bootstrap over players (same resample for every label). Secondary: drafted fallers only; rounds 1-3 picks.
Prediction: w lower on the grade-only label (part of slot's edge was teams playing their own picks), ~0.75-0.8.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
import prospect_model as pm

LABELS = {"stored": pm.LABEL_FILE, "grade-only t20": pm.OUT / "labels_gradeonly_t20.parquet",
          "grade-only t35": pm.OUT / "labels_gradeonly_t35.parquet"}
N_BOOT = 500


def fallers(path):
    pm.LABEL_FILE = path
    u = pm.load()
    u = u[(u.on_board == 1) & u.cls.between(2014, 2022) & u.hasan_rank.notna()].copy()
    u["lpick"] = np.log(u.pick + 10)
    u["lrank"] = np.log(u.hasan_rank + 10)
    u["v"] = u[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    return u[u.lpick > u.lrank].set_index(["player_id", "cls"]).sort_index()


def w_of(d):
    X = pd.concat([d[["lpick", "lrank"]],
                   pd.get_dummies(d.grp, prefix="g", drop_first=True, dtype=float),
                   pd.get_dummies(d.index.get_level_values("cls"), prefix="c", drop_first=True, dtype=float).set_index(d.index)],
                  axis=1)
    m = sm.GLM(d.v, sm.add_constant(X), family=sm.families.Poisson()).fit()
    a, b = -m.params.lpick, -m.params.lrank
    return a / (a + b), a, b


def main():
    F = {nm: fallers(p) for nm, p in LABELS.items()}
    pm.LABEL_FILE = LABELS["stored"]
    keys = F["stored"].index
    for d in F.values():
        keys = keys.intersection(d.index)
    F = {nm: d.loc[keys] for nm, d in F.items()}
    subsets = {"all fallers (undrafted = pick 300)": lambda d: d,
               "drafted fallers": lambda d: d[d.drafted.astype(bool)],
               "fallers picked in rounds 1-3 (pick <= 100)": lambda d: d[d.drafted.astype(bool) & (d.pick <= 100)]}
    rng = np.random.default_rng(0)
    for sname, f in subsets.items():
        S = {nm: f(d) for nm, d in F.items()}
        n = len(S["stored"])
        print(f"\n{sname}: n={n}")
        est = {nm: w_of(d) for nm, d in S.items()}
        boots = {nm: [] for nm in S}
        for _ in range(N_BOOT):
            ix = rng.integers(0, n, n)
            for nm, d in S.items():
                boots[nm].append(w_of(d.iloc[ix])[0])
        for nm, (w, a, b) in est.items():
            lo, hi = np.percentile(boots[nm], [2.5, 97.5])
            line = f"  {nm:16s} w = {w:.2f} (95% {lo:.2f}-{hi:.2f})  [a {a:.2f}, b {b:.2f}]"
            if nm != "stored":
                dd = np.array(boots[nm]) - np.array(boots["stored"])
                line += f"  minus stored {w - est['stored'][0]:+.2f} (95% {np.percentile(dd, 2.5):+.2f} to {np.percentile(dd, 97.5):+.2f})"
            print(line)


if __name__ == "__main__":
    main()
