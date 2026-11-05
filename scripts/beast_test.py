"""Does Brugler's scouting text add anything beyond consensus rank + position?

Baseline: prospect_model 'consensus_pre' (rank + position), each class predicted by a model
trained only on earlier classes (pm.rolling). Players: Beast profiles 2019-2022 matched to
the cfdarko universe (four NFL seasons played). Three questions, all out of sample:
  1. Level   - does the text predict actual 4-season value minus expected?
  2. Spread  - at the same expected value, does boom/bust language predict wider outcomes?
  3. Timing  - does 'ready now' vs 'project' language predict the share of value in year 1?
Text features: (a) Brugler's round grade, (b) hand-picked word groups, (c) a bag-of-words
ridge model fit leave-one-class-out.
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import prospect_model as pm
from outlook import OUT
from public_board import _norm

WORDS = {
    "upside": r"upside|ceiling|tools|traits|potential|untapped|moldable|raw\b|develop|project\b|work in progress|unrefined|lump of clay",
    "ready": r"polished|pro[- ]ready|plug[- ]and[- ]play|day[- ]one|immediate|nfl[- ]ready|refined|technician|reliable|consistent\b|high floor|floor",
    "injury": r"injur|surgery|torn|\bacl\b|meniscus|labrum|concussion|fracture|sprain|missed (?:\w+ )?games|medical|durability",
    "character": r"arrest|suspend|dismiss|maturity|character|off[- ]field|discipline|violat",
    "weak_athlete": r"lacks? (?:elite |ideal )?(?:speed|burst|explosiveness|length|size)|limited athlet|average athlet|marginal",
    "elite": r"elite|special|rare\b|outstanding|explosive|dominant|freaky|exceptional",
}


def match(beast: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    uni = uni.assign(key=uni.player_name.map(_norm), last=uni.player_name.map(lambda n: _norm(n).split()[-1] if _norm(n) else ""))
    beast = beast.rename(columns={"cls": "elig"})
    b = beast.assign(key=beast.name.map(_norm), last=beast.name.map(lambda n: _norm(n).split()[-1] if _norm(n) else ""))
    exact = b.merge(uni, left_on=["year", "key"], right_on=["cls", "key"], how="left", suffixes=("", "_u"))
    exact = exact.drop_duplicates(["year", "name", "school"])
    miss = exact.player_id.isna()
    # fallback: same class + last name + position family, unique
    fam = {"T": "OL", "G": "OL", "C": "OL"}
    for i in exact.index[miss]:
        r = exact.loc[i]
        c = uni[(uni.cls == r.year) & (uni["last"] == r["last"])]
        if len(c) > 1:
            c = c[c.grp.map(lambda g: fam.get(g, g)) == fam.get(r.section, r.section)]
        if len(c) == 1:
            for col in uni.columns:
                if col in exact.columns and col not in ("key", "last", "year"):
                    exact.at[i, col] = c.iloc[0][col]
    return exact


def word_feats(d):
    txt = (d.strengths.fillna("") + " " + d.weaknesses.fillna("") + " " + d.summary.fillna("")).str.lower()
    n = txt.str.split().str.len().clip(lower=50)
    f = pd.DataFrame({k: txt.str.count(v) / n * 1000 for k, v in WORDS.items()}, index=d.index)
    f["weak_share"] = d.weaknesses.fillna("").str.len() / (d.strengths.fillna("").str.len() + d.weaknesses.fillna("").str.len()).clip(lower=1)
    f["full_profile"] = d.strengths.notna().astype(int)
    return f, txt


def ols(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def loyo(y, X, groups, fit):
    """Leave-one-class-out predictions."""
    p = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        p[te] = fit(X[tr], y[tr], X[te])
    return p


def ridge_fit(alpha):
    def f(Xtr, ytr, Xte):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte)
    return f


def r2_gain(y, base, pred):
    """Out-of-sample R² of the residual explained by pred (0 = nothing)."""
    res = y - base
    return 1 - np.mean((res - pred) ** 2) / np.mean((res - res.mean()) ** 2)


def boot_ci(y, base, pred, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        out.append(r2_gain(y[s], base[s], pred[s]))
    return np.percentile(out, [2.5, 97.5])


def main():
    beast = pd.read_parquet(OUT / "beast_profiles.parquet")
    uni = pm.load()
    R = pm.rolling(uni, versions=("consensus_pre",), test_classes=(2019, 2020, 2021, 2022))
    R = R.merge(uni[["player_id", "cls", "log_rank", "on_board"]], on=["player_id", "cls"], how="left")
    d = match(beast[beast.year <= 2022], R)
    print(f"profiles 2019-22: {len(beast[beast.year <= 2022])}, matched: {d.player_id.notna().sum()}")
    d = d[d.player_id.notna()].reset_index(drop=True)
    d["actual"] = d[[f"y{t}" for t in pm.SEASONS]].sum(axis=1).astype(float)
    d["expected"] = d[[f"consensus_pre_p{t}" for t in pm.SEASONS]].sum(axis=1).astype(float)
    f, txt = word_feats(d)
    d["grade"] = d.grade_round.fillna(8.0)
    print(f"on consensus board: {int(d.on_board.sum())}, drafted: {int(d.drafted.sum())}")
    groups = d.year.to_numpy()
    y, base = d.actual.to_numpy(), d.expected.to_numpy()

    # ---- 1. level: residual = actual - expected (both in fraction of cap over 4 seasons)
    print("\n1. LEVEL: does it predict value beyond consensus rank + position? (out-of-sample R² of the miss)")
    lr = np.log(d.expected.clip(lower=1e-3)).to_numpy()
    tests = {
        "Brugler grade (round)": np.column_stack([d.grade, lr]),
        "word groups": np.column_stack([f.to_numpy(), lr]),
        "grade + word groups": np.column_stack([d.grade, f.to_numpy(), lr]),
    }
    for name, X in tests.items():
        p = loyo(y - base, X, groups, ridge_fit(1.0))
        lo, hi = boot_ci(y, base, p, groups)
        print(f"   {name:28s} R² {r2_gain(y, base, p):+.3f}  (95%: {lo:+.3f} to {hi:+.3f})")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix
    best = None
    for a in (3.0, 10.0, 30.0):
        def tf_fit(tr_idx_X, ytr, te_X, a=a):
            return None
        p = np.full(len(y), np.nan)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            vec = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True, max_features=20000)
            Xtr = hstack([vec.fit_transform(txt[tr]), csr_matrix(lr[tr, None])]).tocsr()
            Xte = hstack([vec.transform(txt[te]), csr_matrix(lr[te, None])]).tocsr()
            from sklearn.linear_model import Ridge
            p[te] = Ridge(alpha=a).fit(Xtr, (y - base)[tr]).predict(Xte)
        g2 = r2_gain(y, base, p)
        lo, hi = boot_ci(y, base, p, groups)
        print(f"   all text (bag of words, a={a:<4}) R² {g2:+.3f}  (95%: {lo:+.3f} to {hi:+.3f})")
        if best is None or g2 > best[0]:
            best = (g2, p)
    d["text_pred"] = best[1]

    # Brugler vs consensus head to head: rank correlation with actual among board players
    on = d[d.on_board == 1]
    from scipy.stats import spearmanr
    print(f"   rank correlation with actual value (board players, n={len(on)}): consensus "
          f"{spearmanr(-on.log_rank, on.actual)[0]:.3f}, Brugler grade {spearmanr(-on.grade, on.actual)[0]:.3f}, "
          f"both (consensus model + grade) {spearmanr(on.expected - 0.0 * on.grade, on.actual)[0]:.3f}")

    # ---- 2. spread: at the same expected value, do boom/bust words predict bigger misses?
    print("\n2. SPREAD: at the same expected value, does the text predict how far outcomes land from it?")
    d["bin"] = pd.qcut(d.expected.rank(method="first"), 8, labels=False)
    d["dev"] = np.log1p(d.actual / d.expected.clip(lower=1e-3))  # outcome relative to expectation
    d["absdev"] = (d.dev - d.groupby("bin").dev.transform("median")).abs()
    for k in ["upside", "ready", "injury", "elite", "weak_share"]:
        hi_ = f[k] > f.groupby(d["bin"])[k].transform("median")
        a = d[hi_].groupby("bin").absdev.mean(); b_ = d[~hi_].groupby("bin").absdev.mean()
        sd_hi = d[hi_].groupby("bin").dev.std().mean(); sd_lo = d[~hi_].groupby("bin").dev.std().mean()
        print(f"   more '{k}' words than others at same expectation: spread {sd_hi:.3f} vs {sd_lo:.3f}"
              f"  (ratio {sd_hi / sd_lo:.2f})")
    # chance of a big hit (top 10% outcome relative to expectation)
    top = d.dev > d.groupby("bin").dev.transform(lambda s: s.quantile(0.9))
    bust = d.actual <= d.groupby("bin").actual.transform(lambda s: s.quantile(0.25))
    for k in ["upside", "ready", "elite", "injury"]:
        hi_ = f[k] > f.groupby(d["bin"])[k].transform("median")
        print(f"   '{k}' high vs low: big-hit rate {top[hi_].mean():.1%} vs {top[~hi_].mean():.1%}; "
              f"bottom-quarter rate {bust[hi_].mean():.1%} vs {bust[~hi_].mean():.1%}")

    # ---- 3. timing: share of 4-season value in year 1, for players who produced
    print("\n3. TIMING: does 'ready now' vs 'project' language predict how much value comes in year 1?")
    t = d[d.actual > 0.01].copy()
    t["share1"] = t.y1 / t.actual
    t["exp_share1"] = t.consensus_pre_p1 / t.expected
    ft = f.loc[t.index]
    yy = (t.share1 - t.exp_share1).to_numpy()
    for name, X in {"word groups": ft[["upside", "ready", "elite", "weak_athlete"]].to_numpy(),
                    "Brugler grade": t[["grade"]].to_numpy()}.items():
        p = loyo(yy, X, t.year.to_numpy(), ridge_fit(1.0))
        r = np.corrcoef(p, yy)[0, 1]
        print(f"   {name:15s} out-of-sample correlation with the year-1 surprise: {r:+.3f} (n={len(t)})")
    for k in ["ready", "upside"]:
        print(f"   corr('{k}' words, year-1 share beyond expectation): {np.corrcoef(ft[k], yy)[0, 1]:+.3f}")
    d.to_parquet(OUT / "beast_test.parquet")


if __name__ == "__main__":
    main()
