"""Captain choice test: would a 'college captain' bump (Brugler profile mentions captain) have improved actual
draft choices? (2019-2022 drafts; Nov 22 2026, after captain_own_value x1.21 vs slot in team_leadership_test)

Copy of beast_choice.py. Baseline ('consensus' columns) = consensus prediction x an in-guide multiplier; treatment
('brugler' columns) = consensus x (in guide + captain) multiplier, both fit leave-one-draft-out, so the difference is
the captain bump alone. Specified
before running: primary = picks 101+ (where the in-sample signal was: x1.53), all picks and R5-7 reported too.

Same setup as the earlier choice test (prospect_model.choice_test): at every pick, each
version takes the available player its predictions like best (fallers shaded toward the
pick), and we record what that player actually delivered. Versions:
  consensus      - consensus rank + position (the current engine's model)
  + Brugler      - the same prediction times a text multiplier: in the guide or not, his grade,
                   and the elite / upside / ready / injury word rates
The multiplier is a Poisson model of actual / expected value, fit leave-one-class-out (each
draft's multiplier learned from the other three drafts; the 'forward' check uses only
earlier drafts, for 2021-22). Scored in 4-season value and in Super Bowl odds for a middle
team (QBs assumed to play right away, like the pure-value test).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import prospect_model as pm
from beast_test import word_feats
from outcomes_v2 import CF_OVER_OURS, REG
from outlook import OUT
from pick_value import baseline_paths, pick_effect
from trade import Context

CLASSES = (2019, 2020, 2021, 2022)
TEXT = ["in_guide", "captain"]
FAM2GRP = {"DB": "CB", "OL": "T", "DL": "IDL", "EDGE": "ED"}


def text_table(uni):
    from team_leadership_test import captains
    c = captains()[["player_id", "captain"]].assign(player_id=lambda d: d.player_id.astype(str))
    u = uni[uni.cls.isin(CLASSES)][["player_id", "cls"]].assign(player_id=lambda d: d.player_id.astype(str))
    u = u.merge(c.drop_duplicates("player_id"), on="player_id", how="left")
    u["in_guide"] = u.captain.notna().astype(float)
    u["captain"] = u.captain.fillna(False).astype(float)
    u = u.set_index("player_id")
    u.index = u.index.map(lambda x: x)
    return u


def fit_mult(rows, train_classes, feats=None):
    global TEXT
    keep = TEXT
    if feats is not None:
        TEXT = feats
    try:
        return _fit_mult(rows, train_classes)
    finally:
        TEXT = keep


def _fit_mult(rows, train_classes):
    from sklearn.linear_model import PoissonRegressor
    tr = rows[rows.cls.isin(train_classes) & (rows.expected > 1e-4)]
    X = tr[TEXT].to_numpy(); mu, sd = X.mean(0), X.std(0) + 1e-9
    m = PoissonRegressor(alpha=1e-3, max_iter=1000).fit((X - mu) / sd, tr.actual / tr.expected, sample_weight=tr.expected)
    return lambda Z: np.clip(m.predict((Z - mu) / sd), 0.5, 2.0), dict(zip(TEXT, m.coef_))


def main():
    uni = pm.load()
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("consensus_pre",), test_classes=CLASSES)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    tx = text_table(uni)
    R["player_id"] = R.player_id.astype(str)
    R = R.join(tx[TEXT], on="player_id")
    R[TEXT] = R[TEXT].fillna(0.0)
    R["actual"] = R[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    R["expected"] = R[[f"consensus_pre_p{t}" for t in pm.SEASONS]].sum(axis=1)
    mults = {}
    for c in CLASSES:
        f_loo, coef = fit_mult(R, [x for x in CLASSES if x != c])
        f_fwd = fit_mult(R, [x for x in CLASSES if x < c])[0] if c >= 2021 else None
        g_loo = fit_mult(R, [x for x in CLASSES if x != c], ["in_guide"])[0]
        g_fwd = fit_mult(R, [x for x in CLASSES if x < c], ["in_guide"])[0] if c >= 2021 else None
        mults[c] = (f_loo, f_fwd, g_loo, g_fwd)
        print(f"{c}: multiplier weights (from other drafts, standardized): " + ", ".join(f"{k} {v:+.2f}" for k, v in coef.items()))

    ctx = Context(); sit = {"start": 0.0, "shift": {}, "qb_from": 1}
    base = baseline_paths(sit, np.random.default_rng(0), 2000); b0 = ctx.sb(base).sum(axis=1).mean()
    cache = {}

    def titles(row, k):
        key = (row.player_id, k)
        if key not in cache:
            fam = row.grp
            if fam not in CF_OVER_OURS:
                cache[key] = 0.0
            else:
                y = np.array([max(row[f"y{t}"], 0) for t in pm.SEASONS]) / CF_OVER_OURS[fam]
                y5 = np.r_[y, y[-1]]; price = max(y5.max(), 1e-6)
                pts = pick_effect((y5 / price)[None, :], np.array([price]), int(k), FAM2GRP.get(fam, fam), sit,
                                  ctx.cost, ctx.bar, ctx.rates)
                cache[key] = (ctx.sb(base + pts).sum(axis=1).mean() - b0) * 100
        return cache[key]

    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    rows = []
    try:
        for c in CLASSES:
            te = uni[uni.cls == c].copy(); tr = uni[uni.cls < c]
            te["y14"] = te[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
            Z = tx.reindex(te.player_id.astype(str))[TEXT].fillna(0.0).to_numpy()
            f_loo, f_fwd, g_loo, g_fwd = mults[c]
            m_loo = f_loo(Z); m_fwd = f_fwd(Z) if f_fwd else None
            m_g = g_loo(Z[:, :1]); m_gf = g_fwd(Z[:, :1]) if g_fwd else None
            models = [pm.fit(tr[tr.cls + t - 1 <= c - 1], "consensus_pre", t) for t in pm.SEASONS]
            versions = {"consensus": m_g, "brugler": m_loo}
            if m_fwd is not None:
                versions["brugler_fwd"] = m_fwd; versions["consensus_fwd"] = m_gf
            taken = {v: set() for v in versions}
            for k in range(1, int(te[te.drafted].pick.max()) + 1):
                avail = (te.pick >= k).to_numpy()
                actual = te[te.pick == k]
                r = {"cls": c, "pick": k}
                if len(actual):
                    r["team_value"] = actual.y14.iloc[0]; r["team_titles"] = titles(actual.iloc[0], k)
                cand = te.copy(); lk = np.log(k + 10)
                fell = cand.on_board.astype(bool) & (cand.log_rank < lk)
                cand.loc[fell, "log_rank"] += pm.FALL_WEIGHT * (lk - cand.loc[fell, "log_rank"])
                pred = sum(m.predict(cand[f]) for m, f in models)
                for v, mult in versions.items():
                    score = np.where(avail & ~te.player_id.isin(taken[v]).to_numpy(), pred * mult, -np.inf)
                    i = int(np.argmax(score)); row = te.iloc[i]
                    taken[v].add(row.player_id)
                    r[f"{v}_value"] = row.y14; r[f"{v}_titles"] = titles(row, k); r[f"{v}_player"] = row.player_name
                rows.append(r)
            print(f"{c} done", flush=True)
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    ch = pd.DataFrame(rows)
    ch.to_parquet(OUT / "captain_choice.parquet")
    report(ch)


def report(ch):
    rng = np.random.default_rng(0)
    ch = ch.assign(rnd=pd.cut(ch.pick, [0, 32, 140, 262], labels=["R1", "R2-4", "R5-7"]))
    late = ch[ch.pick > 100]
    same = (ch.consensus_player == ch.brugler_player).mean()
    print(f"\nPicks where the captain bump changed the choice: {1 - same:.0%} of {len(ch)}")
    for unit, scale in [("value", 100), ("titles", 1)]:
        print(f"\nAverage per pick, actual {'4-season value (% of cap)' if unit == 'value' else 'Super Bowl odds for a middle team (% of a title)'}")
        cols = [f"team_{unit}", f"consensus_{unit}", f"brugler_{unit}"]
        t = ch.groupby("rnd", observed=True)[cols].mean() * scale
        t.loc["All"] = ch[cols].mean() * scale
        print(t.round(3).to_string())
        for name, g in [("PRIMARY picks 101+", late), ("All", ch)] + list(ch.groupby("rnd", observed=True)):
            d = ((g[f"brugler_{unit}"] - g[f"consensus_{unit}"]) * scale).to_numpy()
            b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f"   + Brugler minus consensus, {name}: {d.mean():+.3f} (95% {np.percentile(b, 2.5):+.3f} to {np.percentile(b, 97.5):+.3f})")
        f = ch[(ch.cls >= 2021) & (ch.pick > 100)]
        d = ((f[f"brugler_fwd_{unit}"] - f[f"consensus_fwd_{unit}"]) * scale).to_numpy()
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"   forward-only check (2021-22 picks 101+, multiplier from earlier drafts only): {d.mean():+.3f} "
              f"(95% {np.percentile(b, 2.5):+.3f} to {np.percentile(b, 97.5):+.3f})")


if __name__ == "__main__":
    main()
