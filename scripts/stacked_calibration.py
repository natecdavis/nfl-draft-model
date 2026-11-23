"""Stacked signals, direct out-of-sample check (more power than the choice test): does the combined multiplier
predict actual / expected (consensus_pre, 4 seasons) in a draft it wasn't fit on? Leave-one-draft-out Poisson
deviance vs no multiplier (and vs in-guide only for stack B), unshrunk (alpha 1e-3) and with the inner-CV alpha;
plus the calibration slope (actual on expected x mult^b: b = 1 means the multiplier is right, 0 means no signal).
Board players (on_board) and picks 101+ also shown, since that is where the leads came from."""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from stacked_choice import A_FEATS, B_FEATS, build, fit_mult


def dev(y, p):
    y = np.clip(y, 0, None)
    return float(np.sum(2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / p), 0) - (y - p))))


def main():
    uni, classes, R, *_ = build()
    R = R[R.expected > 1e-4].copy()
    R["on_board"] = R.player_id.map(uni.assign(player_id=uni.player_id.astype(str)).set_index("player_id").on_board)
    sets = [("A", A_FEATS, list(classes), None), ("B", A_FEATS + B_FEATS, [c for c in classes if c >= 2019], ["in_guide"])]
    for nm, feats, cl, basef in sets:
        for alpha in [1e-3, None]:
            out = []
            for c in cl:
                oth = [x for x in cl if x != c]
                f, _, a = fit_mult(R, feats, oth, alpha=alpha)
                te = R[R.cls == c].copy()
                te["m"] = f(te[feats].to_numpy(float))
                te["m0"] = fit_mult(R, basef, oth, alpha=1e-3)[0](te[basef].to_numpy(float)) if basef else 1.0
                te["a"] = a
                out.append(te)
            T = pd.concat(out)
            for sub, q in [("all", T), ("board", T[T.on_board == 1]), ("drafted 101+", T[T.drafted.astype(bool) & (T.pick > 100)]), ("undrafted", T[~T.drafted.astype(bool)])]:
                d1, d0 = dev(q.actual, q.expected * q.m), dev(q.actual, q.expected * q.m0)
                g = sm.GLM(q.actual.clip(lower=0), np.log(q.m / q.m0).to_numpy()[:, None], family=sm.families.Poisson(),
                           offset=np.log(q.expected * q.m0)).fit(cov_type="HC1")
                lo, hi = np.asarray(g.conf_int())[0]
                lab = f"stack {nm}, alpha {'inner-CV' if alpha is None else alpha}"
                print(f"{lab:26s} {sub:10s} n={len(q):5d} deviance change {d1 - d0:+8.3f} | slope {np.asarray(g.params)[0]:+.2f} "
                      f"({lo:+.2f} to {hi:+.2f}) | mult sd {np.log(q.m / q.m0).std():.3f}")


if __name__ == "__main__":
    main()
