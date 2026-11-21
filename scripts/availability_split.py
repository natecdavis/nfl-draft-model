"""Injured vs healthy-but-not-playing: does knowing WHY a player missed games help predict his next contract and value?

Missed games classified from nflverse weekly rosters and injury reports (veteran.availability): injured (IR / PUP /
non-football injury, or out / doubtful on the injury report), healthy no-play (on the roster or practice squad, no
injury designation, no snaps), other (suspended, other reserve lists, off the roster). Shares of his team's games.
A  Contract model (fa_market): current features vs + last full season's split (i1, h1, o1) and, planning a year ahead,
   the season in progress so far (pi, ph, po). Leave-one-year-out at lag 1 and lag 2, forward-chained at lag 2.
B  Value forecast (veteran.Comps, skip 1): current shift (age, v1, v2, g1, p_share, p_gp) vs + i1, h1 (anchor's last
   season) and pi, ph (season in progress, weeks 1-11). 4000 random anchors 2014-24; seasons 2-3; and calibration for
   players who missed time injured vs healthy.
Pre-set: adopt A if R2 and median miss improve in all three checks; adopt B if season-2 RMSE falls with a 95% interval
below 0 and calibration stays within 0.95-1.05.
Run: .venv/bin/python scripts/availability_split.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src", "scripts"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402
from partial_season_test import score  # noqa: E402


def part_a():
    rows = []
    D1, D2 = fm.training(1), fm.training(2)
    for lab, D, base, new in [("lag 1", D1, fm.FEATS_BASE, fm.FEATS_AVAIL1), ("lag 2", D2, fm.FEATS_PARTIAL, fm.FEATS_AVAIL2)]:
        for chk in ("leave-year-out", "forward 2022-26") if lab == "lag 2" else ("leave-year-out",):
            for name, feats in (("current", base), ("+ why missed", new)):
                P = []
                years = sorted(D.Y.unique()) if chk == "leave-year-out" else range(2020, 2027)
                for Y in years:
                    tr = D[D.Y != Y] if chk == "leave-year-out" else D[D.Y < Y]
                    te = D[D.Y == Y].copy()
                    te["pred"] = np.exp(fm.fit(tr, feats=feats)["log_apy"].predict(te[feats]))
                    P.append(te)
                P = pd.concat(P)
                if chk.startswith("forward"):
                    P = P[P.Y >= 2022]
                r2, med, w25 = score(P, "pred")
                rows.append({"timing": lab, "check": chk, "features": name, "R2": r2, "median miss": med, "within 25%": w25})
                if name != "current" and lab == "lag 2" and chk == "leave-year-out":
                    P["hurt"] = np.where(P.i1 >= 0.25, "missed 25%+ injured", np.where(P.h1 >= 0.25, "missed 25%+ healthy", "other"))
                    keep = P
    print("A. Contract model")
    print(pd.DataFrame(rows).to_string(index=False))
    print("  actual / predicted by last season's absences (lag 2, + why missed):",
          keep.groupby("hurt").apply(lambda d: f"{d.apy.sum() / (d.pred.sum() * fm.SMEAR[2]):.2f} (n {len(d)})",
                                     include_groups=False).to_dict())


class SplitComps(vt.Comps):
    EXTRA = ["i1", "h1", "pi", "ph"]

    def __init__(self, A):
        super().__init__(A)
        self.beta_s = {}
        f = vt.RAW_FEATS + vt.P_FEATS + self.EXTRA
        for g, G in A.groupby("grp"):
            for k in range(1, vt.H + 1):
                d = G[G[f"y{k}"].notna()]
                X = np.column_stack([np.ones(len(d)), d[f].to_numpy()])
                self.beta_s[(g, k)] = np.linalg.lstsq(X, d[f"y{k}"].to_numpy(), rcond=None)[0][1:]

    def both(self, row, k):
        row, x = self._x(row)
        g = row["grp"]
        nb = self.neighbors(g, x, row["pfr_id"], row["Y"], need=k)
        P = self.A.loc[nb]
        f0 = vt.RAW_FEATS + vt.P_FEATS
        f1 = f0 + self.EXTRA
        b0 = (P[f"y{k}"].to_numpy() + (np.array([row[c] for c in f0], float) - P[f0].to_numpy()) @ self.beta_p[(g, k)]).clip(min=0)
        b1 = (P[f"y{k}"].to_numpy() + (np.array([row[c] for c in f1], float) - P[f1].to_numpy()) @ self.beta_s[(g, k)]).clip(min=0)
        return b0.mean(), b1.mean()


def part_b():
    A = vt.comps().A.copy()  # anchors with p_share, p_gp
    ex = []
    for Y in sorted(A.Y.unique()):
        a1 = vt.availability(int(Y) - 1).rename(columns={"a_inj": "i1", "a_healthy": "h1"})[["i1", "h1"]]
        ap = vt.availability(int(Y), vt.PARTIAL_W).rename(columns={"a_inj": "pi", "a_healthy": "ph"})[["pi", "ph"]]
        ex.append(a1.join(ap, how="outer").assign(Y=Y).reset_index(names="pfr_id"))
    A = A.merge(pd.concat(ex), on=["pfr_id", "Y"], how="left")
    A[["i1", "h1", "pi", "ph"]] = A[["i1", "h1", "pi", "ph"]].fillna(0.0)
    C = SplitComps(A)
    S = A[A.Y.between(2014, 2024)].sample(4000, random_state=5)
    rows = []
    for k in (2, 3):
        d = S[S[f"y{k}"].notna()]
        f = np.array([C.both(r, k) for r in d.to_dict("records")])
        a = d[f"y{k}"].to_numpy()
        e0, e1 = (a - f[:, 0]) ** 2, (a - f[:, 1]) ** 2
        rng = np.random.default_rng(0)
        bs = [(e1 - e0)[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        rows.append({"season": k, "n": len(a), "RMSE current": np.sqrt(e0.mean()) * 100, "RMSE + why missed": np.sqrt(e1.mean()) * 100,
                     "MSE change (x1e4)": f"{(e1 - e0).mean() * 1e4:+.3f} ({np.percentile(bs, 2.5) * 1e4:+.3f} to {np.percentile(bs, 97.5) * 1e4:+.3f})",
                     "actual/forecast current": a.sum() / f[:, 0].sum(), "actual/forecast new": a.sum() / f[:, 1].sum()})
        if k == 2:
            d = d.assign(f0=f[:, 0], f1=f[:, 1], grp_=np.where((d.i1 >= 0.25) | (d.pi >= 0.25), "missed 25%+ injured",
                                                               np.where((d.h1 >= 0.25) | (d.ph >= 0.25), "missed 25%+ healthy", "other")))
            print("  season 2, actual / forecast (current -> new):",
                  {g: f"{x.y2.sum() / x.f0.sum():.2f} -> {x.y2.sum() / x.f1.sum():.2f} (n {len(x)})" for g, x in d.groupby("grp_")})
    print("\nB. Value forecast (% of cap)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    part_a()
    part_b()


if __name__ == "__main__":
    main()
