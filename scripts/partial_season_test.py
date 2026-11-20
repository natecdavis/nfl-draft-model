"""Does the season in progress (weeks 1-11) improve 2027 planning? (data through 2025 + 2026 so far)

Setting replicated in every past year: planning signings / forecasts a year ahead, with full data through season
Y-2 and weeks 1-11 of season Y-1 (snap share per team game, share of team games played, counting stats per 17 team
games; fa_market.partial). No grades for the partial season (PFF 2026 isn't in the data).

A  Contract model (fa_market): lag-2 base vs lag-2 + partial season (FEATS_PARTIAL); lag 1 (full season Y-1, what next
   March will have) as the ceiling. Leave-one-signing-year-out and forward-chained (fit on earlier years, predict
   2020-26, scored 2022-26).
B  Value forecast (veteran.Comps, skip = 1: anchor season Y in progress, forecast seasons Y+1, Y+2 = the comps'
   seasons 2-3): base shift features (age, v1, v2, g1) vs + weeks 1-11 of season Y (p_share, p_gp) in the per-group
   linear shift. 4000 random anchors 2014-2024, comps exclude the same player and season. RMSE and actual / forecast;
   paired bootstrap by player for the RMSE change.
Pre-set rules: A adopt if R2 and median miss improve in both the leave-year-out and the forward check; B adopt if the
season-2 RMSE falls with a 95% interval below 0 and calibration stays within 0.95-1.05.
Run: .venv/bin/python scripts/partial_season_test.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402


def score(P, col):
    e = np.log(P.apy) - np.log(P[col])
    r2 = 1 - (e ** 2).sum() / ((np.log(P.apy) - np.log(P.apy).mean()) ** 2).sum()
    rel = np.abs(P.apy / (P[col] * (P.apy.sum() / P[col].sum())) - 1)
    return round(r2, 3), f"{np.median(rel):.1%}", f"{(rel < 0.25).mean():.0%}"


def part_a():
    rows = []
    D2, D1 = fm.training(2), fm.training(1)
    for lab, D, feats in [("lag 2 base", D2, fm.FEATS_BASE), ("lag 2 + weeks 1-11", D2, fm.FEATS_PARTIAL),
                          ("lag 1 (full season, ceiling)", D1, fm.FEATS_BASE)]:
        for chk in ("leave-year-out", "forward 2022-26"):
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
            rows.append({"model": lab, "check": chk, "n": len(P), "R2": r2, "median miss": med, "within 25%": w25})
    print("A. Contract model, UFA signings")
    print(pd.DataFrame(rows).to_string(index=False))


class PartialComps(vt.Comps):
    """Comps with the anchor season's weeks 1-11 (p_share, p_gp) added to the linear shift."""
    FEATS = vt.RAW_FEATS + ["p_share", "p_gp"]

    def __init__(self, A):
        super().__init__(A)
        self.beta2 = {}
        for g, G in A.groupby("grp"):
            for k in range(1, vt.H + 1):
                d = G[G[f"y{k}"].notna()]
                X = np.column_stack([np.ones(len(d)), d[self.FEATS].to_numpy()])
                self.beta2[(g, k)] = np.linalg.lstsq(X, d[f"y{k}"].to_numpy(), rcond=None)[0][1:]

    def outcomes2(self, row, k):
        row, x = self._x(row)
        g = row["grp"]
        nb = self.neighbors(g, x, row["pfr_id"], row["Y"], need=k)
        P = self.A.loc[nb]
        base = (P[f"y{k}"].to_numpy() + (np.array([row[f] for f in vt.RAW_FEATS], float) - P[vt.RAW_FEATS].to_numpy())
                @ self.beta[(g, k)]).clip(min=0)
        part = (P[f"y{k}"].to_numpy() + (np.array([row[f] for f in self.FEATS], float) - P[self.FEATS].to_numpy())
                @ self.beta2[(g, k)]).clip(min=0)
        return base.mean(), part.mean()


def part_b():
    A = pd.read_parquet(vt.OUT / "vet_anchors.parquet")
    pp = []
    for Y in sorted(A.Y.unique()):
        Pt = fm.partial(int(Y))
        if len(Pt):
            pp.append(Pt[["p_share", "p_gp"]].assign(Y=Y).reset_index(names="pfr_id"))
    A = A.merge(pd.concat(pp), on=["pfr_id", "Y"], how="left")
    A[["p_share", "p_gp"]] = A[["p_share", "p_gp"]].fillna(0.0)
    C = PartialComps(A)
    S = A[A.Y.between(2014, 2024)].sample(4000, random_state=5)
    rows = []
    for k in (2, 3):
        d = S[S[f"y{k}"].notna()]
        f = np.array([C.outcomes2(r, k) for r in d.to_dict("records")])
        a = d[f"y{k}"].to_numpy()
        e0, e1 = (a - f[:, 0]) ** 2, (a - f[:, 1]) ** 2
        rng = np.random.default_rng(0)
        bs = [(e1 - e0)[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        rows.append({"season": k, "n": len(a), "RMSE base": np.sqrt(e0.mean()) * 100, "RMSE + weeks 1-11": np.sqrt(e1.mean()) * 100,
                     "MSE change (x1e4)": f"{(e1 - e0).mean() * 1e4:+.3f} ({np.percentile(bs, 2.5) * 1e4:+.3f} to {np.percentile(bs, 97.5) * 1e4:+.3f})",
                     "actual/forecast base": a.sum() / f[:, 0].sum(), "actual/forecast +partial": a.sum() / f[:, 1].sum()})
        if k == 2:
            d = d.assign(f0=f[:, 0], f1=f[:, 1], hurt=d.p_gp < 0.5)
            for lab, x in d.groupby("hurt"):
                print(f"  season 2, {'missed half+ of weeks 1-11' if lab else 'played most of weeks 1-11'} (n {len(x)}): "
                      f"actual/forecast base {x.y2.sum() / x.f0.sum():.2f}, +partial {x.y2.sum() / x.f1.sum():.2f}")
    print("\nB. Value forecast (% of cap), seasons after the one in progress")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    part_a()
    part_b()


if __name__ == "__main__":
    main()
