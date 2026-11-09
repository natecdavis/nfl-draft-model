"""Why does the market price absent EDGE starters so low (0.38 pts vs 0.89 credited)?

Uses the carry-forward absences (scripts/absence_test.py carry). For each absent starter:
  1. Who takes his snaps: teammates in the same pricing group whose share in that game rose vs their
     average over the team's previous 4 games. Gain-weighted: their prior share (already in the rotation?),
     prior-season quality, and whether the gainer is new (no snaps in the previous 4 games).
  2. Market effect split by the absent starter's prior-season quality (tercile within group), by pre-season
     price, and by backup depth (gain-weighted prior share of his replacements).
Compared across pricing groups, EDGE vs the rest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_units as ru  # noqa: E402
import absence_test as at  # noqa: E402


def gainers(A, pos, p2g, q, neutral):
    s = pd.read_parquet(at.RAW / "snap_counts.parquet")
    s = s[s.game_type == "REG"].copy()
    s["team"] = s.team.replace(at.RELOC)
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["g"] = s.pfr_player_id.map(p2g).map(pos)
    tg = s[["season", "team", "week", "game_id"]].drop_duplicates().sort_values(["season", "team", "week"])
    tg["k"] = tg.groupby(["season", "team"]).cumcount()
    s = s.merge(tg[["game_id", "team", "k"]], on=["game_id", "team"])
    key = s.set_index(["season", "team", "k"])
    kof = dict(zip(zip(tg.game_id, tg.team), zip(tg.season, tg.k)))
    out = []
    ab = A[A.absent & A.g.notna()]
    by_team = {k: G for k, G in s.groupby(["season", "team"])}
    for r in ab.itertuples():
        yr, k = kof[(r.game_id, r.team)]
        G = by_team[(yr, r.team)]
        G = G[G.g == r.g]
        now = G[G.k == k].set_index("pfr_player_id").share
        prev = G[(G.k < k) & (G.k >= k - 4)].groupby("pfr_player_id").share.sum() / min(4, k)
        d = now.subtract(prev.reindex(now.index).fillna(0))
        d = d[(d > 0.02) & (d.index != r.pfr)]
        if d.empty:
            out.append((r.Index, 0.0, np.nan, np.nan, np.nan))
            continue
        ps = prev.reindex(d.index).fillna(0)
        qq = np.array([q.get((p2g.get(p), yr - 1), neutral.get((r.g, yr), neutral[(r.g, None)])) for p in d.index])
        w = d / d.sum()
        out.append((r.Index, d.sum(), float((w * ps).sum()), float((w * qq).sum()), float((w * (ps < 0.05)).sum())))
    O = pd.DataFrame(out, columns=["i", "gain", "repl_prior_share", "repl_q", "repl_new"]).set_index("i")
    return O


def main():
    C = ru.capture()
    pos, p2g, q, neutral, cur = C["pos"], C["pfr2gsis"], C["q"], C["neutral"], C["curves"]
    A = pd.read_parquet(at.OUT / "absence_starters_carry.parquet")
    A["gsis"] = A.pfr.map(p2g)
    A["g"] = A.gsis.map(pos)
    A.loc[~A.g.isin(cur), "g"] = np.nan
    A["q_prior"] = [q.get((gs, y - 1), np.nan) for gs, y in zip(A.gsis, A.season)]
    O = gainers(A, pos, p2g, q, neutral)
    A = A.join(O)
    ab = A[A.absent & A.g.notna()]
    print("\nWho replaces an absent starter (gain-weighted), by pricing group:")
    T = ab.groupby("g").agg(n=("pfr", "size"), starter_share=("avg_share", "mean"), snaps_regained=("gain", "mean"),
                            repl_prior_share=("repl_prior_share", "mean"), repl_new=("repl_new", "mean"),
                            repl_q=("repl_q", "mean"), starter_q=("q_prior", "mean"), price=("price_f", "mean"))
    print(T.round(2).to_string())

    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    cl = None

    def split(name, mask_fn, labels, groups=("EDGE", "DL", "OL", "WR", "DB")):
        cols = []
        for g in groups:
            for lab in labels:
                c = f"{g}|{lab}"
                A[c] = (A.absent & (A.g == g) & mask_fn(A, g, lab)).astype(float)
                cols.append(c)
        # all other absences as controls
        A["other"] = (A.absent & ~A.g.isin(groups)).astype(float)
        A["otherQB"] = (A.absent & (A.grp == "QB")).astype(float)
        A["other"] -= A.otherQB.clip(upper=A.other)
        M, n, g_ = at.design(s, A, cols + ["other", "otherQB"])
        f = at.fit(g_.spread_line.to_numpy(float), M, n, g_.home_team.to_numpy())
        print(f"\nMarket effect (pts) by {name}:")
        for g in groups:
            parts = [f"{lab}: {-f.loc[f'{g}|{lab}', 'b']:.2f} [{-f.loc[f'{g}|{lab}', 'hi']:.2f},"
                     f"{-f.loc[f'{g}|{lab}', 'lo']:.2f}] n={int(A[f'{g}|{lab}'].sum())}" for lab in labels]
            print(f"   {g:5s} " + "   ".join(parts))

    # quality terciles within group (prior season); no prior grade -> own bin
    qt = {}
    for g in ["EDGE", "DL", "OL", "WR", "DB"]:
        x = A[A.absent & (A.g == g)].q_prior.dropna()
        qt[g] = x.quantile([1 / 3, 2 / 3]).to_numpy()

    def qbin(A, g, lab):
        lo, hi = qt[g]
        return {"low": A.q_prior < lo, "mid": A.q_prior.between(lo, hi), "high": A.q_prior > hi,
                "none": A.q_prior.isna()}[lab]
    split("absent starter's prior-season quality (tercile within group)", qbin, ["low", "mid", "high", "none"])

    def pbin(A, g, lab):
        return {"<2%": A.price_f < 2, "2-6%": A.price_f.between(2, 6), ">6%": A.price_f > 6}[lab]
    split("absent starter's pre-season price", pbin, ["<2%", "2-6%", ">6%"])

    def dbin(A, g, lab):
        return {"rotation (repl prior share >= 0.3)": A.repl_prior_share >= 0.3,
                "depth (< 0.3)": A.repl_prior_share < 0.3}[lab]
    split("backup depth", dbin, ["rotation (repl prior share >= 0.3)", "depth (< 0.3)"])

    # what the curve credits by quality bin, EDGE vs others (current rule, rookie currency)
    rate = 0.19
    ab = A[A.absent & A.g.isin(["EDGE", "DL", "OL", "WR", "DB"])].copy()
    ab["credit"] = [rate * (cur[g](min(sh, 1), qq if not np.isnan(qq) else neutral[(g, None)]) / ru.UNITS * 100
                            - 0.5 * sh) for g, sh, qq in zip(ab.g, ab.avg_share, ab.q_prior)]
    ab["qb_"] = pd.cut(ab.groupby("g").q_prior.rank(pct=True), [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"])
    print("\nCurrent rule's credit (pts) by quality tercile:")
    print(ab.groupby(["g", "qb_"], observed=True).credit.mean().unstack().round(2).to_string())
    A.to_parquet(at.OUT / "edge_check.parquet")


if __name__ == "__main__":
    main()
