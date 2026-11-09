"""Within-position spread: does the market's flat response to our credit come from noisy quality measures?

Absent starters (carry-forward). Credit under the current rule in the rookie currency
rate x (curve_g(s, q)/1.10 - 0.5% x s) with q from: prior season, same season (his grade over the games
he did play that season: closer to his true quality then, not caused by the absence game), or their mean.
Regression: spread (and margin) ~ team-season FE + count of absent starters by group (absorbs each
position's level) + beta x credit. beta = within-position slope (1 = the curve spreads value as the market
does). Also beta by QB / non-QB, and the year-to-year reliability of q.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_units as ru  # noqa: E402
import absence_test as at  # noqa: E402

GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


def main():
    C = ru.capture()
    cur, q, neutral, pos, p2g = C["curves"], C["q"], C["neutral"], C["pos"], C["pfr2gsis"]
    A = pd.read_parquet(at.OUT / "absence_starters_carry.parquet")
    A["gsis"] = A.pfr.map(p2g)
    A["g"] = A.gsis.map(pos)
    A = A[A.g.isin(cur) | ~A.absent].copy()
    rate = np.where(A.g == "QB", 0.34, 0.19)
    neu = lambda g, y: neutral.get((g, y), neutral[(g, None)])
    qp = np.array([q.get((a, y - 1), np.nan) for a, y in zip(A.gsis, A.season)])
    qc = np.array([q.get((a, y), np.nan) for a, y in zip(A.gsis, A.season)])
    A["qp"], A["qc"] = qp, qc
    ab = A.absent.to_numpy()
    for name, qq in [("prior", qp), ("current", qc), ("mean", np.nanmean(np.c_[qp, qc], axis=1))]:
        c = np.zeros(len(A))
        for i in np.where(ab)[0]:
            g, s, y = A.g.iat[i], min(A.avg_share.iat[i], 1.0), A.season.iat[i]
            qi = qq[i] if not np.isnan(qq[i]) else neu(g, y)
            c[i] = rate[i] * (cur[g](s, qi) / ru.UNITS * 100 - 0.5 * s)
        A[f"c_{name}"] = c
    for g in GROUPS:
        A[f"n_{g}"] = (A.absent & (A.g == g)).astype(float)
    both = A.absent & ~np.isnan(A.qp) & ~np.isnan(A.qc)
    print(f"absent starters with both grades: {both.sum()} of {A.absent.sum()}")
    # reliability of q: corr(prior, current) among absent starters with both, by group
    rel = A[both].groupby("g").apply(lambda d: d.qp.corr(d.qc), include_groups=False)
    print("year-to-year correlation of quality q (absent starters):", rel.round(2).to_dict())

    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for col in ["home_team", "away_team"]:
        s[col] = s[col].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    cnt = [f"n_{g}" for g in GROUPS]
    print("\nWithin-position slope on the credit (1 = market agrees):")
    for y in ["spread_line", "result"]:
        for name in ["prior", "current", "mean"]:
            M, n, g_ = at.design(s, A, cnt + [f"c_{name}"])
            f = at.fit(g_[y].to_numpy(float), M, n, g_.home_team.to_numpy())
            t = f"c_{name}"
            print(f"   {y:11s} q={name:7s} beta {-f.loc[t, 'b']:.2f} [{-f.loc[t, 'hi']:.2f},{-f.loc[t, 'lo']:.2f}]")
        # horse race prior vs current
        M, n, g_ = at.design(s, A, cnt + ["c_prior", "c_current"])
        f = at.fit(g_[y].to_numpy(float), M, n, g_.home_team.to_numpy())
        print(f"   {y:11s} both: prior {-f.loc['c_prior', 'b']:.2f} [{-f.loc['c_prior', 'hi']:.2f},"
              f"{-f.loc['c_prior', 'lo']:.2f}], current {-f.loc['c_current', 'b']:.2f} "
              f"[{-f.loc['c_current', 'hi']:.2f},{-f.loc['c_current', 'lo']:.2f}]")
    # QB vs non-QB, current quality
    for name in ["prior", "current"]:
        A[f"cq_{name}"] = np.where(A.g == "QB", A[f"c_{name}"], 0)
        A[f"cn_{name}"] = np.where(A.g != "QB", A[f"c_{name}"], 0)
        M, n, g_ = at.design(s, A, cnt + [f"cq_{name}", f"cn_{name}"])
        f = at.fit(g_.spread_line.to_numpy(float), M, n, g_.home_team.to_numpy())
        print(f"   spread q={name}: QB {-f.loc[f'cq_{name}', 'b']:.2f} [{-f.loc[f'cq_{name}', 'hi']:.2f},"
              f"{-f.loc[f'cq_{name}', 'lo']:.2f}], non-QB {-f.loc[f'cn_{name}', 'b']:.2f} "
              f"[{-f.loc[f'cn_{name}', 'hi']:.2f},{-f.loc[f'cn_{name}', 'lo']:.2f}]")
    # per group, current quality
    print("\nPer group, spread, q=current:")
    for gname in GROUPS:
        A["cg"] = np.where(A.g == gname, A.c_current, 0)
        M, n, g_ = at.design(s, A, cnt + ["cg"])
        f = at.fit(g_.spread_line.to_numpy(float), M, n, g_.home_team.to_numpy())
        sd = A[A.absent & (A.g == gname)].c_current.std()
        print(f"   {gname:5s} beta {-f.loc['cg', 'b']:.2f} [{-f.loc['cg', 'hi']:.2f},{-f.loc['cg', 'lo']:.2f}]"
              f"  (credit sd {sd:.2f})")
    A.to_parquet(at.OUT / "within_spread.parquet")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def residual_check():
    """Does the market under-react to quality? margin - spread on the same terms."""
    A = pd.read_parquet(at.OUT / "within_spread.parquet")
    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for col in ["home_team", "away_team"]:
        s[col] = s[col].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    cnt = [f"n_{g}" for g in GROUPS]
    for name in ["prior", "current"]:
        for sub, lab in [(None, "all"), ("QB", "QB"), ("non", "non-QB")]:
            A["cc"] = A[f"c_{name}"] if sub is None else np.where((A.g == "QB") == (sub == "QB"), A[f"c_{name}"], 0)
            M, n, g_ = at.design(s, A, cnt + ["cc"])
            f = at.fit((g_.result - g_.spread_line).to_numpy(float), M, n, g_.home_team.to_numpy())
            print(f"   margin - spread, q={name:7s} {lab:6s}: {-f.loc['cc', 'b']:+.2f} "
                  f"[{-f.loc['cc', 'hi']:+.2f},{-f.loc['cc', 'lo']:+.2f}]")


if __name__ == "__main__" and sys.argv[1:2] == ["resid"]:
    residual_check()


def qb_bins():
    """Market and result effect of an absent starting QB by pre-season price, vs the rule's credit."""
    A = pd.read_parquet(at.OUT / "absence_starters_carry.parquet")
    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for col in ["home_team", "away_team"]:
        s[col] = s[col].replace(at.RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    bins = [("<3%", 0, 3), ("3-8%", 3, 8), ("8-15%", 8, 15), (">15%", 15, 100)]
    cols = []
    for lab, lo, hi in bins:
        c = f"qb {lab}"
        A[c] = (A.absent & (A.grp == "QB") & A.price_f.between(lo, hi, inclusive="left")).astype(float)
        cols.append(c)
    A["nonqb"] = (A.absent & (A.grp != "QB")).astype(float)
    M, n, g_ = at.design(s, A, cols + ["nonqb"])
    fs = at.fit(g_.spread_line.to_numpy(float), M, n, g_.home_team.to_numpy())
    fr = at.fit(g_.result.to_numpy(float), M, n, g_.home_team.to_numpy())
    print("\nAbsent starting QB by pre-season price: market (spread), actual margin, rule's credit (contract prices)")
    for lab, lo, hi in bins:
        c = f"qb {lab}"
        m = A[c] == 1
        print(f"   {lab:6s} n={int(m.sum()):4d}  market {-fs.loc[c, 'b']:.2f} [{-fs.loc[c, 'hi']:.2f},{-fs.loc[c, 'lo']:.2f}]"
              f"  margin {-fr.loc[c, 'b']:.2f} [{-fr.loc[c, 'hi']:.2f},{-fr.loc[c, 'lo']:.2f}]"
              f"  credit {A.loc[m, 'model_pts'].mean():.2f}  avg price {A.loc[m, 'price_f'].mean():.1f}")


if __name__ == "__main__" and sys.argv[1:2] == ["qb"]:
    qb_bins()
