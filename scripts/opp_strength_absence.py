"""Do some positions matter more against strong opponents (and in the playoffs)?

Contracts price a player on what he adds over a regular season against an average schedule; titles are won
in playoff games against good teams. If a unit's worth over its backup rises with opponent strength faster
than other units', title value per curve-dollar differs from pay by position.

Pre-specified (Nov 15 2026), on the betting-line absence data (scripts/absence_test.py, carry-forward
absences, same 9 groups):
  1. Regular season 2013-25. spread = hfa + r[home] - r[away] + sum_pos b_pos D_pos
         + sum_pos c_pos (abs_home_pos x z_away - abs_away_pos x z_home)
     z = opponent's full-strength team-season rating (team-season effects from the base absence model,
     centered within season, in points). c_pos = extra points per absent starter per +1 pt of opponent.
     Tests: all c = 0; c equal across groups (a common c rescales every position and changes no choice).
     Same on actual margin. Title-relevant worth = b + c x (mean rating of playoff opponents).
  2. Robustness: interaction with a "strong opponent" dummy (opponent rating in the top third).
  3. Playoffs 2013-25 (spreads, starters defined over the previous 4 games of any type): b_pos + d_pos x
     playoff, pooled QB / non-QB and by group; team-season ratings from the regular season.
Decision rule: a position re-weighting is worth a choice test only if equal-c is rejected (p < 0.05) on the
spread AND the margin points the same way.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import absence_test as at  # noqa: E402

GROUPS = at.GROUPS
PLAYOFF = {"WC", "DIV", "CON", "SB"}


def absences_all():
    """at.absences() with carry, over regular season + playoffs (game order runs through the playoffs)."""
    s = pd.read_parquet(at.RAW / "snap_counts.parquet")
    s = s[s.season.between(2013, 2025)].copy()
    s["team"] = s.team.replace(at.RELOC)
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["grp"] = s.position.map(at.GRP)
    s = s.dropna(subset=["grp"])
    order = {"REG": 0, "WC": 1, "DIV": 2, "CON": 3, "SB": 4}
    tg = s[["season", "team", "week", "game_id", "game_type"]].drop_duplicates()
    tg["o"] = tg.game_type.map(order)
    tg = tg.sort_values(["season", "team", "o", "week"])
    tg["k"] = tg.groupby(["season", "team"]).cumcount()
    s = s.merge(tg[["game_id", "team", "k"]], on=["game_id", "team"])
    play = s[s.share > 0]
    rows = []
    for (yr, tm), G in play.groupby(["season", "team"]):
        games = tg[(tg.season == yr) & (tg.team == tm)].set_index("k").game_id
        n = len(games)
        P = G.pivot_table(index="pfr_player_id", columns="k", values="share", aggfunc="max").reindex(
            columns=range(n))
        grp = G.groupby("pfr_player_id").grp.agg(lambda x: x.mode().iloc[0])
        A = P.to_numpy()
        out = np.zeros(len(A), bool)
        keep = np.zeros(len(A))
        for k in range(2, n):
            W = A[:, max(0, k - 4):k]
            played = ~np.isnan(W)
            frac = played.mean(axis=1)
            avg = np.where(played.any(axis=1), np.nansum(W, axis=1) / np.maximum(played.sum(axis=1), 1), 0)
            reg = ((frac >= 0.75) & (avg >= 0.6)) | out
            absent = reg & np.isnan(A[:, k])
            out = absent.copy()
            for i in np.where(reg)[0]:
                if avg[i] > 0:
                    keep[i] = avg[i]
                rows.append((games[k], tm, yr, P.index[i], grp[P.index[i]], bool(absent[i]), float(keep[i])))
    return pd.DataFrame(rows, columns=["game_id", "team", "season", "pfr", "grp", "absent", "avg_share"])


def schedule(types):
    s = pd.read_parquet(at.RAW / "schedules.parquet")
    s = s[s.game_type.isin(types) & s.season.between(2013, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(at.RELOC)
    return s


def ts_design(g, ts):
    ix = {t: i for i, t in enumerate(ts)}
    X = np.zeros((len(g), len(ts)))
    X[np.arange(len(g)), (g.home_team + g.season.astype(str)).map(ix)] = 1
    X[np.arange(len(g)), (g.away_team + g.season.astype(str)).map(ix)] = -1
    return X


def team_counts(A, g, cols):
    agg = A.groupby(["game_id", "team"])[cols].sum()
    H = agg.reindex(list(zip(g.game_id, g.home_team))).fillna(0).to_numpy()
    W = agg.reindex(list(zip(g.game_id, g.away_team))).fillna(0).to_numpy()
    return H, W


def ols(y, M, k, groups, names):
    m = sm.OLS(y, M).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = m.conf_int()
    T = pd.DataFrame({"b": m.params[:k], "lo": ci[:k, 0], "hi": ci[:k, 1]}, index=names)
    return m, T


def ratings(g, A, cols, ts):
    """Full-strength team-season ratings from the base absence model on the spread, centered by season."""
    H, W = team_counts(A, g, cols)
    hfa = (g.location != "Neutral").astype(float).to_numpy()[:, None]
    X = ts_design(g, ts)
    M = np.hstack([H - W, hfa, X])
    beta = np.linalg.lstsq(M, g.spread_line.to_numpy(float), rcond=None)[0]
    r = pd.Series(beta[len(cols) + 1:], index=ts)
    yr = pd.Series([t[-4:] for t in ts], index=ts)
    return r - r.groupby(yr).transform("mean")


def wald_equal(m, idx):
    """p-value for equal coefficients among positions idx (and for all zero)."""
    k = len(idx)
    R = np.zeros((k - 1, len(m.params)))
    for j in range(k - 1):
        R[j, idx[j]] = 1
        R[j, idx[j + 1]] = -1
    Z = np.zeros((k, len(m.params)))
    for j in range(k):
        Z[j, idx[j]] = 1
    return float(m.wald_test(R, scalar=True).pvalue), float(m.wald_test(Z, scalar=True).pvalue)


def main():
    cache = at.OUT / "absence_starters_allgames.parquet"
    if cache.exists():
        A = pd.read_parquet(cache)
    else:
        A = absences_all()
        A.to_parquet(cache)
    for gname in GROUPS:
        A[f"abs_{gname}"] = (A.absent & (A.grp == gname)).astype(float)
    A["abs_nonQB"] = (A.absent & (A.grp != "QB")).astype(float)
    cols = [f"abs_{g}" for g in GROUPS]

    reg = schedule({"REG"})
    reg = reg[reg.game_id.isin(set(A.game_id))].reset_index(drop=True)
    post = schedule(PLAYOFF)
    post = post[post.game_id.isin(set(A.game_id))].reset_index(drop=True)
    ts = sorted(set(reg.home_team + reg.season.astype(str)) | set(reg.away_team + reg.season.astype(str)))
    r = ratings(reg, A, cols, ts)
    print(f"regular-season games {len(reg)}, playoff games {len(post)}; absent starters reg "
          f"{int(A[A.game_id.isin(set(reg.game_id))].absent.sum())}, playoffs "
          f"{int(A[A.game_id.isin(set(post.game_id))].absent.sum())}")
    zh = (reg.home_team + reg.season.astype(str)).map(r).to_numpy()
    za = (reg.away_team + reg.season.astype(str)).map(r).to_numpy()
    print(f"team-season rating sd {r.std():.2f} pts")
    zp = pd.concat([(post.home_team + post.season.astype(str)).map(r), (post.away_team + post.season.astype(str)).map(r)])
    zbar = float(zp.mean())
    print(f"mean rating of playoff teams (= playoff opponents) {zbar:+.2f}")

    # 1. continuous interaction
    H, W = team_counts(A, reg, cols)
    hfa = (reg.location != "Neutral").astype(float).to_numpy()[:, None]
    X = ts_design(reg, ts)
    I = H * za[:, None] - W * zh[:, None]
    M = np.hstack([H - W, I, hfa, X])
    names = [f"b_{g}" for g in GROUPS] + [f"c_{g}" for g in GROUPS]
    cl = reg.home_team.to_numpy()
    out = {}
    for y, lab in [("spread_line", "spread"), ("result", "margin")]:
        m, T = ols(reg[y].to_numpy(float), M, 18, cl, names)
        pe, p0 = wald_equal(m, list(range(9, 18)))
        b, c = T.b.iloc[:9].to_numpy(), T.b.iloc[9:].to_numpy()
        tab = pd.DataFrame({"b (avg opp)": b, "c per +1 pt opp": c, "c lo": T.lo.iloc[9:].to_numpy(),
                            "c hi": T.hi.iloc[9:].to_numpy(), f"worth vs playoff opp (z={zbar:+.1f})": b + c * zbar,
                            "ratio": (b + c * zbar) / b}, index=GROUPS)
        # common-c model (one interaction for all groups)
        Ic = I.sum(axis=1, keepdims=True)
        mc, Tc = ols(reg[y].to_numpy(float), np.hstack([H - W, Ic, hfa, X]), 10, cl,
                     [f"b_{g}" for g in GROUPS] + ["c_common"])
        print(f"\n=== 1. {lab}: points per absent starter, by opponent rating ===")
        print(tab.round(3).to_string())
        print(f"all c = 0: p {p0:.3f}; c equal across groups: p {pe:.3f}; common c {Tc.b.iloc[9]:+.3f} "
              f"({Tc.lo.iloc[9]:+.3f} to {Tc.hi.iloc[9]:+.3f})")
        out[lab] = tab
    pd.concat(out, axis=1).to_csv(at.OUT / "opp_strength_absence.csv")

    # QB / non-QB pooled
    for y in ["spread_line", "result"]:
        H2, W2 = team_counts(A, reg, ["abs_QB", "abs_nonQB"])
        I2 = H2 * za[:, None] - W2 * zh[:, None]
        m, T = ols(reg[y].to_numpy(float), np.hstack([H2 - W2, I2, hfa, X]), 4, cl,
                   ["b_QB", "b_nonQB", "c_QB", "c_nonQB"])
        print(f"\nQB / non-QB ({y}):")
        print(T.round(3).to_string())

    # 2. strong-opponent dummy (top third of team-season ratings)
    cut = r.quantile(2 / 3)
    sa, sh = (za >= cut).astype(float), (zh >= cut).astype(float)
    I3 = H * sa[:, None] - W * sh[:, None]
    for y, lab in [("spread_line", "spread"), ("result", "margin")]:
        m, T = ols(reg[y].to_numpy(float), np.hstack([H - W, I3, hfa, X]), 18, cl, names)
        pe, p0 = wald_equal(m, list(range(9, 18)))
        b, d = T.b.iloc[:9].to_numpy(), T.b.iloc[9:].to_numpy()
        print(f"\n=== 2. {lab}: strong opponent (top third, rating >= {cut:+.1f}) ===")
        print(pd.DataFrame({"weak/avg opp": b, "strong opp": b + d, "diff": d, "lo": T.lo.iloc[9:].to_numpy(),
                            "hi": T.hi.iloc[9:].to_numpy()}, index=GROUPS).round(3).to_string())
        print(f"all diffs 0: p {p0:.3f}; diffs equal: p {pe:.3f}")

    # 3. playoffs: regular + playoff games, ratings fixed from the regular season (offset)
    both = pd.concat([reg.assign(po=0.0), post.assign(po=1.0)], ignore_index=True)
    both = both[(both.home_team + both.season.astype(str)).isin(r.index)
                & (both.away_team + both.season.astype(str)).isin(r.index)].reset_index(drop=True)
    off = ((both.home_team + both.season.astype(str)).map(r) - (both.away_team + both.season.astype(str)).map(r)).to_numpy()
    hfa_b = (both.location != "Neutral").astype(float).to_numpy()[:, None]
    po = both.po.to_numpy()[:, None]
    for grpset, lab in [(["abs_QB", "abs_nonQB"], "QB / non-QB"), (cols, "by group")]:
        Hb, Wb = team_counts(A, both, grpset)
        D = Hb - Wb
        k = len(grpset)
        nm = [f"b_{c[4:]}" for c in grpset] + [f"po_{c[4:]}" for c in grpset]
        for y in ["spread_line", "result"]:
            yy = both[y].to_numpy(float) - off
            M = np.hstack([D, D * po, hfa_b, hfa_b * po])
            m, T = ols(yy, M, 2 * k, both.home_team.to_numpy(), nm)
            n_po = (np.abs(D[both.po == 1]) > 0).sum(axis=0)
            print(f"\n=== 3. playoffs, {lab} ({y}, ratings offset from regular season) ===")
            print(pd.DataFrame({"regular": T.b.iloc[:k].to_numpy(), "playoff extra": T.b.iloc[k:].to_numpy(),
                                "lo": T.lo.iloc[k:].to_numpy(), "hi": T.hi.iloc[k:].to_numpy(),
                                "playoff games w/ absence": n_po}, index=[c[4:] for c in grpset]).round(3).to_string())
            if k > 2:
                pe, p0 = wald_equal(m, list(range(k, 2 * k)))
                print(f"all playoff extras 0: p {p0:.3f}; equal: p {pe:.3f}")


if __name__ == "__main__":
    main()
