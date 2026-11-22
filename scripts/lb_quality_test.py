"""A better linebacker quality measure? (lead: scripts/lb_value_test.py / lb_curve_label.py)

The LB quality percentile q (darko talent from PFF overall defense grade) explains LB pay worst of any group (corr 0.35)
and LB pay above the curve buys team strength (+0.22 pts per 1% of cap, p 0.005). Candidate measures from PFF's
defense summary, season before signing (LBs = veteran pricing group LB, >= 200 defensive snaps):
  M0  current: snap share, q, age
  M1  + PFF sub-grades: coverage, run defense, tackling, pass rush
  M2  + role and rates: coverage / pass-rush share of snaps, stops, tackles per snap, missed-tackle rate,
      plays on the ball per coverage snap, pressures per rush snap, yards allowed per coverage snap
  G   season overall grade instead of q (is the talent model's smoothing the problem?)
A  Pay at signing: log APY (veteran contracts: UFA / extension / tag, >= 0.5% of cap, 2015-2026), OLS,
   leave-signing-year-out R2.
B  Persistence: year-to-year correlation of each model's quality part (fitted contribution of the non-share, non-age
   terms) for the same player in consecutive seasons.
C  Team strength (the test that matters): each LB season valued at the model's out-of-year predicted APY x snap share
   x games, rescaled to the old curve's league total each season; team rating ~ other groups' curve value + LB value
   + LB veterans' pay above that value. A better measure leaves no LB excess-pay signal and keeps the LB rate at the
   other groups' level; compared with the current curve value on the same rows.
Pre-set: worth building into the label only if A improves R2 materially (+0.05), C's LB excess falls to ~0 (CI
covering 0) and the team-strength fit is no worse.
Run: .venv/bin/python scripts/lb_quality_test.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import age_pricing as ap  # noqa: E402
import fa_market as fm  # noqa: E402
import lb_value_test as lv  # noqa: E402

PFF = __import__("pathlib").Path("/Users/nathanieldavis/cfdarko/data/raw/pff_nfl")
YEARS = range(2013, 2026)
MIN_SNAPS = 200
Q_FEATS = {"M0": ["q"],
           "M1": ["q", "g_cov", "g_run", "g_tack", "g_prsh"],
           "M2": ["q", "g_cov", "g_run", "g_tack", "g_prsh", "cov_sh", "rush_sh", "stops_ps", "tkl_ps", "mtr",
                  "ball_pc", "pres_pr", "yds_pc"],
           "G": ["g_def"]}
BASE = ["share", "age", "age2"]


def pff_lb():
    pl = pd.read_parquet(fm.RAW / "players.parquet")[["pfr_id", "pff_id", "gsis_id"]]
    xw = pd.read_csv("/Users/nathanieldavis/cfdarko/data/raw/rookie_value/nflverse_players.csv", low_memory=False,
                     usecols=["gsis_id", "pff_id"]).dropna()
    xw = xw.assign(pff_id=xw.pff_id.astype(int)).drop_duplicates("pff_id").merge(
        pl[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id"), on="gsis_id")
    m1 = dict(zip(xw.pff_id, xw.pfr_id))
    m1.update({int(a): b for a, b in zip(pl.pff_id, pl.pfr_id) if pd.notna(a) and pd.notna(b)})
    rows = []
    for y in YEARS:
        d = pd.read_csv(PFF / str(y) / "defense_summary.csv")
        d["season"] = y
        rows.append(d)
    d = pd.concat(rows)
    d["pfr_id"] = d.player_id.map(m1)
    d = d[d.pfr_id.notna() & (d.snap_counts_defense >= MIN_SNAPS)].sort_values("snap_counts_defense")
    d = d.drop_duplicates(["pfr_id", "season"], keep="last")
    n = d.snap_counts_defense
    cov, rush = d.snap_counts_coverage.clip(lower=1), d.snap_counts_pass_rush.clip(lower=1)
    out = pd.DataFrame({"pfr_id": d.pfr_id, "season": d.season, "g_def": d.grades_defense,
                        "g_cov": d.grades_coverage_defense, "g_run": d.grades_run_defense, "g_tack": d.grades_tackle,
                        "g_prsh": d.grades_pass_rush_defense, "cov_sh": d.snap_counts_coverage / n,
                        "rush_sh": d.snap_counts_pass_rush / n, "stops_ps": d.stops / n,
                        "tkl_ps": (d.tackles + d.assists) / n, "mtr": d.missed_tackle_rate,
                        "ball_pc": (d.pass_break_ups + d.interceptions) / cov, "pres_pr": d.total_pressures / rush,
                        "yds_pc": d.yards / cov})
    return out


def season_frame():
    v = pd.read_parquet("data/derived/vet_season_values.parquet").sort_values("value_ours")
    v = v.drop_duplicates(["pfr_id", "season"], keep="last")
    v = v[v.group == "LB"][["pfr_id", "season", "share", "gfrac", "q", "value_ours"]]
    d = v.merge(pff_lb(), on=["pfr_id", "season"])
    born = pd.read_parquet(fm.RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    born = dict(zip(born.pfr_id, pd.to_datetime(born.birth_date, errors="coerce")))
    d["age"] = [(pd.Timestamp(f"{s}-09-01") - born.get(p)).days / 365.25 if pd.notna(born.get(p)) else np.nan
                for p, s in zip(d.pfr_id, d.season)]
    d["age"] = d.age.fillna(d.age.median())
    d["age2"] = (d.age - 27) ** 2
    cols = sorted({c for f in Q_FEATS.values() for c in f} - {"q"})
    d[cols] = d[cols].fillna(d[cols].median())
    return d


def contracts(S):
    H = fm.history()
    rows = []
    for p in S.pfr_id.unique():
        for (y, kind, apy, yrs, gtd, team) in H.get(p, []):
            if kind in ("UFA", "Extension", "Franchise", "Transition") and apy >= 0.005 and 2015 <= y <= 2026:
                rows.append((p, y, apy))
    C = pd.DataFrame(rows, columns=["pfr_id", "Y", "apy"]).drop_duplicates(["pfr_id", "Y"], keep="last")
    C["season"] = C.Y - 1
    C = C.merge(S, on=["pfr_id", "season"])
    C["log_apy"] = np.log(C.apy)
    return C


def formula(m):
    return "log_apy ~ " + " + ".join(BASE + Q_FEATS[m])


def cv_pay(C):
    print(f"\nA  pay at signing: {len(C)} LB contracts {C.Y.min()}-{C.Y.max()} (season-before features)")
    res = {}
    for m in Q_FEATS:
        pred = pd.Series(index=C.index, dtype=float)
        for Y in C.Y.unique():
            tr, te = C.Y != Y, C.Y == Y
            pred[te] = smf.ols(formula(m), C[tr]).fit().predict(C[te])
        r2 = 1 - ((C.log_apy - pred) ** 2).sum() / ((C.log_apy - C.log_apy.mean()) ** 2).sum()
        res[m] = r2
        print(f"  {m}: leave-year-out R2 {r2:.3f}; corr(pred APY, APY) {np.corrcoef(np.exp(pred), C.apy)[0, 1]:.3f}")
    fit = smf.ols(formula("M2"), C).fit()
    print("  M2 terms (t):", {k: round(v, 1) for k, v in fit.tvalues.drop("Intercept").items()})
    return res


def quality_part(fit, d, m):
    return sum(fit.params[c] * d[c] for c in Q_FEATS[m])


def persistence(C, S):
    print("\nB  persistence of each model's quality part (same player, consecutive seasons):")
    for m in Q_FEATS:
        fit = smf.ols(formula(m), C).fit()
        q = S.assign(qp=quality_part(fit, S, m))[["pfr_id", "season", "qp"]]
        j = q.merge(q.assign(season=q.season - 1), on=["pfr_id", "season"], suffixes=("", "_next"))
        print(f"  {m}: r {j.qp.corr(j.qp_next):.3f} (n {len(j)})")


def team_strength(C, S):
    ps = lv.data()
    out = {}
    for m in ["old"] + list(Q_FEATS):
        if m == "old":
            alt = S.assign(alt=S.value_ours)
        else:
            alt = []
            for y in S.season.unique():
                fit = smf.ols(formula(m), C[C.season != y]).fit()  # contracts priced off other seasons
                s = S[S.season == y]
                alt.append(s.assign(alt=np.exp(fit.predict(s)) * s.gfrac * 1.0))
            alt = pd.concat(alt)
            # same league LB total as the old curve each season: only the distribution changes
            k = alt.groupby("season").value_ours.transform("sum") / alt.groupby("season").alt.transform("sum")
            alt["alt"] *= k
        a = alt.set_index(["pfr_id", "season"]).alt
        P = ps.copy()
        lb = P.g == "LB"
        key = list(zip(P.pfr_player_id, P.season))
        has = np.array([k in a.index for k in key]) & lb.to_numpy()
        # LB seasons without a PFF row (< 200 snaps) keep their curve value
        newv = np.array([a.get(k, np.nan) for k in key])
        P.loc[has, "curve"] = newv[has] * P.loc[has, "w"] * 100
        full = (P.share * P.gfrac.fillna(1)).replace(0, np.nan)
        cf = (P.curve / P.w.replace(0, np.nan) / full).where(P.share >= 0.2)
        P["excess"] = np.where(lb & P.vet & cf.notna(), P.share * P.w.clip(upper=1) * (P.price - cf), P.excess)
        print(f"\nC  team strength, LB value = {m}")
        mdl, T = lv.regress(P, ["curve", "excess"], f"curve + excess ({m})", test_lb="excess")
        out[m] = (mdl.rsquared, T.loc["curve_LB"], T.loc["excess_LB"])
    print("\nsummary (R2, LB rate, LB excess):")
    for m, (r2, c, e) in out.items():
        print(f"  {m:4s} R2 {r2:.3f}; LB rate {c['pts per 1%']:.3f} ({c.lo:.3f} to {c.hi:.3f}); "
              f"LB excess {e['pts per 1%']:+.3f} ({e.lo:+.3f} to {e.hi:+.3f})")


def main():
    pd.set_option("display.width", 220)
    S = season_frame()
    print(f"LB seasons with PFF rows: {len(S)} ({S.pfr_id.nunique()} players)")
    C = contracts(S)
    cv_pay(C)
    persistence(C, S)
    team_strength(C, S)


if __name__ == "__main__":
    main()
