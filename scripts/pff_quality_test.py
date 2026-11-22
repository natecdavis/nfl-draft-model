"""Better quality measures from PFF sub-grades and rates, every position group (LB done in scripts/lb_quality_test.py).

The label's quality q (darko talent percentile) comes from one PFF grade per group (QB passing, RB rushing, WR/TE
receiving, OL blocking, DL/EDGE/LB/DB defense overall). For each group, from the same PFF table, season before signing:
  M0  current: snap share, q, age
  M1  + the table's sub-grades
  M2  + role and rate stats (below)
A  Pay at signing (UFA / extension / tag, >= 0.5% of cap, 2015-2026): log APY, OLS, leave-signing-year-out R2.
B  Persistence of each model's quality part, consecutive seasons.
C  Team strength (2013-25 team-seasons, clustered by team): the group's season values replaced by the model's
   out-of-year predicted APY x snap share x games (rescaled to the old league total each season); rating ~ every
   group's value + every group's veteran pay above value. Reported for the changed group: fit R2, its rate, its
   excess-pay slope.
Pre-set (as for LB): worth building into the label when A gains >= +0.05 R2 AND C's team R2 doesn't fall AND the
group's excess-pay slope isn't left > 0. Otherwise documented only.
Result (Nov 22 2026): same-season, DB / EDGE / RB pass (DB team R2 0.607 -> 0.629). Lagged check (quality from the
season before, --lag): every gain vanishes (DB -0.011, EDGE -0.002, RB -0.003, DL / OL / WR within +-0.003; LB -0.003) - the
same-season gains were team context in the rate stats. Nothing adopted.
Run: .venv/bin/python scripts/pff_quality_test.py [GROUP ...]   |   ... --lag [GROUP ...]
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import lb_quality_test as lq  # noqa: E402
import lb_value_test as lv  # noqa: E402

PFF, YEARS = lq.PFF, lq.YEARS
BASE = ["share", "age", "age2"]


def _div(a, b):
    return a / b.where(b > 0)


def defense(d):
    n, cov, rush = d.snap_counts_defense, d.snap_counts_coverage.clip(lower=1), d.snap_counts_pass_rush.clip(lower=1)
    dl = d.snap_counts_dl.clip(lower=1)
    return pd.DataFrame({
        "g_cov": d.grades_coverage_defense, "g_run": d.grades_run_defense, "g_tack": d.grades_tackle,
        "g_prsh": d.grades_pass_rush_defense,
        "cov_sh": d.snap_counts_coverage / n, "rush_sh": d.snap_counts_pass_rush / n, "stops_ps": d.stops / n,
        "mtr": d.missed_tackle_rate, "ball_pc": (d.pass_break_ups + d.interceptions) / cov,
        "pres_pr": d.total_pressures / rush, "sack_pr": d.sacks / rush, "yds_pc": d.yards / cov,
        "slot_sh": d.snap_counts_slot / n, "corner_sh": d.snap_counts_corner / n, "box_sh": d.snap_counts_box / n,
        "fs_sh": d.snap_counts_fs / n, "dl_sh": d.snap_counts_dl / n,
        "agap_sh": d.snap_counts_dl_a_gap / dl, "outt_sh": d.snap_counts_dl_outside_t / dl,
        "qbr_against": d.qb_rating_against, "catch_rate": d.catch_rate}), d.snap_counts_defense


SPECS = {
    "QB": ("passing_summary", "passing_snaps", 150, lambda d: (pd.DataFrame({
        "g_pass": d.grades_pass, "g_run": d.grades_run, "btt": d.btt_rate, "twp": d.twp_rate,
        "acc": d.accuracy_percent, "p2s": d.pressure_to_sack_rate, "adot": d.avg_depth_of_target,
        "ttt": d.avg_time_to_throw, "ypa": d.ypa, "sackp": d.sack_percent}), d.passing_snaps),
        ["g_pass", "g_run"], ["btt", "twp", "acc", "p2s", "adot", "ttt", "ypa", "sackp"]),
    "RB": ("rushing_summary", "attempts", 50, lambda d: (pd.DataFrame({
        "g_run": d.grades_run, "g_route": d.grades_pass_route, "g_pblk": d.grades_pass_block,
        "elu": d.elusive_rating, "yco": d.yco_attempt, "brk": d.breakaway_percent, "yprr": d.yprr,
        "tgt_pr": _div(d.targets, d.routes), "ypa": d.ypa}), d.attempts),
        ["g_run", "g_route", "g_pblk"], ["elu", "yco", "brk", "yprr", "tgt_pr", "ypa"]),
    "WR": ("receiving_summary", "routes", 150, lambda d: (pd.DataFrame({
        "g_route": d.grades_pass_route, "g_drop": d.grades_hands_drop, "g_blk": d.grades_pass_block,
        "yprr": d.yprr, "tgt_pr": _div(d.targets, d.routes), "adot": d.avg_depth_of_target,
        "drop": d.drop_rate, "cc": d.contested_catch_rate, "slot": d.slot_rate,
        "yac": d.yards_after_catch_per_reception, "avt": _div(d.avoided_tackles, d.receptions)}), d.routes),
        ["g_route", "g_drop"], ["yprr", "tgt_pr", "adot", "drop", "cc", "slot", "yac", "avt"]),
    "TE": ("receiving_summary", "routes", 100, lambda d: (pd.DataFrame({
        "g_route": d.grades_pass_route, "g_drop": d.grades_hands_drop, "g_blk": d.grades_pass_block,
        "yprr": d.yprr, "tgt_pr": _div(d.targets, d.routes), "adot": d.avg_depth_of_target,
        "drop": d.drop_rate, "inline": d.inline_rate, "slot": d.slot_rate, "route_rate": d.route_rate,
        "yac": d.yards_after_catch_per_reception}), d.routes),
        ["g_route", "g_drop", "g_blk"], ["yprr", "tgt_pr", "adot", "drop", "inline", "slot", "route_rate", "yac"]),
    "OL": ("offense_blocking", "snap_counts_offense", 200, lambda d: (pd.DataFrame({
        "g_pb": d.grades_pass_block, "g_rb": d.grades_run_block,
        "pres_pb": _div(d.pressures_allowed, d.non_spike_pass_block), "sack_pb": _div(d.sacks_allowed, d.non_spike_pass_block),
        "pen_ps": d.penalties / d.snap_counts_offense, "pbe": d.pbe,
        "tackle_sh": (d.snap_counts_lt + d.snap_counts_rt) / d.snap_counts_offense,
        "center_sh": d.snap_counts_ce / d.snap_counts_offense}), d.snap_counts_offense),
        ["g_pb", "g_rb"], ["pres_pb", "sack_pb", "pen_ps", "pbe", "tackle_sh", "center_sh"]),
    "DL": ("defense_summary", "snap_counts_defense", 200, defense,
           ["g_prsh", "g_run", "g_tack"], ["pres_pr", "sack_pr", "stops_ps", "rush_sh", "mtr", "agap_sh", "outt_sh"]),
    "EDGE": ("defense_summary", "snap_counts_defense", 200, defense,
             ["g_prsh", "g_run", "g_tack", "g_cov"], ["pres_pr", "sack_pr", "stops_ps", "rush_sh", "mtr", "dl_sh", "cov_sh"]),
    "LB": ("defense_summary", "snap_counts_defense", 200, defense,
           ["g_cov", "g_run", "g_tack", "g_prsh"], ["cov_sh", "rush_sh", "stops_ps", "mtr", "ball_pc", "pres_pr", "yds_pc"]),
    "DB": ("defense_summary", "snap_counts_defense", 200, defense,
           ["g_cov", "g_run", "g_tack"], ["ball_pc", "yds_pc", "qbr_against", "catch_rate", "mtr", "stops_ps",
                                          "slot_sh", "corner_sh", "box_sh", "fs_sh"]),
}


def crosswalk():
    pl = pd.read_parquet(fm.RAW / "players.parquet")[["pfr_id", "pff_id", "gsis_id"]]
    xw = pd.read_csv("/Users/nathanieldavis/cfdarko/data/raw/rookie_value/nflverse_players.csv", low_memory=False,
                     usecols=["gsis_id", "pff_id"]).dropna()
    xw = xw.assign(pff_id=xw.pff_id.astype(int)).drop_duplicates("pff_id").merge(
        pl[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id"), on="gsis_id")
    m = dict(zip(xw.pff_id, xw.pfr_id))
    m.update({int(a): b for a, b in zip(pl.pff_id, pl.pfr_id) if pd.notna(a) and pd.notna(b)})
    return m


def season_frame(g, xw):
    table, vol, mn, feats, m1, m2 = SPECS[g]
    rows = []
    for y in YEARS:
        d = pd.read_csv(PFF / str(y) / f"{table}.csv")
        d["pfr_id"] = d.player_id.map(xw)
        d = d[d.pfr_id.notna() & (d[vol] >= mn)].sort_values(vol).drop_duplicates("pfr_id", keep="last")
        F, _ = feats(d)
        F["pfr_id"], F["season"] = d.pfr_id.to_numpy(), y
        rows.append(F)
    P = pd.concat(rows)
    v = pd.read_parquet("data/derived/vet_season_values.parquet").sort_values("value_ours")
    v = v.drop_duplicates(["pfr_id", "season"], keep="last")
    v = v[v.group == g][["pfr_id", "season", "share", "gfrac", "q", "value_ours"]]
    d = v.merge(P, on=["pfr_id", "season"])
    born = pd.read_parquet(fm.RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    born = dict(zip(born.pfr_id, pd.to_datetime(born.birth_date, errors="coerce")))
    d["age"] = [(pd.Timestamp(f"{s}-09-01") - born.get(p)).days / 365.25 if pd.notna(born.get(p)) else np.nan
                for p, s in zip(d.pfr_id, d.season)]
    d["age"] = d.age.fillna(d.age.median())
    d["age2"] = (d.age - 27) ** 2
    cols = m1 + m2
    d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
    d[cols] = d[cols].fillna(d[cols].median())
    # standardize so OLS terms are comparable and outliers in rates don't dominate
    d[cols] = ((d[cols] - d[cols].mean()) / d[cols].std()).clip(-4, 4).fillna(0)
    return d, {"M0": ["q"], "M1": ["q"] + m1, "M2": ["q"] + m1 + m2}


def run_group(g, ps, xw):
    S, QF = season_frame(g, xw)
    lq.Q_FEATS = QF
    C = lq.contracts(S)
    print(f"\n================ {g}: {len(S)} seasons ({S.pfr_id.nunique()} players), {len(C)} contracts ================")
    if len(C) < 60:
        print("  too few contracts")
        return None
    r2 = {}
    for m in QF:
        pred = pd.Series(index=C.index, dtype=float)
        for Y in C.Y.unique():
            pred[C.Y == Y] = smf.ols(lq.formula(m), C[C.Y != Y]).fit().predict(C[C.Y == Y])
        r2[m] = 1 - ((C.log_apy - pred) ** 2).sum() / ((C.log_apy - C.log_apy.mean()) ** 2).sum()
    fit = smf.ols(lq.formula("M2"), C).fit()
    top = fit.tvalues.drop(["Intercept"] + BASE + ["q"]).abs().sort_values(ascending=False).head(4)
    print(f"  A pay R2: M0 {r2['M0']:.3f}  M1 {r2['M1']:.3f}  M2 {r2['M2']:.3f}; strongest added terms (t): "
          + ", ".join(f"{k} {fit.tvalues[k]:+.1f}" for k in top.index))
    pers = {}
    for m in QF:
        f_ = smf.ols(lq.formula(m), C).fit()
        q = S.assign(qp=lq.quality_part(f_, S, m))[["pfr_id", "season", "qp"]]
        j = q.merge(q.assign(season=q.season - 1), on=["pfr_id", "season"], suffixes=("", "_n"))
        pers[m] = j.qp.corr(j.qp_n)
    print(f"  B persistence: M0 {pers['M0']:.2f}  M1 {pers['M1']:.2f}  M2 {pers['M2']:.2f}")
    out = {}
    for m in ["old", "M0", "M2"]:
        if m == "old":
            alt = S.assign(alt=S.value_ours)
        else:
            parts = []
            for y in S.season.unique():
                f_ = smf.ols(lq.formula(m), C[C.season != y]).fit()
                s = S[S.season == y]
                parts.append(s.assign(alt=np.exp(f_.predict(s)) * s.gfrac))
            alt = pd.concat(parts)
            alt["alt"] *= alt.groupby("season").value_ours.transform("sum") / alt.groupby("season").alt.transform("sum")
        a = alt.set_index(["pfr_id", "season"]).alt
        P = ps.copy()
        isg = (P.g == g).to_numpy()
        key = list(zip(P.pfr_player_id, P.season))
        newv = np.array([a.get(k, np.nan) for k in key])
        has = isg & ~np.isnan(newv)
        P.loc[has, "curve"] = newv[has] * P.loc[has, "w"] * 100
        full = (P.share * P.gfrac.fillna(1)).replace(0, np.nan)
        cf = (P.curve / P.w.replace(0, np.nan) / full).where(P.share >= 0.2)
        P["excess"] = np.where(isg & P.vet & cf.notna(), P.share * P.w.clip(upper=1) * (P.price - cf), P.excess)
        mdl, T = quiet(lambda: lv.regress(P, ["curve", "excess"], m))
        out[m] = (mdl.rsquared, T.loc[f"curve_{g}"], T.loc[f"excess_{g}"])
        c, e = out[m][1], out[m][2]
        print(f"  C {m:3s}: team R2 {mdl.rsquared:.3f}; {g} rate {c['pts per 1%']:.3f} ({c.lo:.3f} to {c.hi:.3f}); "
              f"{g} excess {e['pts per 1%']:+.3f} ({e.lo:+.3f} to {e.hi:+.3f})")
    gain = r2["M2"] - r2["M0"]
    e = out["M2"][2]
    passes = gain >= 0.05 and out["M2"][0] >= out["old"][0] and not e.lo > 0
    return {"group": g, "contracts": len(C), "pay R2 M0": r2["M0"], "pay R2 M2": r2["M2"], "gain": gain,
            "persist M0": pers["M0"], "persist M2": pers["M2"], "team R2 old": out["old"][0],
            "team R2 M2": out["M2"][0], "excess old": out["old"][2]["pts per 1%"],
            "excess M2": e["pts per 1%"], "passes": passes}


def quiet(f):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return f()


def main():
    pd.set_option("display.width", 220)
    groups = sys.argv[1:] or list(SPECS)
    ps, xw = lv.data(), crosswalk()
    res = [r for g in groups if (r := run_group(g, ps, xw))]
    print("\nsummary:")
    print(pd.DataFrame(res).set_index("group").round(3).to_string())


if __name__ == "__main__" and "--lag" not in sys.argv:
    main()


def lagged(groups):
    """C with quality from the season BEFORE (share, games, age from the season itself): a player's PFF rates partly
    reflect his whole defense / offense that season, which would inflate the same-season team fit. M0 and M2 both
    lagged, so the comparison is like for like."""
    ps, xw = lv.data(), crosswalk()
    print("\nC lagged (quality features from season s-1 for season s):")
    for g in groups:
        S, QF = season_frame(g, xw)
        lq.Q_FEATS = QF
        C = lq.contracts(S)
        qcols = QF["M2"]
        prev = S[["pfr_id", "season"] + qcols].assign(season=S.season + 1)
        SL = S.drop(columns=qcols).merge(prev, on=["pfr_id", "season"])
        res = {}
        for m in ["M0", "M2"]:
            parts = []
            for y in SL.season.unique():
                f_ = smf.ols(lq.formula(m), C[(C.season != y) & (C.season != y - 1)]).fit()
                s = SL[SL.season == y]
                parts.append(s.assign(alt=np.exp(f_.predict(s)) * s.gfrac))
            alt = pd.concat(parts)
            alt["alt"] *= alt.groupby("season").value_ours.transform("sum") / alt.groupby("season").alt.transform("sum")
            a = alt.set_index(["pfr_id", "season"]).alt
            P = ps.copy()
            isg = (P.g == g).to_numpy()
            newv = np.array([a.get(k, np.nan) for k in zip(P.pfr_player_id, P.season)])
            has = isg & ~np.isnan(newv)
            P.loc[has, "curve"] = newv[has] * P.loc[has, "w"] * 100
            full = (P.share * P.gfrac.fillna(1)).replace(0, np.nan)
            cf = (P.curve / P.w.replace(0, np.nan) / full).where(P.share >= 0.2)
            P["excess"] = np.where(isg & P.vet & cf.notna(), P.share * P.w.clip(upper=1) * (P.price - cf), P.excess)
            mdl, T = quiet(lambda: lv.regress(P, ["curve", "excess"], m))
            res[m] = mdl
            c = T.loc[f"curve_{g}"]
            print(f"  {g:4s} {m}: team R2 {mdl.rsquared:.4f}; rate {c['pts per 1%']:.3f} ({c.lo:.3f} to {c.hi:.3f}); "
                  f"seasons replaced {int(has.sum())}")
        print(f"  {g:4s} M2 - M0 team R2 {res['M2'].rsquared - res['M0'].rsquared:+.4f}")


if __name__ == "__main__" and "--lag" in sys.argv:
    lagged([a for a in sys.argv[1:] if a != "--lag"] or list(SPECS))
