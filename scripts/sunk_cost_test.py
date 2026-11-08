"""Sunk-cost playing time: do higher picks get more snaps than their play earns? (Keefer 2015 J Labor Res,
2017 J Sports Econ.) Matters here because the outcome labels (rookie_value_labels, value_y1..y4) are priced
on snap share AND PFF grade: extra snaps handed to high picks would show up as draft-slot "value".

Data: every drafted player 2006-22 (data/raw/draft_picks.parquet), linked to PFF NFL season files
(~/cfdarko/data/raw/pff_nfl/<season>/offense_blocking.csv, defense_summary.csv) through nflverse pff_id.
Seasons 1-4 of each career (through 2025). Snap share = player's snaps on his side of the ball / median team
snaps that season (0 if he didn't play). Quality = PFF overall grade (offense or defense) in seasons with 200+ snaps.

Pre-specified (before running):
  A. Round-boundary discontinuity (Keefer 2017). Running variable = pick minus the last pick of the earlier
     round; local linear on each side, |r| <= 16, position + class fixed effects + drafting team's prior-season
     win %, HC1. Outcomes: mean snap share years 1-4 (primary), mean grade (quality), label value y1-4
     (classes 2012-22). Boundaries 1/2, 2/3, 3/4. Draft order makes every boundary compare the league's best
     teams (end of a round) with its worst (start of the next), so the comparison of interest is the 1/2 jump
     (first-round contract: bigger guarantee, fifth-year option) MINUS the average 2/3 and 3/4 jumps.
     Prediction: snap jump > 0 at 1/2 beyond the other boundaries, no matching grade jump; larger for the
     2006-10 classes (no rookie wage scale, much larger first-round guarantees).
  B. Performance feedback (Keefer 2015). Player-seasons in years 2-4 with 200+ snaps the season before:
     snap share_t on log(pick + 10), controlling share_{t-1}, grade_{t-1}, position, career year, season.
     Then the same model for grade_t. If pick predicts next season's snaps but not next season's grade (given
     the same past play), teams are playing high picks for being high picks. Prediction: snaps > 0, grade ~0.
Bandwidth checks: |r| <= 10 and <= 24.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
import prospect_model as pm
from outlook import OUT

PFF = os.path.expanduser("~/cfdarko/data/raw/pff_nfl/")
LAST = 2025
MIN_Q = 200
TEAM = {"GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV", "NOR": "NO", "NWE": "NE", "SDG": "SD", "SFO": "SF",
        "TAM": "TB"}


def pff_seasons():
    rows = []
    for y in range(2006, LAST + 1):
        o = pd.read_csv(PFF + f"{y}/offense_blocking.csv", usecols=["player_id", "snap_counts_offense", "grades_offense"])
        d = pd.read_csv(PFF + f"{y}/defense_summary.csv", usecols=["player_id", "snap_counts_defense", "grades_defense"])
        o = o.rename(columns={"snap_counts_offense": "snaps", "grades_offense": "grade"}).assign(side="O")
        d = d.rename(columns={"snap_counts_defense": "snaps", "grades_defense": "grade"}).assign(side="D")
        for x in (o, d):
            x = x.groupby("player_id", as_index=False).agg(snaps=("snaps", "sum"), grade=("grade", "mean"),
                                                            side=("side", "first"))
            # league-typical full season on this side: median over teams of the most snaps by one player
            rows.append(x.assign(season=y))
    s = pd.concat(rows)
    full = {}
    for y in range(2006, LAST + 1):
        for side, f, c in [("O", "offense_blocking", "snap_counts_offense"), ("D", "defense_summary", "snap_counts_defense")]:
            t = pd.read_csv(PFF + f"{y}/{f}.csv", usecols=["team_name", c])
            t = t[~t.team_name.astype(str).str.contains(",")]
            full[(y, side)] = t.groupby("team_name")[c].max().median()
    s["full"] = [full[(y, sd)] for y, sd in zip(s.season, s.side)]
    s["share"] = (s.snaps / s.full).clip(upper=1.0)
    # one row per player-season: the side he played most
    return s.sort_values("snaps").drop_duplicates(["player_id", "season"], keep="last")


def win_pct():
    g = pd.read_parquet("data/raw/schedules.parquet")
    g = g[(g.game_type == "REG") & g.result.notna()]
    h = pd.DataFrame({"season": g.season, "team": g.home_team, "w": (g.result > 0) + 0.5 * (g.result == 0)})
    a = pd.DataFrame({"season": g.season, "team": g.away_team, "w": (g.result < 0) + 0.5 * (g.result == 0)})
    return pd.concat([h, a]).groupby(["season", "team"]).w.mean().rename("win_prev").reset_index()


def build():
    d = pd.read_parquet("data/raw/draft_picks.parquet")
    d = d[d.season.between(2006, 2022)][["season", "round", "pick", "team", "pfr_player_id", "position", "side"]]
    d = d.rename(columns={"season": "cls", "pfr_player_id": "pfr_id", "side": "dside"})
    p = pd.read_parquet("data/raw/players.parquet", columns=["pfr_id", "pff_id"]).dropna().drop_duplicates("pfr_id")
    d = d.merge(p, on="pfr_id", how="left")
    d["pff_id"] = pd.to_numeric(d.pff_id, errors="coerce")
    d = d[d.dside.isin(["O", "D"])]  # drop specialists
    w = win_pct()
    d["tm"] = d.team.map(TEAM).fillna(d.team)
    d = d.merge(w.assign(cls=w.season + 1).drop(columns="season").rename(columns={"team": "tm"}), on=["cls", "tm"], how="left")
    ends = d.groupby(["cls", "round"]).pick.max()
    for b in (1, 2, 3):
        d[f"r{b}"] = d.pick - d.cls.map(ends.xs(b, level="round"))

    s = pff_seasons()
    panel = []
    for t in range(1, 5):
        x = d[["pfr_id", "pff_id", "cls"]].assign(t=t, season=d.cls + t - 1)
        panel.append(x[x.season <= LAST])
    P = pd.concat(panel).merge(s[["player_id", "season", "snaps", "grade", "share"]],
                               left_on=["pff_id", "season"], right_on=["player_id", "season"], how="left")
    P["snaps"] = P.snaps.fillna(0)
    P["share"] = P.share.fillna(0)
    P.loc[P.snaps < MIN_Q, "grade"] = np.nan
    agg = P.groupby("pfr_id").agg(share14=("share", "mean"), grade14=("grade", "mean"), n_seasons=("t", "size"))
    d = d.merge(agg, on="pfr_id", how="left")
    d = d[d.n_seasons == 4]  # full four seasons observed (classes through 2022)
    uni = pm.load()
    lab = uni[uni.drafted.astype(bool)][["pfr_id", "cls", "y1", "y2", "y3", "y4"]].dropna()
    lab["label14"] = lab[["y1", "y2", "y3", "y4"]].sum(axis=1)
    d = d.merge(lab[["pfr_id", "cls", "label14"]].drop_duplicates("pfr_id"), on=["pfr_id", "cls"], how="left")
    return d, P


def rd(d, b, y, h, era=None):
    x = d[d[f"r{b}"].abs() <= h].dropna(subset=[y, "win_prev"]).copy()
    if era == "pre":
        x = x[x.cls <= 2010]
    elif era == "post":
        x = x[x.cls >= 2011]
    x["D"] = (x[f"r{b}"] <= 0).astype(float)  # earlier round
    x["r"] = x[f"r{b}"].astype(float)
    x["Dr"] = x.D * x.r
    fe = pd.concat([pd.get_dummies(x.position, prefix="p", drop_first=True, dtype=float),
                    pd.get_dummies(x.cls, prefix="c", drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(x[y], sm.add_constant(pd.concat([x[["D", "r", "Dr", "win_prev"]], fe], axis=1))).fit(cov_type="HC1")
    return m.params.D, m.bse.D, len(x), x.loc[x.D == 0, y].mean()


def part_a(d):
    print("A. ROUND-BOUNDARY JUMPS (earlier round minus later round, at the boundary)")
    for y, nm, fmt in [("share14", "snap share, yrs 1-4", "{:+.3f}"), ("grade14", "PFF grade, 200+ snap seasons", "{:+.2f}"),
                       ("label14", "label value y1-4 (% cap), 2012-22", "{:+.4f}")]:
        for era in (None, "pre", "post"):
            if y == "label14" and era == "pre":
                continue
            for h in (16, 10, 24):
                if h != 16 and era is not None:
                    continue
                res = {b: rd(d, b, y, h, era) for b in (1, 2, 3)}
                other = np.mean([res[2][0], res[3][0]])
                dd = res[1][0] - other
                se = np.sqrt(res[1][1] ** 2 + (res[2][1] ** 2 + res[3][1] ** 2) / 4)
                tag = {None: "2006-22", "pre": "2006-10", "post": "2011-22"}[era]
                print(f"  {nm:34s} {tag} |r|<={h:2d}: " +
                      "  ".join(f"{b}/{b + 1} " + fmt.format(res[b][0]) + f" (se {res[b][1]:.3f}, n {res[b][2]})" for b in (1, 2, 3)) +
                      f"  | 1/2 minus others " + fmt.format(dd) + f" (z {dd / se:+.2f})" +
                      f"  [later-round mean at 1/2: {res[1][3]:.3f}]")


def part_b(d, P):
    print("\nB. PERFORMANCE FEEDBACK: years 2-4, same snaps and grade the season before")
    P = P.merge(d[["pfr_id", "pick", "position", "round"]], on="pfr_id")
    P = P.sort_values(["pfr_id", "t"])
    P["share_prev"] = P.groupby("pfr_id").share.shift()
    P["grade_prev"] = P.groupby("pfr_id").grade.shift()
    P["snaps_prev"] = P.groupby("pfr_id").snaps.shift()
    Q = P[(P.t >= 2) & (P.snaps_prev >= MIN_Q) & P.grade_prev.notna()].copy()
    Q["lpick"] = np.log(Q.pick + 10)
    Q["rd1"] = (Q["round"] == 1).astype(float)
    fe = pd.concat([pd.get_dummies(Q.position, prefix="p", drop_first=True, dtype=float),
                    pd.get_dummies(Q.t, prefix="t", drop_first=True, dtype=float),
                    pd.get_dummies(Q.season, prefix="s", drop_first=True, dtype=float)], axis=1)
    for y, sub, nm in [("share", Q, "snap share next season"),
                       ("grade", Q[Q.grade.notna()], "PFF grade next season (200+ snaps)")]:
        for xs in (["lpick"], ["lpick", "rd1"]):
            X = sm.add_constant(pd.concat([sub[xs + ["share_prev", "grade_prev"]], fe.loc[sub.index]], axis=1))
            m = sm.OLS(sub[y], X).fit(cov_type="cluster", cov_kwds={"groups": sub.pfr_id})
            sd = sub[y].std()
            print(f"  {nm:36s} n={len(sub):5d}  " + "  ".join(
                f"{x} {m.params[x]:+.4f} (se {m.bse[x]:.4f}, {m.params[x] / sd:+.3f} sd)" for x in xs))
    # Keefer-style check (added after B): same-season grade as the quality control, seasons with 200+ snaps in
    # both years (conditioning on playing, so this bounds the effect among regulars)
    sub = Q[Q.grade.notna()]
    for xs in (["lpick"], ["lpick", "rd1"]):
        X = sm.add_constant(pd.concat([sub[xs + ["grade", "share_prev", "grade_prev"]], fe.loc[sub.index]], axis=1))
        m = sm.OLS(sub.share, X).fit(cov_type="cluster", cov_kwds={"groups": sub.pfr_id})
        print(f"  {'snap share, + same-season grade':36s} n={len(sub):5d}  " + "  ".join(
            f"{x} {m.params[x]:+.4f} (se {m.bse[x]:.4f})" for x in xs) + f"  grade {m.params.grade:+.4f}/pt")
    for era, s2 in [("2006-10", sub[sub.cls <= 2010]), ("2011-22", sub[sub.cls >= 2011])]:
        X = sm.add_constant(pd.concat([s2[["lpick", "rd1", "grade", "share_prev", "grade_prev"]], fe.loc[s2.index]], axis=1))
        m = sm.OLS(s2.share, X).fit(cov_type="cluster", cov_kwds={"groups": s2.pfr_id})
        print(f"    {era}: n={len(s2):5d}  lpick {m.params.lpick:+.4f} (se {m.bse.lpick:.4f})  rd1 {m.params.rd1:+.4f} "
              f"(se {m.bse.rd1:.4f})")
    return Q


def main():
    d, P = build()
    print(f"drafted players with 4 seasons observed: {len(d)}; linked to PFF: {d.pff_id.notna().mean():.0%}; "
          f"with label: {d.label14.notna().sum()}\n")
    part_a(d)
    part_b(d, P)
    d.to_parquet(OUT / "sunk_cost.parquet")


if __name__ == "__main__":
    main()
