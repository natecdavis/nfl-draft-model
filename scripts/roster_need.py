"""Does the team's existing roster at a position predict how much a rookie plays and delivers?

Roster need, as known on draft day (drafts 2014-2025; weekly snap counts from 2013):
  last season's snaps at the position (starter-equivalents = sum of snap % / games), counting
  only players still on the team on draft day, plus veterans added before the draft.
    left      = signed with another team that year, or traded away (Aug of the prior season
                through draft day). Release / retirement without a new deal isn't detectable,
                so those players still count as returning (adds noise, biases toward no effect).
    added     = signed with this team that year (not a rookie of this draft), or traded for
                before the draft: their prior-season starter-equivalents at the position.
  open spots = typical starter-equivalents at the position (league median) - (returning + added).
  Note: contracts carry only the year signed, so a few signings after the draft leak in.
Position of a roster player: from his draft record (same mapping as the rookies), else his
snap-count position.

Outcomes for each drafted rookie (his drafting team): year-1 snap share, years 1-2 snap share,
cfdarko season values y1, y2, y3-4 and y1-4 (in % of cap, through the 2022 class for y3-4).
Controls: position group x log(pick), position group x log(consensus rank), off-board flag.
Standard errors clustered by team-draft.

Run:  python scripts/roster_need.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "src")
from outlook import RAW, OUT  # noqa: E402
from review import DRAFT_POS, DRAFT_START, PFR_TEAM  # noqa: E402

team = lambda s: PFR_TEAM.get(s, s)
SNAP_POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "T", "OT": "T", "G/T": "T",
            "T/G": "T", "G/OT": "T", "G": "G", "OG": "G", "OL": "G", "G/C": "G", "C": "C", "C/G": "C", "DE": "ED",
            "OLB": "ED", "DE/L": "ED", "DT": "IDL", "NT": "IDL", "DL": "IDL", "DT/D": "IDL", "LB": "LB", "ILB": "LB",
            "MLB": "LB", "CB": "CB", "DB": "CB", "FS": "S", "SS": "S", "S": "S"}
DEF = {"ED", "IDL", "LB", "CB", "S"}
FAMILY = {"T": "OL", "G": "OL", "C": "OL", "CB": "DB", "S": "DB"}
GROUPS = ["RB", "WR", "TE", "T", "G", "C", "ED", "IDL", "LB", "CB", "S"]  # QB handled by the QB rules


def load():
    s = pd.read_parquet(RAW / "snap_counts.parquet")
    s = s[s.game_type == "REG"].assign(team=lambda d: d.team.map(team))
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dpos = dict(zip(dp.pfr_player_id, dp.position.map(DRAFT_POS)))
    s["grp"] = [dpos.get(p) or SNAP_POS.get(q) for p, q in zip(s.pfr_player_id, s.position)]
    s["pct"] = np.where(s.grp.isin(DEF), s.defense_pct, s.offense_pct).astype(float)
    s["pct"] = s.pct / (100 if s.pct.max() > 1.5 else 1)
    games = s.groupby(["season", "team"]).game_id.nunique()
    # one row per player-season-team: starter-equivalents and his main group
    ps = (s.groupby(["season", "team", "pfr_player_id"])
          .agg(pct=("pct", "sum"), grp=("grp", lambda x: x.mode().iloc[0] if x.notna().any() else None)).reset_index())
    ps["se"] = ps.pct / ps.set_index(["season", "team"]).index.map(games)
    return s, ps, games, dp


def departures_and_adds(year):
    """pfr ids that left each team before the draft, and pfr ids each team added."""
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    c = pd.read_parquet(RAW / "contracts.parquet")
    nick = dict(zip(*pd.read_parquet(RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    c["tm"] = c.team.map(lambda x: team(nick.get(str(x).split("/")[-1], str(x).split("/")[-1])))
    c["pfr"] = c.gsis_id.map(g2p)
    cy = c[(c.year_signed == year) & c.pfr.notna() & (c.draft_year != year)]
    signed = cy.groupby("pfr").tm.agg(set).to_dict()  # player -> teams he signed with this year
    t = pd.read_parquet(RAW / "trades.parquet")
    t["date"] = pd.to_datetime(t.trade_date)
    t = t[(t.date > pd.Timestamp(f"{year - 1}-08-01")) & (t.date < pd.Timestamp(DRAFT_START[year])) & t.pfr_id.notna()]
    traded_away = {(p, team(g)) for p, g in zip(t.pfr_id, t.gave)}
    traded_in = {(p, team(r)) for p, r in zip(t.pfr_id, t.received)}
    return signed, traded_away, traded_in


def roster_need(ps, year):
    prev = ps[(ps.season == year - 1) & ps.grp.isin(GROUPS)]
    signed, away, tin = departures_and_adds(year)
    left = [(p, tm) in away or (p in signed and tm not in signed[p]) for p, tm in zip(prev.pfr_player_id, prev.team)]
    ret = prev[~np.array(left, bool)].groupby(["team", "grp"]).se.sum()
    # additions: their prior-season starter-equivalents (any team), by group
    se_any = prev.groupby(["pfr_player_id", "grp"]).se.sum().reset_index()
    adds = []
    for p, g, se in se_any.itertuples(index=False):
        for tm in signed.get(p, set()) | {t_ for (pp, t_) in tin if pp == p}:
            played_for = set(prev[prev.pfr_player_id == p].team)
            if tm not in played_for:
                adds.append((tm, g, se))
    add = pd.DataFrame(adds, columns=["team", "grp", "se"]).groupby(["team", "grp"]).se.sum()
    total = prev.groupby(["team", "grp"]).se.sum()
    idx = pd.MultiIndex.from_product([sorted(prev.team.unique()), GROUPS], names=["team", "grp"])
    out = pd.DataFrame({"total_prev": total.reindex(idx, fill_value=0), "returning": ret.reindex(idx, fill_value=0),
                        "added": add.reindex(idx, fill_value=0)}).reset_index()
    out["year"] = year
    return out


def main():
    s, ps, games, dp = load()
    need = pd.concat([roster_need(ps, y) for y in range(2014, 2026)], ignore_index=True)
    typical = need.groupby("grp").total_prev.median()
    need["have"] = need.returning + need.added
    need["open"] = need.grp.map(typical) - need.have
    need["fam"] = need.grp.map(lambda g: FAMILY.get(g, g))
    fam = need.groupby(["year", "team", "fam"])[["have"]].sum()
    fam_typ = need.groupby("fam").apply(lambda d: d.groupby(["year", "team"]).total_prev.sum().median(), include_groups=False)
    fam["open_fam"] = fam.index.get_level_values("fam").map(fam_typ) - fam.have
    print("Typical starter-equivalents by group (league median of last season's snaps / games):")
    print(typical.round(2).to_string())
    print("\nOpen starting spots on draft day, distribution by group (p10 / median / p90):")
    print(need.groupby("grp").open.quantile([0.1, 0.5, 0.9]).unstack().round(2).to_string())

    # rookies
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[R.year.between(2014, 2025) & R.grp.isin(GROUPS)].copy()
    pid = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    R["pfr"] = [pid.get((y, k)) for y, k in zip(R.year, R.pick)]
    me = ps.set_index(["season", "team", "pfr_player_id"]).se
    snap = lambda y, tm, p: float(me.get((y, tm, p), 0.0))
    R["snap1"] = [snap(y, tm, p) for y, tm, p in zip(R.year, R.team, R.pfr)]
    R["snap2"] = [snap(y + 1, tm, p) if y + 1 <= 2025 else np.nan for y, tm, p in zip(R.year, R.team, R.pfr)]
    R["snap12"] = (R.snap1 + R.snap2) / 2
    L = pd.read_parquet(Path.home() / "cfdarko/data/processed/rookie_value_labels.parquet").dropna(subset=["pfr_id"])
    L = L.drop_duplicates("pfr_id").set_index("pfr_id")
    for t in range(1, 5):
        R[f"v{t}"] = R.pfr.map(L[f"value_y{t}"].clip(lower=0)) * 100
    R.loc[R.year > 2024, "v2"] = np.nan
    R.loc[R.year > 2023, "v3"] = np.nan
    R.loc[R.year > 2022, "v4"] = np.nan
    R["v34"] = R.v3 + R.v4
    R["v14"] = R.v1 + R.v2 + R.v3 + R.v4
    R = R.merge(need[["year", "team", "grp", "open", "returning", "added"]], on=["year", "team", "grp"], how="left")
    R["fam"] = R.grp.map(lambda g: FAMILY.get(g, g))
    R = R.merge(fam.reset_index()[["year", "team", "fam", "open_fam"]], on=["year", "team", "fam"], how="left")
    R["lp"] = np.log(R.pick + 10)
    R["off"] = R.board_rank.isna().astype(int)
    R["lr"] = np.log(R.board_rank.fillna(330) + 10)
    R["cl"] = R.year.astype(str) + R.team
    R["open_c"] = R.open.clip(-1.5, 1.5)
    R["open_fam_c"] = R.open_fam.clip(-2, 2)
    R["open_any"] = (R.open >= 0.5).astype(int)
    R.to_parquet(OUT / "roster_need_rookies.parquet")
    print(f"\nRookies: {len(R)} (2014-2025 drafts, non-QB, non-specialist); {R.v14.notna().sum()} with 4 seasons of values")

    ctrl = "C(grp) + C(grp):lp + C(grp):lr + off"
    rows = []
    for out, lab in [("snap1", "Year-1 snap share"), ("snap12", "Years 1-2 snap share"), ("v1", "Value y1 (% cap)"),
                     ("v2", "Value y2"), ("v34", "Value y3+y4"), ("v14", "Value y1-4")]:
        for x, xl in [("open_c", "open spots at group (+1 spot)"), ("open_fam_c", "open spots at family (+1 spot)"),
                      ("open_any", "group has >= 0.5 open (yes vs no)")]:
            d = R.dropna(subset=[out, x])
            m = smf.ols(f"{out} ~ {x} + {ctrl}", d).fit(cov_type="cluster", cov_kwds={"groups": d.cl})
            lo, hi = m.conf_int().loc[x]
            rows.append({"outcome": lab, "need measure": xl, "n": len(d), "effect": m.params[x], "lo95": lo, "hi95": hi,
                         "p": m.pvalues[x], "outcome mean": d[out].mean()})
    T = pd.DataFrame(rows)
    print("\nEffect of roster need, controlling for position x pick and position x consensus rank:")
    print(T.round(3).to_string(index=False))
    T.to_csv(OUT / "roster_need_effects.csv", index=False)

    # By round, and the model's assumption for comparison
    print("\nYear-1 snap share by round and open spots at the group (raw means):")
    R["rnd"] = pd.cut(R["round"], [0, 1, 3, 7], labels=["R1", "R2-3", "R4-7"])
    R["open_bin"] = pd.cut(R.open, [-9, -0.25, 0.5, 9], labels=["covered (<= -0.25)", "about full", "open (>= 0.5)"])
    print(R.pivot_table(index="rnd", columns="open_bin", values="snap1", aggfunc=["mean", "size"], observed=True).round(2).to_string())
    print("\nModel's assumption (draftday.NEED_EARLY): years 1-2 playing time x1.0 starter / x0.6 rotation / x0.3 depth")
    by = []
    for rn, d in R.groupby("rnd", observed=True):
        d = d.dropna(subset=["snap12"])
        m = smf.ols(f"snap12 ~ open_c + {ctrl}", d).fit(cov_type="cluster", cov_kwds={"groups": d.cl})
        lo, hi = m.conf_int().loc["open_c"]
        by.append({"round": rn, "n": len(d), "+1 open spot -> yrs 1-2 snap share": m.params["open_c"], "lo95": lo, "hi95": hi,
                   "mean snap share": d.snap12.mean()})
    print(pd.DataFrame(by).round(3).to_string(index=False))
    by = []
    for g, d in R.groupby("grp"):
        d = d.dropna(subset=["snap12"])
        m = smf.ols("snap12 ~ open_c + lp + lr + off", d).fit(cov_type="cluster", cov_kwds={"groups": d.cl})
        lo, hi = m.conf_int().loc["open_c"]
        by.append({"grp": g, "n": len(d), "+1 open spot -> yrs 1-2 snap share": m.params["open_c"], "lo95": lo, "hi95": hi})
    print(pd.DataFrame(by).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
