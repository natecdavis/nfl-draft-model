"""Which veteran snaps can a rookie displace? Incumbent quality, pay, draft capital and recency.

Incumbents on draft day (scripts/roster_need.py rules): players at the rookie's group with snaps for the team last
season and not gone before the draft, plus veterans added before the draft; size = last season's
starter-equivalents (se: sum of snap % / games, regular season). Attributes known on draft day:
  quality  DARKO-for-PFF talent after last season (talent_post, data/derived/darko_pff.parquet), percentile within
           the group among players with >= 300 snaps that season; terciles, plus "unknown"
  pay      cap % this season (OTC season history; drafted players' rookie deals included); paid = >= 3% of cap
  capital  own recent high pick: drafted by this team in rounds 1-2 within the last 3 drafts
  age      age on Sep 1
Drafts 2014-2024, groups RB..S (QB has its own rules).

A  Rookie side: rookie's year-1 (and years 1-2) se at his team ~ group x log(pick) + group x log(rank) + off-board
   + incumbent se split by type. Coefficient = rookie snaps per incumbent starter-equivalent of that type.
B  Incumbent side: incumbent's se this season at the team minus last season ~ his attributes + rookie arrival
   (sum over this draft's rookies at his group of their predicted year-1 se from A's controls) x type.
   Coefficient on arrival x type = share of a rookie's snaps taken from that kind of incumbent.
SEs clustered by team-draft.

Run:  python scripts/displacement.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import roster_need as rn  # noqa: E402
from outlook import OUT, RAW  # noqa: E402

YEARS = range(2014, 2025)
PAID = 0.03


def attributes():
    pl = pd.read_parquet(RAW / "players.parquet")
    pl = pl[pl.pfr_id.notna()]
    p2g = dict(zip(pl.pfr_id, pl.gsis_id))
    p2pff = {p: int(f) for p, f in zip(pl.pfr_id, pl.pff_id) if pd.notna(f)}
    born = dict(zip(pl.pfr_id, pd.to_datetime(pl.birth_date, errors="coerce")))
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    drafted = {p: (int(s), int(r), rn.team(t)) for p, s, r, t in zip(dp.pfr_player_id, dp.season, dp["round"], dp.team)}
    # quality percentile within group
    t = pd.read_parquet(OUT / "darko_pff.parquet")[["player_id", "season", "g", "snaps", "talent_post"]]
    ref = t[t.snaps >= 300]
    t["q"] = np.nan
    for (s, g), d in t.groupby(["season", "g"]):
        r = np.sort(ref[(ref.season == s) & (ref.g == g)].talent_post.to_numpy())
        if len(r):
            t.loc[d.index, "q"] = np.searchsorted(r, d.talent_post.to_numpy(), side="right") / len(r)
    q = t.sort_values("snaps").drop_duplicates(["player_id", "season"], keep="last").set_index(["player_id", "season"]).q
    # cap % by season
    c = pd.read_parquet(RAW / "contracts.parquet")
    cap = {}
    for gs, hist in zip(c.gsis_id, c.season_history):
        if gs is None or hist is None:
            continue
        for h in hist:
            try:
                y, v = int(h["year"]), h.get("cap_percent")
            except (TypeError, ValueError):
                continue
            if v is not None and not np.isnan(v):
                cap[(gs, y)] = max(cap.get((gs, y), 0.0), float(v))
    return p2g, p2pff, born, drafted, q, cap


def incumbents(ps, year, A):
    p2g, p2pff, born, drafted, q, cap = A
    prev = ps[(ps.season == year - 1) & ps.grp.isin(rn.GROUPS)]
    signed, away, tin = rn.departures_and_adds(year)
    left = np.array([(p, tm) in away or (p in signed and tm not in signed[p]) for p, tm in zip(prev.pfr_player_id, prev.team)], bool)
    ret = prev[~left][["team", "grp", "pfr_player_id", "se"]]
    se_any = prev.groupby(["pfr_player_id", "grp"]).se.sum().reset_index()
    teams_of = prev.groupby("pfr_player_id").team.agg(set).to_dict()
    adds = []
    for p, g, se in se_any.itertuples(index=False):
        for tm in signed.get(p, set()) | {t_ for (pp, t_) in tin if pp == p}:
            if tm not in teams_of.get(p, set()):
                adds.append((tm, g, p, se))
    I = pd.concat([ret, pd.DataFrame(adds, columns=["team", "grp", "pfr_player_id", "se"])], ignore_index=True)
    I = I.groupby(["team", "grp", "pfr_player_id"], as_index=False).se.sum()
    I["year"] = year
    I["q"] = [float(np.ravel(q.get((p2pff.get(p), year - 1), np.nan))[-1]) for p in I.pfr_player_id]
    I["cap"] = [cap.get((p2g.get(p), year), np.nan) for p in I.pfr_player_id]
    d = [drafted.get(p) for p in I.pfr_player_id]
    I["own_high"] = [bool(x and x[2] == tm and x[1] <= 2 and year - x[0] <= 3) for x, tm in zip(d, I.team)]
    I["age"] = [(pd.Timestamp(f"{year}-09-01") - born.get(p)).days / 365.25 if pd.notna(born.get(p)) else np.nan
                for p in I.pfr_player_id]
    return I


def missed(ps, year, A, grp_of=None):
    """Draft-day roster players with no snaps last season (any team), who incumbents() can't see (room check,
    Nov 16 2026): out1 = "missed": snaps for the team two seasons ago, or signed by the team last season, and none
    anywhere last season (injured all year: Derwin James, Chargers 2021; sat out: Deshaun Watson, Browns 2022);
    also players traded for or signed this offseason with no snaps last season;
    "red12" / "red3": drafted by the team last season in rounds 1-2 / 3+ and no snaps (J.J. McCarthy, Vikings 2025).
    Gone if he signed with another team (roster_need.signed_by_draft) or was traded away since. se = 0; group from
    two seasons ago, else grp_of (pfr -> group), else his draft position. Same columns as incumbents(), plus out1."""
    p2g, p2pff, born, drafted, q, cap = A
    played = set(ps[ps.season == year - 1].pfr_player_id)
    two = ps[(ps.season == year - 2) & ps.grp.isin(rn.GROUPS)]
    s1, s0 = rn.signed_by_draft(year - 1), rn.signed_by_draft(year)
    _, away0, _ = rn.departures_and_adds(year)
    _, away1, _ = rn.departures_and_adds(year - 1) if (year - 1) in rn.DRAFT_START else ({}, set(), set())
    cand = {}
    for p, tm, g in zip(two.pfr_player_id, two.team, two.grp):
        cand[(p, tm)] = ("missed", g)
    g2 = dict(zip(two.pfr_player_id, two.grp))
    for p, ts in s1.items():
        for tm in ts:
            cand.setdefault((p, tm), ("missed", (grp_of or {}).get(p) or g2.get(p)))
    _, _, tin0 = rn.departures_and_adds(year)
    for p, tm in set(tin0) | {(p, tm) for p, ts in s0.items() for tm in ts}:
        cand.setdefault((p, tm), ("missed", (grp_of or {}).get(p) or g2.get(p)))  # added this offseason
    for p, (y, r, tm) in drafted.items():
        if y == year - 1:
            cand[(p, tm)] = ("red12" if r <= 2 else "red3", (grp_of or {}).get(p))
    rows = []
    for (p, tm), (k, g) in cand.items():
        if p in played or g is None or g not in rn.GROUPS:
            continue
        if any(x != tm for x in s0.get(p, ())) or (p, tm) in away0 or (p, tm) in away1:
            continue  # signed elsewhere before this draft, or traded away
        if s1.get(p) and tm not in s1[p] and tm not in s0.get(p, ()) and (p, tm) not in tin0:
            continue  # signed elsewhere last season
        rows.append((tm, g, p, 0.0, k))
    I = pd.DataFrame(rows, columns=["team", "grp", "pfr_player_id", "se", "out1"]).drop_duplicates(["team", "pfr_player_id"])
    I["year"] = year
    I["q"] = [float(np.ravel(q.get((p2pff.get(p), year - 2), np.nan))[-1]) if k == "missed" else np.nan
              for p, k in zip(I.pfr_player_id, I.out1)]
    I["cap"] = [cap.get((p2g.get(p), year), np.nan) for p in I.pfr_player_id]
    d = [drafted.get(p) for p in I.pfr_player_id]
    I["own_high"] = [bool(x and x[2] == tm and x[1] <= 2 and year - x[0] <= 3) for x, tm in zip(d, I.team)]
    I["age"] = [(pd.Timestamp(f"{year}-09-01") - born.get(p)).days / 365.25 if pd.notna(born.get(p)) else np.nan
                for p in I.pfr_player_id]
    return I


def types(I):
    I["qt"] = np.select([I.q.isna(), I.q >= 2 / 3, I.q >= 1 / 3], ["unk", "high", "mid"], "low")
    I["paid"] = (I.cap.fillna(0) >= PAID)
    return I


def main():
    s, ps, games, dp = rn.load()
    A = attributes()
    I = types(pd.concat([incumbents(ps, y, A) for y in YEARS], ignore_index=True))
    print(f"Incumbents {len(I)}; quality known for {I.q.notna().mean() * 100:.0f}% of rows, "
          f"{(I.se * I.q.notna()).sum() / I.se.sum() * 100:.0f}% of incumbent snaps; pay known {I.cap.notna().mean() * 100:.0f}%")

    # rookies (drafted, review picks)
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[R.year.isin(YEARS) & R.grp.isin(rn.GROUPS)].copy()
    pid = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    R["pfr"] = [pid.get((y, k)) for y, k in zip(R.year, R.pick)]
    me = ps.set_index(["season", "team", "pfr_player_id"]).se
    R["s1"] = [float(me.get((y, tm, p), 0.0)) for y, tm, p in zip(R.year, R.team, R.pfr)]
    R["s2"] = [float(me.get((y + 1, tm, p), 0.0)) for y, tm, p in zip(R.year, R.team, R.pfr)]
    R["s12"] = (R.s1 + R.s2) / 2
    R["lp"], R["lr"] = np.log(R.pick + 10), np.log(R.board_rank.fillna(330) + 10)
    R["off"] = R.board_rank.isna().astype(int)
    R["cl"] = R.year.astype(str) + R.team
    # rookies taken earlier in this draft at the group (control)
    R = R.sort_values("pick")
    R["earlier"] = R.groupby(["year", "team", "grp"]).cumcount()

    splits = {
        "total": lambda d: pd.Series({"inc": d.se.sum()}),
        "quality": lambda d: d.groupby("qt").se.sum().reindex(["high", "mid", "low", "unk"], fill_value=0).add_prefix("q_"),
        "pay": lambda d: pd.Series({"paid": d.se[d.paid].sum(), "unpaid": d.se[~d.paid].sum()}),
        "capital": lambda d: pd.Series({"own_high": d.se[d.own_high].sum(), "other": d.se[~d.own_high].sum()}),
        "mixed": lambda d: pd.Series({
            "hq_paid": d.se[(d.qt == "high") & d.paid].sum(), "hq_cheap": d.se[(d.qt == "high") & ~d.paid].sum(),
            "own_high_nothq": d.se[d.own_high & (d.qt != "high")].sum(),
            "lowmid_paid": d.se[d.qt.isin(["low", "mid"]) & d.paid & ~d.own_high].sum(),
            "lowmid_cheap": d.se[d.qt.isin(["low", "mid"]) & ~d.paid & ~d.own_high].sum(),
            "unk": d.se[(d.qt == "unk") & ~d.own_high].sum()}),
    }
    base = "C(grp):lp + C(grp):lr + off + C(grp) + earlier"
    print("\nA. Rookie year-1 / years 1-2 snaps (starter-equivalents) per incumbent starter-equivalent of each type")
    print("   (negative = that kind of incumbent holds the rookie down; drafts 2014-24 year 1, 2014-23 years 1-2)")
    for name, fn in splits.items():
        X = I.groupby(["year", "team", "grp"]).apply(fn, include_groups=False).reset_index()
        D = R.merge(X, on=["year", "team", "grp"], how="left").fillna({c: 0 for c in X.columns[3:]})
        cols = list(X.columns[3:])
        for y_, lab, sub in [("s1", "yr1", D), ("s12", "yr1-2", D[D.year <= 2023])]:
            m = smf.ols(f"{y_} ~ {base} + " + " + ".join(cols), sub).fit(cov_type="cluster", cov_kwds={"groups": sub.cl})
            print(f"  {name:8s} {lab:6s} " + "  ".join(f"{c} {m.params[c]:+.3f} ({m.bse[c]:.3f})" for c in cols))

    # B: incumbent side
    mA = smf.ols(f"s1 ~ {base}", R).fit()
    R["pred1"] = mA.predict(R).clip(lower=0)
    arr = R.groupby(["year", "team", "grp"]).pred1.sum().rename("arrive")
    I = I.join(arr, on=["year", "team", "grp"]).fillna({"arrive": 0})
    I["now"] = [float(me.get((y, tm, p), 0.0)) for y, tm, p in zip(I.year, I.team, I.pfr_player_id)]
    I["d"] = I.now - I.se
    I["cl"] = I.year.astype(str) + I.team
    I["age_"] = I.age.fillna(I.age.median())
    I["age2"] = I.age_ ** 2
    I["typ"] = np.where(I.own_high & (I.qt != "high"), "own_high", np.where(I.qt == "high", np.where(I.paid, "hq_paid", "hq_cheap"),
                        np.where(I.qt == "unk", "unk", np.where(I.paid, "lowmid_paid", "lowmid_cheap"))))
    m = smf.ols("d ~ C(grp) + se + C(typ) + age_ + age2 + C(typ):arrive", I).fit(cov_type="cluster", cov_kwds={"groups": I.cl})
    print("\nB. Incumbent's change in snaps (se) per 1.0 predicted rookie se arriving at his group, by incumbent type")
    print("   (share of their own snaps they lose; the arrival is split across all incumbents, so values are per incumbent)")
    for t in sorted(I.typ.unique()):
        k = f"C(typ)[{t}]:arrive"
        n = (I.typ == t).sum()
        print(f"  {t:13s} n={n:5d} mean se {I.se[I.typ == t].mean():.2f}  {m.params[k]:+.3f} ({m.bse[k]:.3f})")
    # where do the rookie's snaps come from: total lost by type per unit arrival = coef x incumbents' count at the group
    cnt = I[I.arrive > 0].groupby("typ").size() / I[I.arrive > 0].groupby(["year", "team", "grp"]).ngroups
    src = pd.Series({t: m.params[f"C(typ)[{t}]:arrive"] * cnt.get(t, 0) for t in cnt.index})
    print("  snaps taken from each type per 1.0 arriving (coef x avg count of that type in a room):",
          src.round(3).to_dict(), f"sum {src.sum():+.3f}")
    I.to_parquet(OUT / "displacement_incumbents.parquet")
    print("\nRookie year-1 se, mean by round:", R.groupby("round").s1.mean().round(2).to_dict())
    rm = I.groupby(["year", "team", "grp"]).se.sum().groupby("grp").median()
    print("Median incumbent se per room:", rm.round(2).to_dict())


if __name__ == "__main__":
    main()
