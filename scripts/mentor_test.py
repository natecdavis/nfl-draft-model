"""Veteran mentors: do rookies grow more per snap when a good veteran is in their position room?

Specified before looking (Nov 22 2026). PFF season grades (data/derived/darko_pff.parquet, 2006-25; z = grade minus
the group-season snap-weighted mean, so it is per-snap quality, not playing time).
  Rookies: rows in their first NFL season (entry), classes 2008-2024, >= 100 snaps in year 1 and >= 100 snaps
  in year 2 at the same position group (any team).
  Mentor: a veteran (4th+ NFL season) on the rookie's year-1 team, same group, >= 200 snaps that year, whose
  pre-season talent estimate (talent_pre, past seasons only) is in the top quarter of that group-season's veterans.
  PRIMARY: OLS of year-2 z on mentor, controlling for year-1 z (shrunk), log year-1 snaps, log pick (UDFA 300), age,
  group and season effects, and the team's other players' snap-weighted talent_pre in years 1 and 2 (team context);
  SEs clustered by year-1 team-season.
  KEY CHECK: split by whether the mentor is still on the rookie's team in year 2. Mentoring is development, so it
  should show when he is gone; if only when he stays, it is same-field context (e.g. coverage drawn), not mentoring.
  SECONDARY: continuous best-veteran talent_pre (sd units); old veteran (8th+ season, any quality); outcome
  year-3 z (>= 100 snaps); talent_post growth y1 -> y2; by side (offense / defense); year-2 snaps (blocking side).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/derived"
OFF = {"QB", "RB", "WR", "TE", "OL"}


def build():
    s = pd.read_parquet(OUT / "darko_pff.parquet")
    s["shrunk"] = s.z * s.snaps / (s.snaps + 300)
    pl = pd.read_parquet(ROOT / "data/raw/players.parquet")[["pff_id", "draft_pick"]].dropna(subset=["pff_id"])
    pl = pl.assign(pff_id=pl.pff_id.astype(int)).drop_duplicates("pff_id").set_index("pff_id")
    s["pick"] = s.player_id.map(pl.draft_pick).fillna(300)
    s["vet"] = s.entry.notna() & (s.season - s.entry >= 3)
    s["old"] = s.entry.notna() & (s.season - s.entry >= 7)
    v = s[s.vet & s.talent_pre.notna()]
    cut = v.groupby(["g", "season"]).talent_pre.quantile(0.75).rename("q75")
    sd = v.groupby(["g", "season"]).talent_pre.agg(["mean", "std"])
    s = s.join(cut, on=["g", "season"]).join(sd, on=["g", "season"])
    s["tp_z"] = (s.talent_pre - s["mean"]) / s["std"]
    s["good_vet"] = s.vet & (s.snaps >= 200) & (s.talent_pre >= s.q75)
    s["old_vet"] = s.old & (s.snaps >= 200)
    # team context: snap-weighted talent_pre of everyone on the team-season, per group (to exclude own group)
    s["tw"] = s.talent_pre.fillna(0) * s.snaps
    tg = s.groupby(["tm", "season", "g"])[["tw", "snaps"]].sum()
    tt = tg.groupby(["tm", "season"]).sum()
    nxt = s[["player_id", "g", "season", "tm", "z", "snaps", "talent_post"]]
    return s, tg, tt, nxt


def ctx(tg, tt, tm, yr, g):
    try:
        a, b = tt.loc[(tm, yr)], tg.loc[(tm, yr, g)]
    except KeyError:
        return np.nan
    return (a.tw - b.tw) / max(a.snaps - b.snaps, 1)


def main():
    s, tg, tt, nxt = build()
    r = s[(s.season == s.entry) & s.entry.between(2008, 2024) & (s.snaps >= 100)].copy()
    room = s[s.vet & (s.snaps >= 200)]
    key = nxt.set_index(["player_id", "g", "season"])
    rows = []
    for x in r.itertuples():
        y2 = key.loc[(x.player_id, x.g, x.season + 1)] if (x.player_id, x.g, x.season + 1) in key.index else None
        y3 = key.loc[(x.player_id, x.g, x.season + 2)] if (x.player_id, x.g, x.season + 2) in key.index else None
        rm = room[(room.tm == x.tm) & (room.season == x.season) & (room.g == x.g)]
        m = rm[rm.good_vet]
        stays = bool(len(m)) and bool(s[(s.season == x.season + 1) & s.player_id.isin(m.player_id)
                                       & (s.tm == x.tm)].shape[0])
        rows.append(dict(player_id=x.player_id, g=x.g, season=x.season, tm=x.tm, z1=x.z, sh1=x.shrunk,
                         snaps1=x.snaps, pick=x.pick, age=x.age, tpost1=x.talent_post,
                         mentor=float(len(m) > 0), mentor_stays=float(stays), mentor_gone=float(len(m) > 0 and not stays),
                         best_vet=rm.tp_z.max() if len(rm) else np.nan, old_vet=float(rm.old_vet.any()),
                         n_vet=len(rm),
                         z2=y2.z if y2 is not None else np.nan, snaps2=y2.snaps if y2 is not None else 0.0,
                         tm2=y2.tm if y2 is not None else None, tpost2=y2.talent_post if y2 is not None else np.nan,
                         z3=y3.z if y3 is not None else np.nan, snaps3=y3.snaps if y3 is not None else 0.0,
                         ctx1=ctx(tg, tt, x.tm, x.season, x.g)))
    d = pd.DataFrame(rows)
    d["ctx2"] = [ctx(tg, tt, t, y + 1, g) if t else np.nan for t, y, g in zip(d.tm2, d.season, d.g)]
    d["lsn1"] = np.log(d.snaps1)
    d["lpick"] = np.log(d.pick)
    d["side"] = np.where(d.g.isin(OFF), "off", "def")
    d["best_vet"] = d.best_vet.fillna(d.best_vet.min())
    print(f"rookies with >= 100 snaps in year 1: {len(d)}; mentor in room {int(d.mentor.sum())} "
          f"(stays {int(d.mentor_stays.sum())}, gone {int(d.mentor_gone.sum())}); year 2 >= 100 snaps "
          f"{int((d.snaps2 >= 100).sum())}")
    base = "sh1 + lsn1 + lpick + age + C(g) + C(season) + ctx1 + ctx2"
    sd2 = d.loc[d.snaps2 >= 100, "z2"].std()
    print(f"sd of year-2 z: {sd2:.2f} grade points")

    def run(lab, y, term, data, f=base):
        q = data.dropna(subset=[y, "ctx1", "ctx2"] if "ctx2" in f else [y, "ctx1"])
        m = smf.ols(f"{y} ~ {term} + {f}", q).fit(cov_type="cluster",
                                                   cov_kwds={"groups": q.tm.astype(str) + q.season.astype(str)})
        terms = [t for t in m.params.index if t in term.split(" + ")]
        print(f"  {lab:44s} n={len(q):5d} | " + " | ".join(
            f"{t} {m.params[t]:+.2f} ({m.conf_int().loc[t, 0]:+.2f} to {m.conf_int().loc[t, 1]:+.2f}, p {m.pvalues[t]:.2f})"
            for t in terms))
        return m

    y2 = d[d.snaps2 >= 100]
    print("\nPRIMARY: year-2 per-snap grade (z) on good veteran in year-1 room")
    run("good veteran in room", "z2", "mentor", y2)
    print("\nKEY CHECK: mentor still there in year 2 vs gone")
    run("stays / gone", "z2", "mentor_stays + mentor_gone", y2)
    print("\nSECONDARY")
    run("best veteran talent (sd)", "z2", "best_vet", y2)
    run("old veteran (8th+ season, any quality)", "z2", "old_vet", y2)
    run("by side: offense", "z2", "mentor", y2[y2.side == "off"])
    run("by side: defense", "z2", "mentor", y2[y2.side == "def"])
    run("stayed with year-1 team", "z2", "mentor", y2[y2.tm2 == y2.tm])
    run("year-3 z (>= 100 snaps)", "z3", "mentor", d[d.snaps3 >= 100], f=base.replace(" + ctx2", ""))
    y2t = y2.assign(dtp=y2.tpost2 - y2.tpost1)
    run("talent_post growth y1 -> y2", "dtp", "mentor", y2t)
    d["lsn2"] = np.log1p(d.snaps2)
    run("year-2 log snaps (blocking side)", "lsn2", "mentor", d, f=base.replace(" + ctx2", ""))
    for g, q in y2.groupby("g"):
        if q.mentor.sum() >= 30:
            run(f"  group {g}", "z2", "mentor", q, f=base.replace(" + C(g)", ""))
    d.to_parquet(OUT / "mentor_test.parquet")


if __name__ == "__main__":
    main()
