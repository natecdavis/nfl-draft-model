"""Room check v2 (simulator, Nov 16 2026): crowding, players who missed last season, the QB starter, contract fixes.

The room check projects each team's open snaps per position on draft day (README "Room check: projected open
snaps"). Spot checks of past drafts (rooms projected full that teams drafted into anyway, and rooms left thin)
found four foreseeable gaps; this script builds the fixed version and tests it out of sample.

1. Crowding. Each incumbent was projected on his own, so the projections ignored how crowded his room is. Snaps are
   shared: in rooms left under 80% full after the draft incumbents played 1.66 starters vs 1.32 projected; in rooms
   over 110% full, 1.72 vs 2.18. Fix: incumbents' total = projected + k x open, open = room - projected -
   expected season-1 share of the team's picks; k per room, fit across team-seasons. The page then shows
   (1 - k) x open as the space left for rookies.
2. Players with no snaps last season (injured all year, sat out, redshirted rookies) weren't incumbents at all
   (displacement.missed): projected with indicators by kind (missed / drafted R1-2 / R3+) and their se two seasons ago.
3. QB: one quarterback plays nearly every snap. The draft-day starter (review_states.qb, the QB-need model's
   starter) is projected from the designated starters' record instead of his own last season (Mahomes 2018: 0.28).
4. Contracts (roster_need.signed_by_draft): signings come from OTC's chronological contract history, dropping
   trade copies of a deal, practice-squad deals and signings after the draft (screened by snaps and order).

Test: each season 2017-25 predicted from earlier seasons only (incumbent regression, starter rule, k), RMSE of the
incumbents' total per team-room (actual = their snaps for the team that season) and of the room's total.

Usage: python scripts/room_crowding.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import displacement as dz  # noqa: E402
import incumbent_development as idv  # noqa: E402
import room_persistence as rp  # noqa: E402
import roster_need as rn  # noqa: E402
from outlook import OUT, RAW  # noqa: E402

ROOM_GROUPS = ["QB", "RB", "WR", "TE", "OL", "ED", "IDL", "LB", "CB", "S"]  # T / G / C share one OL room
FORM = idv.FORMS["+out"]
rg_of = lambda g: "OL" if g in ("T", "G", "C") else g


def build(years):
    """Incumbents (with missed-season players) and their regression inputs; team rooms R; season-level frames."""
    if "QB" not in rn.GROUPS:
        rn.GROUPS = rn.GROUPS + ["QB"]
    t = rp.features(rp.totals())
    b = {g: float(rp.fit(d[d.season.between(2014, 2025) & d.dev1.notna()], ["dev1"])[0]) for g, d in t.groupby("grp")}
    lg = t.groupby(["grp", "season"]).tot.median()
    last = t.set_index(["season", "team", "grp"]).dev
    s, _, games, dp = rn.load()
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    ppos = {i: rp.PFF.get(q) for i, q in zip(pl.pfr_id, pl.pff_position) if rp.PFF.get(q)}
    dpos = dict(zip(dp.pfr_player_id, dp.position.map(rn.DRAFT_POS)))
    s["grp"] = [ppos.get(p) or rn.SNAP_POS.get(q) or dpos.get(p) for p, q in zip(s.pfr_player_id, s.position)]
    ps = (s.groupby(["season", "team", "pfr_player_id"])
          .agg(pct=("pct", "sum"), grp=("grp", lambda x: x.mode().iloc[0] if x.notna().any() else None)).reset_index())
    ps["se"] = ps.pct / ps.set_index(["season", "team"]).index.map(games)
    A = dz.attributes()
    grp_of = {**dpos, **ppos}
    I = dz.types(pd.concat([pd.concat([dz.incumbents(ps, y, A).assign(out1="no"), dz.missed(ps, y, A, grp_of)],
                                      ignore_index=True) for y in years], ignore_index=True))
    me = ps.set_index(["season", "team", "pfr_player_id"]).se
    last_season = int(ps.season.max())
    I["now"] = [float(me.get((y, tm, p), 0.0)) if y <= last_season else np.nan
                for y, tm, p in zip(I.year, I.team, I.pfr_player_id)]
    first = ps.groupby("pfr_player_id").season.min()
    dy = dict(zip(dp.pfr_player_id, dp.season))
    I["exp"] = I.year - np.array([dy.get(p, first.get(p, np.nan)) for p in I.pfr_player_id], float)
    I["age_"] = I.age.fillna(I.age.median())
    I["ab"] = pd.cut(I.age_, idv.AGE_BINS, labels=idv.AGE_LAB).astype(str)
    I["eb"] = pd.cut(I.exp.fillna(7), idv.EXP_BINS, labels=idv.EXP_LAB).astype(str)
    I = idv.add_availability(I, s, games)
    I["rg"] = I.grp.map(rg_of)
    names = s.drop_duplicates("pfr_player_id").set_index("pfr_player_id").player
    I["name"] = I.pfr_player_id.map(names).fillna(I.pfr_player_id)
    # the draft-day starting QB (QB-need model)
    S = pd.read_parquet(OUT / "review_states.parquet")
    qb = {(int(y), tm): n for y, tm, n in zip(S.year, S.team, S.qb) if isinstance(n, str) and n}
    I["starter"] = [(rg == "QB") and qb.get((y, tm)) == n for rg, y, tm, n in zip(I.rg, I.year, I.team, I.name)]
    rooms = {}
    for y in years:
        for tm in sorted(ps[ps.season == min(y - 1, last_season)].team.unique()):
            room = {}
            for g in ROOM_GROUPS:
                med = lg[g][[x for x in lg[g].index if y - 4 <= x < y]].mean()
                dev = last.get((y - 1, tm, g), np.nan)
                room[g] = float(med + (b[g] * dev if pd.notna(dev) else 0.0))
            rooms[(y, tm)] = room
    ps["rg"] = ps.grp.map(rg_of)
    return I, rooms, ps


def project(I, train, starter=False):
    """Fit the incumbent regression on `train` rows and project I. starter=True: the draft-day starting QB gets at
    least the designated starters' mean share (tested, not adopted: crowding already fills the QB room)."""
    m = smf.ols(FORM, train[train.now.notna()]).fit()
    proj = m.predict(I).clip(lower=0).fillna(0.0).to_numpy()
    if starter:
        proj = np.where(I.starter, np.maximum(proj, float(train[train.starter & train.now.notna()].now.mean())), proj)
    return proj


def pick_adds(years):
    """Expected season-1 share of each team's real picks by room (consensus rank curve, as the page)."""
    import strict_follower as sf
    from review import OFF_BOARD_RANK
    sh = sf.season_shares()
    f = sf.expected_curve(sf.board_players(), sh.groupby(["pfr_player_id", "season"]).share.sum())
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[R.year.isin(years)]
    R["e1"] = [float(f(g, r if pd.notna(r) else OFF_BOARD_RANK)[0]) for g, r in zip(R.grp, R.board_rank)]
    R["rg"] = R.grp.map(rg_of)
    return R.groupby(["year", "team", "rg"]).e1.sum(), f


def veteran_snaps(ps):
    """Each team-room's snaps by everyone but that season's draft class (incumbents, later signings, undrafted)."""
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dy = dict(zip(dp.pfr_player_id, dp.season))
    v = ps[[dy.get(p) != s for p, s in zip(ps.pfr_player_id, ps.season)]]
    return v.groupby(["season", "team", "rg"]).se.sum()


def team_rooms(I, rooms, ps, picks, proj_col, vet=None):
    rows = []
    inc = I.groupby(["year", "team", "rg"])[proj_col].sum()
    now = I.groupby(["year", "team", "rg"]).now.sum()
    act = ps.groupby(["season", "team", "rg"]).se.sum()
    for (y, tm), room in rooms.items():
        for g, R in room.items():
            rows.append((y, tm, g, R, float(inc.get((y, tm, g), 0.0)), float(picks.get((y, tm, g), 0.0)),
                         float(now.get((y, tm, g), np.nan)), float(act.get((y, tm, g), np.nan))))
    U = pd.DataFrame(rows, columns=["year", "team", "rg", "room", "inc", "picks", "now", "actual"])
    U["open"] = U.room - U.inc - U.picks
    if vet is not None:
        U["vet"] = [float(vet.get((y, tm, g), 0.0)) for y, tm, g in zip(U.year, U.team, U.rg)]
    return U


def fit_k(U, target="vet"):
    """Share of the open space (room - projected incumbents - expected share of the team's picks) taken by veterans
    (target "vet": incumbents plus anyone added after the draft; "now": the incumbents alone), per room. The page
    shows (1 - k) x open as the space left for rookies."""
    U = U.dropna(subset=[target])
    return {g: float(np.linalg.lstsq(d[["open"]].to_numpy(), (d[target] - d.inc).to_numpy(), rcond=None)[0][0])
            for g, d in U.groupby("rg")}


def final(I, rooms, ps, picks):
    """Fit on all seasons with outcomes (what the simulator uses): projections for I, k per room."""
    I = I.copy()
    fit = I[I.now.notna()]
    I["proj"] = project(I, fit)
    vet = veteran_snaps(ps)
    U = team_rooms(I[I.now.notna()], {r: v for r, v in rooms.items() if r[0] in set(fit.year)}, ps, picks, "proj", vet)
    return I, fit_k(U)


def main():
    years = list(range(2015, 2027))
    I, rooms, ps = build(years)
    I = I[I.year <= 2025]
    picks, _ = pick_adds(years)
    vet = veteran_snaps(ps)
    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
    pd.set_option("display.width", 200)
    cands = {"+inj, last season's players only": ("inj", False), "+ no snaps last season": ("out", False),
             "+ QB starter rule": ("out", True)}
    res = {}
    for name, (kind, st) in cands.items():
        out = []
        for y in range(2017, 2026):
            tr, te = I[I.year < y].copy(), I[I.year == y].copy()
            if kind == "inj":
                tr, te = tr[tr.out1 == "no"], te[te.out1 == "no"]
                m = smf.ols(idv.FORMS["+inj"], tr[tr.now.notna()]).fit()
                tr["p"], te["p"] = m.predict(tr).clip(lower=0), m.predict(te).clip(lower=0)
            else:
                tr["p"], te["p"] = project(tr, tr, st), project(te, tr, st)
            ry = lambda f: {r: v for r, v in rooms.items() if f(r[0])}
            Ut = team_rooms(tr, ry(lambda x: x < y), ps, picks, "p", vet)
            U = team_rooms(te, ry(lambda x: x == y), ps, picks, "p", vet)
            for tg in ("now", "vet"):
                k = fit_k(Ut, tg)
                U[f"crowd_{tg}"] = U.inc + U.rg.map(k) * U.open
            out.append(U)
        res[name] = pd.concat(out)
    rows = []
    for name, U in res.items():
        rows.append([name, rmse(U.now, U.inc), rmse(U.now, U.crowd_now), rmse(U.vet, U.inc), rmse(U.vet, U.crowd_vet)])
    U = res["+ no snaps last season"]
    print(f"Out of sample 2017-25 ({len(U)} team-rooms; regression, starter rule and k fit on earlier seasons). RMSE in "
          "starters of the incumbents' total (their snaps for the team) and of all veterans' snaps in the room "
          "(incumbents + anyone added after the draft; everyone but the draft class):")
    print(pd.DataFrame(rows, columns=["projection", "incumbents", "+ crowding", "veterans", "+ crowding "])
          .set_index("projection").round(3).to_string())
    print("\nby room, veterans' snaps (+inj -> adopted: no-snaps players + crowding):")
    U0 = res["+inj, last season's players only"]
    print(pd.DataFrame({"before": U0.groupby("rg").apply(lambda d: rmse(d.vet, d.inc), include_groups=False),
                        "after": U.groupby("rg").apply(lambda d: rmse(d.vet, d.crowd_vet), include_groups=False)})
          .round(3).T.to_string())
    U["band"] = pd.cut((U.inc + U.picks) / U.room, [0, .8, .9, 1.0, 1.1, 9],
                       labels=["<80%", "80-90", "90-100", "100-110", ">110"])
    print("\nby projected fill after the draft (mean starters): incumbents projected / + crowding / veterans actual")
    print(U.groupby("band", observed=True)[["inc", "crowd_vet", "vet"]].mean().round(2).T.to_string())
    If, k = final(I, rooms, ps, picks)
    print("\nfit on 2015-25, k (share of open space veterans take):", {g: round(v, 2) for g, v in k.items()})
    st = If[If.starter]
    print(f"draft-day starting QBs {len(st)}: projected {st.proj.mean():.2f}, actual {st.now.mean():.2f}")
    print("\nplayers with no snaps last season, 2015-25: n / projected / actual mean share")
    mo = If[If.out1 != "no"]
    print(mo.groupby("out1").agg(n=("now", "size"), proj=("proj", "mean"), now=("now", "mean")).round(3).to_string())
    big = mo[mo.proj >= 0.3].sort_values("proj", ascending=False)
    print(big[["year", "team", "rg", "name", "out1", "se2", "proj", "now"]].head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
