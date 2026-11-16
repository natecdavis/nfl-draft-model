"""Cross-year need term: a strict follower that remembers its own draftees, vs one that doesn't.

Measured first (README 'Cross-year need term'): on real teams 2014-22, a rookie's playing time doesn't depend on
how much of the room the team's own draftees from the previous 3 drafts already hold (rookie-season ratio flat
~1.1 from 0 to ~70% of the room; e x load interaction +0.04, se 0.09): veterans absorb the squeeze. Real teams
never get past ~70%, so the data can't say what happens beyond the room. Rule used ("room"): draftees displace
veterans freely, but the team's own draftees can't play more than the room. A newcomer's useful snaps in season
s = min(L_s + e_s, R) - min(L_s, R), where L_s = expected snap share of the team's own draftees at the group in
that season (last 4 drafts incl. this draft's earlier picks; R1 option year), e_s = his expected share, R = the
group's median room (scripts/strict_follower.py). m = useful / expected, snap-weighted over his seasons.

Value at multiplier m: quadratic through (0, cost only), (0.88, table level 2), (1, table level 0) in the
playing-time multiplier (titles are convex in playing time; exact vs quadratic within ~5%, linear from the 0.88
slope is far off at low m). The within-draft step stays (level 1 / 2 = x0.95 / x0.88), times m. QBs: the QB need
probability p_need is multiplied by m (an own young QB holding the room makes the need less likely real).

Followers (all 32 teams, drafts 2014-2025, sequential): "base" = simulator.options as is; "room" = with m.
Both start from the team's real 2011-2013 draftees. Scored on realized titles (review.RealizedV2, classes
2014-22, recommended QBs by the fair rule), capped at the room two ways (as scripts/strict_follower_realized.py):
best-first on realized snaps (favorable) and proportional on projected snaps; the team's real picks as they were.

Run:  python scripts/cross_year_need.py draft      -> data/derived/cross_year_need_picks.parquet
      python scripts/cross_year_need.py score
"""
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import strict_follower as sf  # noqa: E402
from outlook import GROUPS, OUT, RAW  # noqa: E402

YEARS = list(range(2014, 2026))
PICKS_OUT = OUT / "cross_year_need_picks.parquet"
SCORE_OUT = OUT / "cross_year_need_scores.parquet"
WD = (1.0, 0.95, 0.88)  # draftday.WITHIN_DRAFT


def seasons(pick):
    return 5 if pick <= 32 else 4


class Load:
    """Expected snap share of a team's own draftees by (season, group)."""
    def __init__(self, f):
        self.f, self.L = f, defaultdict(float)

    def add(self, year, pick, grp, rank):
        e = self.f(grp, rank)
        for t in range(1, seasons(pick) + 1):
            self.L[(year + t - 1, grp)] += e[min(t, 4) - 1]

    def m(self, year, pick, grp, rank, room):
        e = self.f(grp, rank)
        num = den = 0.0
        for t in range(1, seasons(pick) + 1):
            es, L = e[min(t, 4) - 1], self.L[(year + t - 1, grp)]
            num += min(L + es, room[grp]) - min(L, room[grp])
            den += es
        return num / den if den > 1e-9 else 1.0


def cost_only(sit, ks):
    """Titles (fraction) of a pick that never plays but is paid (shares -> 0+), by pick: the m = 0 point."""
    from draftday import DraftEngine
    from pick_value import pick_effect
    e = DraftEngine(sit, ctx=_ctx(), n=3000)
    sh, pr = np.full((1, 5), 1e-9), np.ones(1)
    return {k: e.titles(pick_effect(sh, pr, k, "WR", {**sit, "qb_from": 99}, e.ctx.cost, e.ctx.bar, e.ctx.rates))
            for k in ks}


_C = []


def _ctx():
    if not _C:
        from trade import Context
        _C.append(Context())
    return _C[0]


def quad(v0, v88, v1, x):
    """Quadratic through (0, v0), (0.88, v88), (1, v1), at x."""
    a = ((v88 - v0) - 0.88 * (v1 - v0)) / (0.88 ** 2 - 0.88)
    b = (v1 - v0) - a
    return a * x * x + b * x + v0


def draft():
    from positions import blended, weights
    from review import OFF_BOARD_RANK, draft_night
    from simulator import get_table, pool, situation
    from specialists import SPEC_GROUPS  # noqa: F401
    sh = sf.season_shares()
    ps = sh.groupby(["pfr_player_id", "season"]).share.sum()
    room = sf.rooms(sh)
    room = room[room.season.between(2014, 2025)].groupby("grp").share.median().to_dict()
    bp = sf.board_players()
    f = sf.expected_curve(bp, ps)
    rk = {(y, k): r for y, k, r in zip(bp.year, bp.pick, bp["rank"])}
    # initial loads: the team's real 2011-2013 draftees (draft_picks team codes -> review codes via 2014 picks)
    from draftday import POS
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    code = dict(zip(dp[dp.season == 2014].pick, dp[dp.season == 2014].team))
    p14, _ = draft_night(2014)
    pfr2rev = {code[k]: t for k, t in zip(p14.pick, p14.team) if k in code}  # same franchises 2011-13
    S = pd.read_parquet(OUT / "review_states.parquet")
    teams = sorted(S.team.unique())
    loads = {(v, tm): Load(f) for v in ("base", "room") for tm in teams}
    old = dp[dp.season.between(2011, 2013)]
    for r in old.itertuples():
        g, tm = POS.get(str(r.position).upper()), pfr2rev.get(r.team, r.team)
        if g in GROUPS:
            for v in ("base", "room"):
                if (v, tm) in loads:
                    loads[(v, tm)].add(int(r.season), int(r.pick), g, 300.0)
    rows = []
    for y in YEARS:
        picks, _ = draft_night(y)
        P = pool(y)
        W = weights(y)
        St = S[S.year == y]
        for _, s in St.iterrows():
            sit = situation(s)
            tab = get_table(sit, year=y)
            mine = sorted(int(k) for k in picks[picks.team == s.team].pick)
            c0 = cost_only(sit, mine)
            for v in ("base", "room"):
                ld = loads.get((v, s.team)) or loads.setdefault((v, s.team), Load(f))
                taken, counts = set(), {}
                for k in mine:
                    a = P[(P.drafted_at.isna() | (P.drafted_at >= k)) & ~P.pid.isin(list(taken)) & P.grp.isin(GROUPS)]
                    if not len(a):
                        continue

                    def lk(g, level, r, pk, p_need=1.0):
                        if v == "base":
                            return tab.lookup(g, level, r, pk, p_need)
                        mm = ld.m(y, pk, g, r, room)
                        if g == "QB":
                            return tab.lookup(g, level, r, pk, p_need * mm)
                        v1, v88 = tab.lookup(g, 0, r, pk), tab.lookup(g, 2, r, pk)
                        return quad(c0[pk], v88, v1, WD[level] * mm)
                    ranks = [OFF_BOARD_RANK if pd.isna(x) else x for x in a["rank"]]
                    vals = np.array([blended(lk, lab, g, min(counts.get(g, 0), 2), r, k, float(s.p_need), W)
                                     for lab, g, r in zip(a.label, a.grp, ranks)])
                    i = int(np.argmax(vals))
                    b = a.iloc[i]
                    r_ = OFF_BOARD_RANK if pd.isna(b["rank"]) else float(b["rank"])
                    rows.append({"version": v, "year": y, "team": s.team, "pick": k, "name": b["name"], "grp": b.grp,
                                 "rank": r_, "drafted_at": b.drafted_at, "titles": vals[i] * 100,
                                 "m": ld.m(y, k, b.grp, r_, room), "p_need": float(s.p_need)})
                    taken.add(b.pid)
                    counts[b.grp] = counts.get(b.grp, 0) + 1
                    ld.add(y, k, b.grp, r_)
        print(y, len(rows), flush=True)
    pd.DataFrame(rows).to_parquet(PICKS_OUT)


def keeps(F, f, room, ps, pid):
    """Per pick: share of value kept under the room, best-first on realized snaps and proportional on projected."""
    rows = []
    for i, r in F.iterrows():
        e = f(r.grp, r["rank"])
        p = pid.get((r.year, int(r.drafted_at))) if pd.notna(r.drafted_at) else None
        for t in range(1, seasons(r.pick) + 1):
            s = r.year + t - 1
            rows.append((i, r.team, s, r.grp, e[min(t, 4) - 1], ps.get((p, s), 0.0) if p else 0.0))
    X = pd.DataFrame(rows, columns=["i", "team", "season", "grp", "proj", "real"])
    R = X.grp.map(room)
    tot = X.groupby(["team", "season", "grp"]).proj.transform("sum")
    X["pk"] = X.proj * np.minimum(1, R / tot.where(tot > 0, 1))
    X = X.sort_values("real", ascending=False)
    cum = X.groupby(["team", "season", "grp"]).real.cumsum()
    X["rk"] = np.clip(R - (cum - X.real), 0, X.real)
    g = X.groupby("i")[["proj", "pk", "real", "rk"]].sum()
    return (g.pk / g.proj.where(g.proj > 0)).fillna(1.0), (g.rk / g.real.where(g.real > 0)).fillna(1.0)


def score():
    from review import RealizedV2
    from simulator import situation
    F = pd.read_parquet(PICKS_OUT)
    sh = sf.season_shares()
    ps = sh.groupby(["pfr_player_id", "season"]).share.sum()
    room = sf.rooms(sh)
    room = room[room.season.between(2014, 2025)].groupby("grp").share.median().to_dict()
    f = sf.expected_curve(sf.board_players(), ps)
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pid = {(int(s), int(k)): i for s, k, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    S = pd.read_parquet(OUT / "review_states.parquet").set_index(["year", "team"])
    R = RealizedV2(_ctx())
    out = []
    for v, Fv in F.groupby("version"):
        Fv = Fv.reset_index(drop=True)
        kp, kb = keeps(Fv, f, room, ps, pid)
        Fv["keep_prop"], Fv["keep_best"] = kp, kb
        Fv = Fv[Fv.year <= 2022]
        for c, sc in [("real", None), ("real_best", "keep_best"), ("real_prop", "keep_prop")]:
            vals = []
            for r in Fv.itertuples():
                st = S.loc[(r.year, r.team)]
                sit = situation(st)
                scale = 1.0 if sc is None else getattr(r, sc)
                x = R.titles(sit, r.grp, r.year, r.drafted_at, r.pick, r.name, scale=scale)
                if r.grp == "QB" and r.p_need < 1:
                    sat = R.titles({**sit, "qb_from": None}, r.grp, r.year, r.drafted_at, r.pick, r.name, scale=scale, qb="rule")
                    x = r.p_need * x + (1 - r.p_need) * sat
                vals.append(x)
            Fv[c] = vals
        out.append(Fv)
        print(v, "scored", flush=True)
    pd.concat(out).to_parquet(SCORE_OUT)
    report()


def boot(x, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = x.to_numpy()
    s = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return f"{a.mean():+.2f} ({np.percentile(s, 2.5):+.2f} to {np.percentile(s, 97.5):+.2f})"


def report():
    D = pd.read_parquet(SCORE_OUT)
    rp = pd.read_parquet(OUT / "review_picks.parquet")
    rp = rp[rp.year.le(2022) & rp.grp.ne("SPEC")]
    team = rp.groupby(["year", "team"]).real_team.sum()
    print("Picks by group 2014-25 (follower base / room / teams):")
    allp = pd.read_parquet(PICKS_OUT)
    tp = pd.read_parquet(OUT / "review_picks.parquet")
    print(pd.DataFrame({"base": allp[allp.version == "base"].grp.value_counts(),
                        "room": allp[allp.version == "room"].grp.value_counts(),
                        "teams": tp[tp.grp.isin(GROUPS)].grp.value_counts()}).fillna(0).astype(int).to_string(), "\n")
    print("Realized titles (% of one) per team-draft 2014-22, follower minus team's real picks (95% CI, team-drafts):")
    T = {}
    for v in ("base", "room"):
        d = D[D.version == v].groupby(["year", "team"])[["real", "real_best", "real_prop"]].sum()
        T[v] = d
        j = d.join(team, how="inner")
        print(f"  {v:5s} uncapped {boot(j.real - j.real_team)} | capped best-first {boot(j.real_best - j.real_team)}"
              f" | capped proportional {boot(j.real_prop - j.real_team)}")
    dd = T["room"].join(T["base"].add_suffix("_b"), how="inner")
    print("Room minus base, paired:")
    for c in ("real", "real_best", "real_prop"):
        print(f"  {c:10s} {boot(dd[c] - dd[c + '_b'])}")
    same = allp.pivot_table(index=["year", "team", "pick"], columns="version", values="name", aggfunc="first")
    print(f"\nSame choice as base: {(same.base == same.room).mean() * 100:.0f}% of picks")
    rooms = []
    sh = sf.season_shares()
    rm = sf.rooms(sh)
    rm = rm[rm.season.between(2014, 2025)].groupby("grp").share.median()
    f = sf.expected_curve(sf.board_players(), sh.groupby(["pfr_player_id", "season"]).share.sum())
    for v in ("base", "room"):
        st = sf.stack(allp[allp.version == v].assign(year=lambda x: x.year), f)
        st = st[st.season.between(2017, 2025)].groupby(["team", "season", "grp"]).proj.sum().reset_index()
        st["over"] = st.proj > st.grp.map(rm)
        rooms.append(st.groupby("grp").agg(**{f"{v}_mean": ("proj", "mean"), f"{v}_%over": ("over", "mean")}))
    r = pd.concat(rooms, axis=1)
    r["room"] = rm
    r[["base_%over", "room_%over"]] *= 100
    print("\nOwn draftees' projected snap share per team-season 2017-25 (team-seasons with any at the group):")
    print(r.round(2).to_string())
    print("\nNE room follower R1-4:")
    x = allp[(allp.version == "room") & (allp.team == "NE") & (allp.pick <= 140)]
    print(x[["year", "pick", "name", "grp", "rank", "m"]].round(2).to_string(index=False))


if __name__ == "__main__":
    {"draft": draft, "score": score, "report": report}[sys.argv[1]]()
