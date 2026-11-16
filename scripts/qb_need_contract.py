"""QB need while the veteran is under contract (open idea from complaint #1).

review.p_need: chance a flagged QB need is real, by status only (stopgap 0.79, later-round rookie 0.62,
below-average paid vet 0.54, age 36+ 0.52, no starter 1.0), measured as the share of flagged team-years whose
main QB later was someone else. Does the draft-day QB's remaining contract (known on draft day) split this?

Team-years: the review's draft-day states (data/derived/review_states.parquet), drafts 2014-2024.
Incumbent's contract: as review.States.qb (latest deal known on draft day); end = year_signed + years - 1;
left = end - draft year (0: expires after this season). Outcome: main QB (most regular-season offensive snaps
for the team) in seasons year+1 and year+2 is not the draft-day QB. Also shown excluding team-years where the
team took a QB in rounds 1-2 that draft (the pick itself can cause the change).

Choice test (`choice`): at every pick 2014-22 of a team flagged "age 36+" or "below-average paid vet" with a known
contract, replay the review's choice (same board, states, tables and need counts as the production review; tables
from the simulator cache, leave-one-draft-out engine) with p_need = the rate for "under contract next year" or
"expiring" (both veteran flags pooled, that draft year left out) instead of 0.52-0.54. Other teams' picks are
unchanged. Scored three ways: fair yardstick with the old p_need, with the new p_need, and "what happened" (the
rec QB counts as played if the incumbent was in fact replaced the next season, else as sat; for teams that took a
QB in rounds 1-2 the replacement may be the pick's doing, so the new p_need is used there).

Run: .venv/bin/python scripts/qb_need_contract.py [report | choice [report]]
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import review as Rv  # noqa: E402
from outlook import OUT, RAW  # noqa: E402

YEARS = range(2014, 2025)


def main_qbs():
    s = pd.concat([pd.read_parquet(RAW / "snap_counts.parquet"), pd.read_parquet(RAW / "snap_counts_2026.parquet")])
    s = s[(s.position == "QB") & (s.game_type == "REG")].assign(team=lambda x: x.team.map(Rv.team))
    g = s.groupby(["season", "team", "pfr_player_id"]).offense_snaps.sum().reset_index()
    g = g.sort_values("offense_snaps").drop_duplicates(["season", "team"], keep="last")
    return {(int(a), b): c for a, b, c in zip(g.season, g.team, g.pfr_player_id)}, int(s.season.max())


def main():
    St = Rv.States()
    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(YEARS)].copy()
    main, last = main_qbs()
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dp["tm"] = dp.team.map(Rv.team)
    took = set(zip(dp[(dp.position == "QB") & (dp["round"] <= 2)].season, dp[(dp.position == "QB") & (dp["round"] <= 2)].tm))
    rows = []
    for r in S.itertuples():
        y, tm = int(r.year), r.team
        # the incumbent (same rule as review.States.qb)
        wk = St.snaps[(St.snaps.season == y) & (St.snaps.team == tm)]
        pid = None
        if len(wk):
            w1 = wk[wk.week == wk.week.min()].sort_values("offense_snaps").iloc[-1]
            dy = St.players.draft_year.get(w1.pfr_player_id)
            if not (pd.notna(dy) and int(dy) == y):
                pid = w1.pfr_player_id
        if pid is None:
            prev = St.snaps[(St.snaps.season == y - 1) & (St.snaps.team == tm)]
            if len(prev):
                pid = prev.groupby("pfr_player_id").offense_snaps.sum().idxmax()
        left, gtd = np.nan, np.nan
        if pid is not None and pid in St.players.index:
            otc = pd.to_numeric(St.players.loc[pid].otc_id, errors="coerce")
            c = St.contracts[St.contracts.otc_id == otc]
            k = c[(c.year_signed < y) | ((c.year_signed == y) & (c.team_code.map(Rv.team) == tm))].sort_values("year_signed")
            if len(k):
                lt = k.iloc[-1]
                left = lt.year_signed + lt.years - 1 - y
                gtd = lt.guaranteed / lt.value if lt.value else np.nan
        out = {}
        for h in (1, 2):
            m = main.get((y + h, tm))
            out[f"chg{h}"] = np.nan if y + h > last or m is None else float(m != pid)
        rows.append({"year": y, "team": tm, "qb": r.qb, "status": r.status, "p_need": r.p_need, "qb_from": r.qb_from,
                     "left": left, "gtd_share": gtd, "took_qb": (y, tm) in took, **out})
    D = pd.DataFrame(rows)
    D["cls"] = np.select([D.status.str.startswith("no starter"), D.status.str.startswith("stopgap"),
                          D.status.str.startswith("rookie deal (later"), D.status.str.contains("below-average"),
                          D.status.str.match(r"veteran, \d")],
                         ["no starter", "stopgap", "later-round rookie", "below-avg vet", "age 36+"], "unflagged")
    D["under"] = np.where(D.left.isna(), "unknown", np.where(D.left >= 1, "under contract next year", "expiring"))
    D.to_parquet(OUT / "qb_need_contract.parquet")
    report(D)


def report(D):
    pd.set_option("display.width", 200)
    for lab, d in [("all team-years", D), ("excluding teams that took a QB in rounds 1-2", D[~D.took_qb])]:
        print(f"\n{lab}: main QB changed (share), by status x contract")
        t = d.groupby(["cls", "under"]).agg(n=("chg1", "size"), chg_next=("chg1", "mean"), chg_2=("chg2", "mean"),
                                           p_need=("p_need", "mean"))
        print(t.round(2).to_string())
        print(d.groupby("cls").agg(n=("chg1", "size"), chg_next=("chg1", "mean"), chg_2=("chg2", "mean")).round(2).to_string())
    f = D[D.cls != "unflagged"]
    print("\nflagged, years left on deal:")
    print(f.groupby(f.left.clip(upper=3)).agg(n=("chg2", "size"), chg_next=("chg1", "mean"), chg_2=("chg2", "mean")).round(2).to_string())


VET = ("age 36+", "below-avg vet")
rng = np.random.default_rng(0)


def new_rates(D):
    """(year, team) -> p_need under the contract split, rates from the other draft years."""
    v = D[D.cls.isin(VET) & (D.under != "unknown")]
    out = {}
    for r in v.itertuples():
        o = v[(v.year != r.year) & (v.under == r.under)]
        out[(r.year, r.team)] = float(o.chg1.mean())
    return out


def choices(D):
    from positions import blended, weights
    from simulator import get_table, situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    rp = pd.read_parquet(OUT / "review_picks.parquet").set_index(["year", "pick"])
    NEW = new_rates(D)
    rows = []
    for y in range(2014, 2023):
        aff = {tm for (yy, tm) in NEW if yy == y}
        picks, _ = Rv.draft_night(y)
        b = Rv.board(y, picks)
        W = weights(y)
        st = {r.team: r for r in S[S.year == y].itertuples()}
        tabs, counts = {}, {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            c = counts.setdefault(tm, {})
            if tm in aff:
                r = st[tm]
                sit = situation(r._asdict())
                key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
                if key not in tabs:
                    tabs[key] = get_table(sit, year=y)
                tab = tabs[key]
                avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
                row = {"year": y, "pick": k, "round": p["round"], "team": tm, "player": p.pfr_player_name, "grp": p.grp,
                       "p_old": r.p_need, "p_new": NEW[(y, tm)]}
                for vn, pn in (("now", r.p_need), ("contract", NEW[(y, tm)])):
                    v = np.array([blended(tab.lookup, lab, g, min(c.get(g, 0), 2), rk, k, pn, W)
                                  for lab, g, rk in zip(avail.pos, avail.grp_label, avail["rank"])])
                    best = avail.iloc[int(np.argmax(v))]
                    row[f"rec_{vn}"], row[f"grp_{vn}"], row[f"at_{vn}"], row[f"pred_{vn}"] = best["name"], best.grp, best.drafted_at, v.max()
                row["prod_rec"] = rp.rec_player.get((y, k))
                rows.append(row)
            if p.grp != "SPEC":
                c[p.grp] = c.get(p.grp, 0) + 1
        print(f"{y} done ({len(aff)} teams)", flush=True)
    return pd.DataFrame(rows)


def score(C, D):
    from simulator import situation
    from trade import Context
    S = pd.read_parquet(OUT / "review_states.parquet")
    sits = {(r.year, r.team): situation(r._asdict()) for r in S.itertuples()}
    real = Rv.RealizedV2(Context())
    happ = {(r.year, r.team): (r.chg1, r.took_qb) for r in D.itertuples()}
    cache = {}

    def parts(y, tm, g, at, k, n):
        key = (y, tm, k, n)
        if key not in cache:
            Rv.REALIZED_QB = "played"
            v = real.titles(sits[(y, tm)], g, y, at, k, n)
            sat = v
            if g == "QB" and pd.notna(v):
                Rv.REALIZED_QB = "rule"
                sat = real.titles({**sits[(y, tm)], "qb_from": None}, g, y, at, k, n)
                Rv.REALIZED_QB = "played"
            cache[key] = (v, sat)
        return cache[key]

    C["real_team"] = [parts(y, tm, g, k, k, n)[0] for y, tm, g, k, n in zip(C.year, C.team, C.grp, C.pick, C.player)]
    for vn in ("now", "contract"):
        P = [parts(y, tm, g, at, k, n) for y, tm, g, at, k, n in
             zip(C.year, C.team, C[f"grp_{vn}"], C[f"at_{vn}"], C.pick, C[f"rec_{vn}"])]
        pl, sat = np.array([x[0] for x in P], float), np.array([x[1] for x in P], float)
        C[f"old_{vn}"] = C.p_old * pl + (1 - C.p_old) * sat
        C[f"new_{vn}"] = C.p_new * pl + (1 - C.p_new) * sat
        ph = np.array([C.p_new.iloc[i] if happ[(y, tm)][1] or pd.isna(happ[(y, tm)][0]) else happ[(y, tm)][0]
                       for i, (y, tm) in enumerate(zip(C.year, C.team))])
        C[f"happ_{vn}"] = ph * pl + (1 - ph) * sat
    return C


def choice_report(C):
    print(f"\n{len(C)} picks of {C.groupby(['year', 'team']).ngroups} affected team-drafts 2014-22; "
          f"'now' matches the production review at {(C.rec_now == C.prod_rec).mean():.1%}")
    for lab, d in [("under contract (p down)", C[C.p_new < C.p_old]), ("expiring (p up)", C[C.p_new > C.p_old])]:
        ch = d[d.rec_now != d.rec_contract]
        print(f"\n{lab}: {d.groupby(['year', 'team']).ngroups} team-drafts, {len(d)} picks, rec changed at {len(ch)}; "
              f"QB recs {(d.grp_now == 'QB').mean():.0%} -> {(d.grp_contract == 'QB').mean():.0%} "
              f"(round 1 {(d[d['round'] == 1].grp_now == 'QB').mean():.0%} -> {(d[d['round'] == 1].grp_contract == 'QB').mean():.0%})")
        for ys in ("old", "new", "happ"):
            per = d.assign(x=d[f"{ys}_contract"] - d[f"{ys}_now"]).groupby("year").x.sum()
            bs = [rng.choice(per.to_numpy(), len(per)).mean() for _ in range(2000)]
            print(f"   contract minus now, yardstick {ys:4s}: {per.mean():+.2f} ({np.percentile(bs, 2.5):+.2f} to "
                  f"{np.percentile(bs, 97.5):+.2f}) per draft; changed picks only mean {(ch[f'{ys}_contract'] - ch[f'{ys}_now']).mean():+.2f}")
        cols = ["year", "team", "pick", "p_old", "p_new", "rec_now", "grp_now", "rec_contract", "grp_contract", "happ_now", "happ_contract"]
        print(ch[cols].round(2).to_string(index=False))
    for ys in ("old", "new", "happ"):
        per = C.assign(x=C[f"{ys}_contract"] - C[f"{ys}_now"]).groupby("year").x.sum()
        bs = [rng.choice(per.to_numpy(), len(per)).mean() for _ in range(2000)]
        print(f"all affected, yardstick {ys}: {per.mean():+.2f} ({np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f}) per draft")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["report"]:
        report(pd.read_parquet(OUT / "qb_need_contract.parquet"))
    elif a[:1] == ["choice"]:
        f = OUT / "qb_need_contract_choice.parquet"
        if a[1:2] == ["report"] and f.exists():
            choice_report(pd.read_parquet(f))
        else:
            D = pd.read_parquet(OUT / "qb_need_contract.parquet")
            C = score(choices(D), D)
            C.to_parquet(f)
            choice_report(C)
    else:
        main()
