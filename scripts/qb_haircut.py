"""Round-1 QB over-valuation: replay the review's choices with QB title value scaled by h, scored fairly.

scripts/qb_need_yardstick.py found the model's QB recommendations to teams with an uncertain QB need delivered
0.57x their predicted titles in round 1 once scored the way the model predicts them (p_need x as played +
(1 - p_need) x sat). Here: at every pick of 2014-22 (same board, states, value tables and need counts as the
production review; tables from the simulator cache, leave-one-draft-out engine), the model's choice with the
positive part of every QB value multiplied by h. Scored with the fair yardstick: non-QBs as played, QBs at a
team with p_need < 1 as p_need x played + (1 - p_need) x sat (team picks as they happened).

Run:  python scripts/qb_haircut.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import review as R  # noqa: E402
from outlook import OUT  # noqa: E402
from positions import blended, weights  # noqa: E402
from simulator import get_table, situation  # noqa: E402
from trade import Context  # noqa: E402

YEARS = range(2014, 2023)
H = [1.0, 0.8, 0.65, 0.5, 0.35]
rng = np.random.default_rng(0)


def choices():
    S = pd.read_parquet(OUT / "review_states.parquet")
    rows = []
    for y in YEARS:
        picks, _ = R.draft_night(y)
        b = R.board(y, picks)
        W = weights(y)
        st = {r.team: r for r in S[S.year == y].itertuples()}
        tabs, counts = {}, {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            r = st[tm]
            sit = situation(r._asdict())
            key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
            if key not in tabs:
                tabs[key] = get_table(sit, year=y)
            tab = tabs[key]
            c = counts.setdefault(tm, {})
            avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
            lv = lambda g: min(c.get(g, 0), 2)
            # split each candidate's value into its QB part (positive part scaled by h) and the rest
            qb_part = np.array([blended(lambda g, *a: tab.lookup(g, *a) if g == "QB" else 0.0, lab, g, lv(g), rk, k, r.p_need, W)
                                for lab, g, rk in zip(avail.pos, avail.grp_label, avail["rank"])])
            rest = np.array([blended(lambda g, *a: 0.0 if g == "QB" else tab.lookup(g, *a), lab, g, lv(g), rk, k, r.p_need, W)
                             for lab, g, rk in zip(avail.pos, avail.grp_label, avail["rank"])])
            row = {"year": y, "pick": k, "round": p["round"], "team": tm, "player": p.pfr_player_name, "grp": p.grp,
                   "p_need": r.p_need}
            for h in H:
                v = rest + np.where(qb_part > 0, h * qb_part, qb_part)
                i = int(np.argmax(v))
                best = avail.iloc[i]
                row[f"rec_{h}"], row[f"grp_{h}"], row[f"at_{h}"] = best["name"], best.grp, best.drafted_at
            rows.append(row)
            if p.grp != "SPEC":
                c[p.grp] = c.get(p.grp, 0) + 1
        print(f"{y} done", flush=True)
    return pd.DataFrame(rows)


def score(D):
    S = pd.read_parquet(OUT / "review_states.parquet")
    sits = {(r.year, r.team): situation(r._asdict()) for r in S.itertuples()}
    real = R.RealizedV2(Context())
    cache = {}

    def fair(y, tm, grp, at, k, name, pn):
        key = (y, tm, k, name)
        if key not in cache:
            R.REALIZED_QB = "played"
            v = real.titles(sits[(y, tm)], grp, y, at, k, name)
            if grp == "QB" and pn < 1 and pd.notna(v):
                R.REALIZED_QB = "rule"
                sat = real.titles({**sits[(y, tm)], "qb_from": None}, grp, y, at, k, name)
                v = pn * v + (1 - pn) * sat
                R.REALIZED_QB = "played"
            cache[key] = v
        return cache[key]

    D["real_team"] = [fair(y, tm, g, k, k, n, 1.0) for y, tm, g, k, n in zip(D.year, D.team, D.grp, D.pick, D.player)]
    for h in H:
        D[f"real_{h}"] = [fair(y, tm, g, at, k, n, pn) for y, tm, g, at, k, n, pn in
                          zip(D.year, D.team, D[f"grp_{h}"], D[f"at_{h}"], D.pick, D[f"rec_{h}"], D.p_need)]
    return D


def boot_years(d, col):
    ys = d.year.unique()
    g = {y: d.loc[d.year == y, col].to_numpy() for y in ys}
    b = [np.concatenate([g[y] for y in rng.choice(ys, len(ys))]).mean() for _ in range(2000)]
    return f"{d[col].mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


def report(D):
    print("Model minus team where they disagree, each rec once, averaged (titles x100, 95% over draft years); fair yardstick")
    for h in H:
        d = D.rename(columns={f"rec_{h}": "rec_player", f"grp_{h}": "rec_grp", f"at_{h}": "rec_drafted_at",
                              f"real_{h}": "real_model"})[["year", "pick", "round", "team", "player", "grp", "real_team",
                                                           "rec_player", "rec_grp", "rec_drafted_at", "real_model"]]
        d = d[d.real_team.notna() & d.real_model.notna()].copy()
        d["d_t"] = d.real_model - d.real_team
        out = []
        for how in ("mean", "first"):
            o = R.once_per_draft(d, how=how)
            out.append(f"{how}: R1 {boot_years(o[o.band == 'R1'], 'd_t')} n={(o.band == 'R1').sum()}, "
                       f"R2-4 {boot_years(o[o.band == 'R2-4'], 'd_t')}")
        r1 = D[D["round"] == 1]
        print(f"  h={h:<4}  " + " | ".join(out) + f" | R1 QB recs {(r1[f'grp_{h}'] == 'QB').mean():.0%}, "
              f"same R1 rec as h=1 {(r1[f'rec_{h}'] == r1['rec_1.0']).mean():.0%}")
    print("\nPaired vs h=1, every pick (repeat recs included), sum per draft, mean over drafts (95% over draft years):")
    for h in H[1:]:
        for rn, lab in [((1, 1), "R1"), ((2, 4), "R2-4")]:
            d = D[D["round"].between(*rn)].dropna(subset=[f"real_{h}", "real_1.0"])
            per = d.assign(x=d[f"real_{h}"] - d["real_1.0"]).groupby("year").x.sum()
            b = [rng.choice(per.to_numpy(), len(per)).mean() for _ in range(2000)]
            print(f"  h={h:<4} {lab:4s} {per.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f}) per draft")


def main():
    f = OUT / "qb_haircut.parquet"
    if f.exists() and sys.argv[1:2] == ["report"]:
        D = pd.read_parquet(f)
    else:
        D = choices()
        rp = pd.read_parquet(OUT / "review_picks.parquet")
        chk = D.merge(rp[["year", "pick", "rec_player"]], on=["year", "pick"])
        print(f"check: h=1 matches the production review's recommendation at {(chk['rec_1.0'] == chk.rec_player).mean():.1%} of picks")
        D = score(D)
        D.to_parquet(f)
    report(D)


if __name__ == "__main__":
    main()
