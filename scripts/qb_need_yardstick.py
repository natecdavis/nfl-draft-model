"""Fair realized value for the model's QB recommendations to teams whose QB need is uncertain.

The review scores a recommended QB "as played" (review.REALIZED_QB): his real career, which happened at the team
that actually drafted him, usually one that started him. But the model recommended him to a team whose flagged QB
need was real only with probability p_need (review.p_need: 0.52-0.79); otherwise its incumbent kept the job and the
rookie sat. The model's own prediction already discounts by p_need; the yardstick didn't. Fair score for a QB
recommendation: p_need x (as played) + (1 - p_need) x (sat through the window: seasons count zero, cost charged).
Team picks are unchanged (what happened is what happened). Drafts 2014-22, production review (leave-one-draft-out).
Adopted Nov 13 2026 as the review's default (review.REALIZED_REC = "fair"); this script recomputes both versions.

Prints the scorecard (model minus team where they disagree, each rec once) both ways, and delivered / predicted
for the model's QB recommendations by round, both ways.

Run:  python scripts/qb_need_yardstick.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import review as R  # noqa: E402
from outlook import OUT  # noqa: E402
from trade import Context  # noqa: E402

YEARS = range(2014, 2023)
rng = np.random.default_rng(0)


def boot_years(d, col):
    """Mean with 95% range resampling draft years."""
    ys = d.year.unique()
    g = {y: d.loc[d.year == y, col].to_numpy() for y in ys}
    b = []
    for _ in range(2000):
        s = np.concatenate([g[y] for y in rng.choice(ys, len(ys))])
        b.append(s.mean() if len(s) else np.nan)
    return f"{d[col].mean():+.2f} ({np.nanpercentile(b, 2.5):+.2f} to {np.nanpercentile(b, 97.5):+.2f}) n={len(d)}"


def main():
    p = pd.read_parquet(OUT / "review_picks.parquet")
    s = pd.read_parquet(OUT / "review_states.parquet")
    p = p[p.year.isin(YEARS)].merge(s[["year", "team", "p_need", "status"]], on=["year", "team"], how="left")
    R.REALIZED_QB = "rule"  # with qb_from None -> 99: a QB contributes nothing in the window (he sat)
    real = R.RealizedV2(Context())
    sat_sit = lambda r: {"start": r.start, "shift": eval(r.shift) if isinstance(r.shift, str) else r.shift, "qb_from": None}
    st = {(r.year, r.team): sat_sit(r) for r in s.itertuples()}
    m = (p.rec_grp == "QB") & (p.p_need < 1) & p.real_model.notna()
    # the saved review is scored fairly since Nov 13 2026 (review.REALIZED_REC): recompute "as played" here
    real_sit = {(r.year, r.team): {**sat_sit(r), "qb_from": None if pd.isna(r.qb_from) else int(r.qb_from)} for r in s.itertuples()}
    p.loc[m, "real_model"] = [real.titles(real_sit[(y, t)], "QB", y, a, k, n, qb="played") for y, t, a, k, n in
                              zip(p[m].year, p[m].team, p[m].rec_drafted_at, p[m].pick, p[m].rec_player)]
    p["real_model_sat"] = np.nan
    p.loc[m, "real_model_sat"] = [real.titles(st[(y, t)], "QB", y, a, k, n) for y, t, a, k, n in
                                  zip(p[m].year, p[m].team, p[m].rec_drafted_at, p[m].pick, p[m].rec_player)]
    p["real_model_fair"] = np.where(m, p.p_need * p.real_model + (1 - p.p_need) * p.real_model_sat, p.real_model)
    print(f"QB recs with uncertain need, rescored: {m.sum()} picks; as played {p.loc[m, 'real_model'].mean():+.2f} -> "
          f"sat {p.loc[m, 'real_model_sat'].mean():+.2f} -> fair {p.loc[m, 'real_model_fair'].mean():+.2f}")

    print("\nModel minus team where they disagree, each rec once (titles x100, 95% over draft years):")
    for lab, col in [("as played (current yardstick)", "real_model"), ("fair (p_need-weighted)", "real_model_fair")]:
        d = p[p.real_team.notna()].copy()
        d["real_model"] = d[col]
        d["d_t"] = d.real_model - d.real_team
        for how in ("first", "mean"):
            once = R.once_per_draft(d, how=how)
            row = "  ".join(f"{gl} {boot_years(once[once.band == gl], 'd_t')}" for gl in ["R1", "R2-4"])
            print(f"  {lab:30s} {how:5s} {row}")
        once = R.once_per_draft(d, how="mean")
        r1 = once[once.band == "R1"]
        print(f"  {'':30s} R1 split: QB recs {r1[r1.rec_grp == 'QB'].d_t.mean():+.2f} (n {(r1.rec_grp == 'QB').sum()}), "
              f"non-QB recs {r1[r1.rec_grp != 'QB'].d_t.mean():+.2f} (n {(r1.rec_grp != 'QB').sum()})")

    print("\nModel's QB recommendations, delivered / predicted titles (each rec once per draft and band, mean):")
    q = p[(p.rec_grp == "QB") & p.real_model.notna() & (p.player != p.rec_player)].copy()
    q["pred"] = q.value_rec
    q["band"] = R.band_of(q["round"])
    for gl in ["R1", "R2-4"]:
        g = q[q.band == gl].groupby(["year", "rec_player"])[["pred", "real_model", "real_model_fair", "p_need"]].mean()
        print(f"  {gl}: n {len(g)}, predicted {g.pred.mean():+.2f}, as played {g.real_model.mean():+.2f} "
              f"({g.real_model.sum() / g.pred.sum():.2f}x), fair {g.real_model_fair.mean():+.2f} "
              f"({g.real_model_fair.sum() / g.pred.sum():.2f}x), mean p_need {g.p_need.mean():.2f}")
    p.to_parquet(OUT / "qb_need_yardstick.parquet")


if __name__ == "__main__":
    main()
