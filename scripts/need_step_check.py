"""Within-draft need step: old rule (x0.6 / x0.3 of years 1-2 playing time) vs measured (x0.95 / x0.88, all
seasons; draftday.NEED_STEP). Compares the production review before (data/derived/before_need_step/) and after
the rebuild: model minus team where they disagree (each rec once, as the Past drafts scorecard), drafts 2014-22,
how many recommendations changed, and the position mix of recommendations.

Run:  python scripts/need_step_check.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402
from review import Careers, once_per_draft  # noqa: E402

YEARS = range(2014, 2023)
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f}) n={len(x)}"


def main():
    old = pd.read_parquet(OUT / "before_need_step" / "review_picks.parquet")
    new = pd.read_parquet(OUT / "review_picks.parquet")
    car = Careers()
    print("Model minus team where they disagree, each rec once, drafts 2014-22 (titles x100; starts; AV)")
    for lab, d in [("old rule (x0.6 / x0.3 yrs 1-2)", old), ("measured (x0.95 / x0.88)", new)]:
        d = d[d.year.isin(YEARS) & d.real_team.notna()].copy()
        d["d_t"] = d.real_model - d.real_team
        for m in ("gs", "av"):
            d[f"d_{m}"] = [car.get(m, y, a, rn) - car.get(m, y, k, tn) for y, a, rn, k, tn in
                           zip(d.year, d.rec_drafted_at, d.rec_player, d.pick, d.player)]
        once = once_per_draft(d)
        print(f"  {lab}")
        for gl in ["R1", "R2-4", "R5-7"]:
            o = once[once.band == gl]
            print(f"    {gl:5s} titles {boot(o.d_t)} | starts {o.d_gs.mean():+.2f} | AV {o.d_av.mean():+.1f}")
    m = old.merge(new, on=["year", "pick"], suffixes=("_o", "_n"))
    m = m[m.year.isin(YEARS)]
    ch = m.rec_player_o != m.rec_player_n
    print(f"\nRecommendation changed at {ch.mean():.0%} of picks; by round:")
    print(m.assign(ch=ch).groupby("round_o").ch.mean().round(2).to_string())
    both = m[m.real_model_o.notna() & m.real_model_n.notna()]
    print(f"Paired, all picks (new minus old realized titles x100, repeat recs included): {boot(both.real_model_n - both.real_model_o)}")
    print("\nPosition mix of recommendations, rounds 2-4 (old -> new):")
    r = m[m.round_o.between(2, 4)]
    mix = pd.concat([r.rec_grp_o.value_counts(normalize=True), r.rec_grp_n.value_counts(normalize=True)], axis=1).fillna(0)
    print((mix * 100).round(0).set_axis(["old %", "new %"], axis=1).to_string())
    # how often a team gets 2+ recs at one group in a draft (the rule's direct target)
    for lab, s in [("old", "rec_grp_o"), ("new", "rec_grp_n")]:
        c = m.groupby(["year", "team_o", s]).size()
        print(f"{lab}: team-drafts x group with 2+ recommendations: {(c >= 2).mean():.0%}, 3+: {(c >= 3).mean():.0%}")


if __name__ == "__main__":
    main()
