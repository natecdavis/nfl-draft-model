"""Hybrid position pricing: the family-units (OL-friendly) choice in rounds 2-4 where the two pricings disagree.

"Position pricing robustness" (scripts/units_robust.py): where single-units (production) and family-units
recommendations disagree in rounds 2-4 (~40% of picks; mostly EDGE / WR / TE vs OL), titles tie and the family
recommendations did better on starts, AV and second-contract pay. Round 1 flips were worth ~0 and the round-1 edge
over teams sits in recommendations that hold under both. Rule tested here:

  hybrid   round 1 and rounds 5-7: the production recommendation; rounds 2-4: the family-units recommendation
           where it differs from production.

Both reviews leave the draft being valued out of the outcome engine (production review_picks.parquet; family:
OUTCOME_UNITS=family scripts/holdout_review.py, rerun Nov 16 2026 with the QB-need contract split so the two differ
only in pricing). Each pick has the team's real roster and earlier picks, so per-pick recommendations are the
whole-draft choices. Scored 2014-22 on both title yardsticks (single and family units, fair QB scoring) and on
seasons started, AV and second-contract pay (% of cap). Paired hybrid minus production over all picks; intervals
resample draft years.

Usage: python scripts/hybrid_units_choice.py
"""
import sys

sys.path[:0] = ["src", "scripts"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import units_robust as ur  # noqa: E402
from outlook import OUT  # noqa: E402

ur.CACHE = OUT / "hybrid_units.parquet"
COLS = {"titles (single)": ("real_model_single", "real_model_single_f", "real_team_single"),
        "titles (family)": ("real_model_family", "real_model_family_f", "real_team_family"),
        "starts": ("gs_model", "gs_model_f", "gs_team"), "AV": ("av_model", "av_model_f", "av_team"),
        "2nd-contract pay": ("pay_model", "pay_model_f", "pay_team")}


def boot(x, years, rng, reps=2000):
    x, years = np.asarray(x, float), np.asarray(years)
    ys = np.unique(years)
    by = {y: x[years == y] for y in ys}
    b = []
    for _ in range(reps):
        s = rng.choice(ys, len(ys))
        v = np.concatenate([by[y] for y in s])
        b.append(np.nanmean(v))
    return f"{np.nanmean(x):+.3f} ({np.nanpercentile(b, 2.5):+.3f} to {np.nanpercentile(b, 97.5):+.3f})"


def main():
    rng = np.random.default_rng(0)
    m = ur.load()
    m = m[m.grp.ne("SPEC")].copy()
    m["band"] = pd.cut(m["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    sw = (m.band == "R2-4") & (m.rec_player_f != m.rec_player)
    m["switch"] = sw
    m["hyb_player"] = np.where(sw, m.rec_player_f, m.rec_player)
    m["hyb_grp"] = np.where(sw, m.rec_grp_f, m.rec_grp)
    print(f"picks 2014-22: {len(m)}; hybrid changes {sw.sum()} ({sw.mean() * 100:.0f}% of all picks, "
          f"{sw[m.band == 'R2-4'].mean() * 100:.0f}% of rounds 2-4)")
    print("rounds 2-4 recommendations by group, production -> hybrid:")
    r24 = m[m.band == "R2-4"]
    print(pd.DataFrame({"production": r24.rec_grp.value_counts(), "hybrid": r24.hyb_grp.value_counts()}).fillna(0).astype(int).T.to_string())
    print("\nHybrid minus production, paired over all picks (2014-22), and over the switched picks only:")
    for k, (a, b, _) in COLS.items():
        d = np.where(m.switch, m[b] - m[a], 0.0)
        parts = [f"{band}: {boot(d[m.band == band], m.year[m.band == band], rng)}" for band in ["R2-4"]]
        s = m[m.switch]
        print(f"  {k:17s} all picks {boot(d, m.year, rng)} | {parts[0]} | switched picks ({len(s)}): {boot(s[b] - s[a], s.year, rng)}")
    print("\nRecommendation minus the team's pick where they disagree (each rec once per draft, mean over its picks):")
    for who, pc, sfx in [("production", "rec_player", ""), ("hybrid", "hyb_player", "")]:
        line = []
        for k, (a, b, t) in COLS.items():
            val = np.where((who == "hybrid") & m.switch, m[b], m[a])
            x = m.assign(v=val - m[t])
            x = x[x[pc] != x.player].groupby(["year", "band", pc], observed=True).agg(v=("v", "mean")).reset_index()
            x24 = x[x.band == "R2-4"]
            line.append(f"{k} R2-4 {boot(x24.v, x24.year, rng)}")
        print(f"  {who:10s} " + " | ".join(line))


if __name__ == "__main__":
    main()
