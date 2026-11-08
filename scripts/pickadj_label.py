"""Pick-attributable correction to the rookie value label: remove only the playing time that draft position
explains at the same quality (follow-up to scripts/label_bracket.py, where the grade-only lower bound moved pick
values too far to leave the labels alone).

Same harness as scripts/grade_only_label.py (cfdarko's build_label, read only; with nothing changed it reproduces
the stored label exactly). Specified before fitting:
  fit     drafted players' rookie-deal seasons (years 1-4) with snaps > 0 and a PFF quality q, 2013-2025:
          share ~ log(pick+10) + log(pick+10) x QB + q + q^2 + prior-season share + first-season flag
                  + career year + pricing group + season   (OLS)
          q = the label's own quality (within position x season percentile of the shrunk PFF grade, same season)
  adjust  share' = share + b_g * (log(REF+10) - log(pick+10)), clipped to [0, 1]; b_g = b (+ QB term for QBs);
          REF = 100 (a mid-round pick). Each player keeps his own residual: only the pick-explained part moves.
  scope   drafted players' rookie-deal seasons with share > 0; zero-snap seasons, undrafted players and veteran
          seasons unchanged (pay curves are fit on veterans, so they don't move).
  caveat  b also carries what teams know beyond PFF grades, so this is still an UPPER bound on the correction
          (the sunk-cost test can't separate the two); it is narrower than the grade-only bound, which replaced
          all team discretion.
Writes data/derived/labels_pickadj.parquet.

RESULT OF THAT SPEC ("pre", kept for the record): b = -0.117 per log(pick+10) (QB -0.080), shifting a #1 pick's
share by -0.27, more than the grade-only lower bound took off many top picks. The quality control failed: q
entered U-shaped (q -0.86, q^2 +0.92; d share / d q ~ 0 at the median), because the label's q is a percentile of
a grade shrunk toward the group mean by snaps (200-snap prior), so low-snap players sit near neutral q. With
quality badly controlled, pick soaks up ability. Revised spec ("measured", the default), mirroring
scripts/sunk_cost_test.py part B (the basis of the <= ~15% bound): fit only seasons where quality is measured
(share >= 0.2), add the prior season's q (0 + flag when he had no measured prior season), and adjust only
seasons with share >= 0.2 (as the grade-only label does). Result: b = -0.075 (QB -0.051); split by year, rookie
season -0.126, years 2-4 -0.049 (close to the sunk-cost test's -0.035 to -0.042). In a rookie season there is no NFL
evidence yet, so draft position is the team's scouting information and can't be told apart from sunk cost; the
pooled b leans on it. Added spec "narrow": fit and adjust years 2-4 only (the part the sunk-cost test supports);
rookie seasons unchanged. Usage: python scripts/pickadj_label.py [measured|narrow|pre]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SPEC = sys.argv[1] if len(sys.argv) > 1 else "measured"
sys.argv = sys.argv[:1]  # grade_only_label reads a threshold from argv at import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_only_label as go  # noqa: E402

rv = go.rv
REF = 100
MIN_SHARE = 0.0 if SPEC == "pre" else 0.2
FIRST_YR = 1 if SPEC == "narrow" else 0  # career year index: 0 = rookie season
OUT = Path(__file__).resolve().parents[1] / "data/derived" / f"labels_pickadj{'' if SPEC == 'measured' else '_' + SPEC}.parquet"


def snaps_pickadj():
    s = go._load_snaps()
    q, pos = go.CAP["q"], go.CAP["pos"]
    p2g = go.crosswalk()
    bp = pd.read_parquet(rv.PROCESSED_DIR / "bridge_players.parquet",
                         columns=["pfr_id", "draft_year", "overall_pick", "is_udfa"]).dropna(subset=["pfr_id", "draft_year"])
    bp = bp.drop_duplicates("pfr_id").set_index("pfr_id")
    drafted = bp.overall_pick.notna() & ~bp.is_udfa.fillna(False).astype(bool)
    s["share0"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["gsis"] = s.pfr_player_id.map(p2g)
    s["g"] = s.gsis.map(pos)
    s["q"] = [q.get((a, y), np.nan) for a, y in zip(s.gsis, s.season)]
    s["yr"] = s.season - s.pfr_player_id.map(bp.draft_year)
    s["pick"] = s.pfr_player_id.map(bp.overall_pick.where(drafted))
    s = s.sort_values(["pfr_player_id", "season"])
    prev = s.set_index(["pfr_player_id", "season"]).share0
    s["prior"] = [prev.get((p, y - 1), 0.0) for p, y in zip(s.pfr_player_id, s.season)]
    qprev = {(p, y): (qq if sh >= 0.2 else np.nan) for p, y, qq, sh in zip(s.pfr_player_id, s.season, s.q, s.share0)}
    s["prior_q"] = [qprev.get((p, y - 1), np.nan) for p, y in zip(s.pfr_player_id, s.season)]
    s["no_prior_q"] = s.prior_q.isna().astype(float)
    s["prior_q"] = s.prior_q.fillna(0.0)
    rook = s.pick.notna() & s.yr.between(FIRST_YR, 3) & (s.share0 > MIN_SHARE) & s.g.notna()
    F = s[rook & s.q.notna()].copy()
    F["lpick"] = np.log(F.pick + 10)
    F["lpick_qb"] = F.lpick * (F.g == "QB")
    F["q2"] = F.q ** 2
    F["first"] = (F.yr == 0).astype(float)
    extra = [] if SPEC == "pre" else ["prior_q", "no_prior_q"]
    X = pd.concat([F[["lpick", "lpick_qb", "q", "q2", "prior", "first"] + extra],
                   pd.get_dummies(F.yr.astype(int), prefix="yr", drop_first=True, dtype=float),
                   pd.get_dummies(F.g, prefix="g", drop_first=True, dtype=float),
                   pd.get_dummies(F.season, prefix="s", drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(F.share0, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": F.pfr_player_id})
    b, bq = m.params.lpick, m.params.lpick_qb
    for nm, sub in ([("rookie season only", F.yr == 0), ("years 2-4", F.yr >= 1)] if FIRST_YR == 0 else []):  # diagnostic
        Xs = X.loc[sub].loc[:, X.loc[sub].std() > 0]
        ms = sm.OLS(F.share0[sub], sm.add_constant(Xs)).fit(cov_type="cluster", cov_kwds={"groups": F.pfr_player_id[sub]})
        print(f"  [diagnostic] {nm}: n={int(sub.sum())}, per log(pick+10) non-QB {ms.params.lpick:+.4f} (se {ms.bse.lpick:.4f})")
    print(f"\nPICK-ATTRIBUTABLE SHARE, spec '{SPEC}' (fit on {len(F)} rookie-deal seasons, {F.pfr_player_id.nunique()} players)")
    print(f"  per log(pick+10): non-QB {b:+.4f} (se {m.bse.lpick:.4f}); QB {b + bq:+.4f} (QB term se {m.bse.lpick_qb:.4f})")
    print(f"  q {m.params.q:+.3f}, q^2 {m.params.q2:+.3f}, prior share {m.params.prior:+.3f}; R^2 {m.rsquared:.3f}")
    for pk in (1, 10, 32, 64, 100, 160, 250):
        d = np.log(REF + 10) - np.log(pk + 10)
        print(f"  pick {pk:3d}: share shift non-QB {b * d:+.3f}, QB {(b + bq) * d:+.3f}")
    slope = np.where(s.g == "QB", b + bq, b)
    shift = slope * (np.log(REF + 10) - np.log(s.pick + 10))
    s["new"] = np.where(rook, (s.share0 + shift).clip(0, 1), s.share0)
    r = s[rook]
    print(f"  seasons adjusted: {int(rook.sum())}; mean share {r.share0.mean():.3f} -> {r.new.mean():.3f}; "
          f"corr {r.share0.corr(r.new):.3f}")
    for c in ("offense_pct", "defense_pct"):
        s[c] = np.where(rook, s.new, s[c])
    return s[[c for c in go._load_snaps().columns]]


def main():
    rv.quality_table_pff = go.quality_capture
    rv.load_snaps = snaps_pickadj
    rv.build_label(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
