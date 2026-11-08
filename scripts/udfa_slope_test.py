"""Does the pick-attributable playing-time slope continue past the draft? (Decides whether the narrow label's
year 2-4 share shift, scripts/pickadj_label.py, should also apply to undrafted players.)

Same data and controls as the narrow fit (years 2-4 of a career, seasons with snap share >= 0.2 and a measured
PFF quality q, 2013-2025), now with undrafted players added:
  share ~ D*log(pick+10) + D*QB*log(pick+10) + U + q + q^2 + prior share + prior q (+ flag)
          + career year + pricing group + season
  D = drafted, U = undrafted indicator (its coefficient is the undrafted level, on the same scale as b*log(pick+10)).
Specified before running:
  on the line   U - b*log(310)   (undrafted where the drafted line, extended to pick 300, puts them)
  flat          U - b*log(272)   (undrafted where the last pick of the draft, ~262, sits)
  rule          extend the shift to undrafted players only if they play clearly less than the last picks
                ("flat" difference < 0 with 95% interval excluding 0) and are consistent with the line ("on the
                line" interval includes 0). Otherwise leave them unchanged.
Runs cfdarko's build_label only far enough to get the label's own quality table, then stops (nothing written).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_only_label as go  # noqa: E402

rv = go.rv


class Done(Exception):
    pass


def test():
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
    s["D"] = s.pfr_player_id.map(drafted).astype(float)
    s["pick"] = s.pfr_player_id.map(bp.overall_pick.where(drafted))
    prev = s.set_index(["pfr_player_id", "season"]).share0
    s["prior"] = [prev.get((p, y - 1), 0.0) for p, y in zip(s.pfr_player_id, s.season)]
    qprev = {(p, y): (qq if sh >= 0.2 else np.nan) for p, y, qq, sh in zip(s.pfr_player_id, s.season, s.q, s.share0)}
    s["prior_q"] = [qprev.get((p, y - 1), np.nan) for p, y in zip(s.pfr_player_id, s.season)]
    s["no_prior_q"] = s.prior_q.isna().astype(float)
    s["prior_q"] = s.prior_q.fillna(0.0)
    F = s[s.D.notna() & s.yr.between(1, 3) & (s.share0 >= 0.2) & s.g.notna() & s.q.notna()].copy()
    F["lpick"] = np.where(F.D == 1, np.log(F.pick + 10), 0.0)
    F["lpick_qb"] = F.lpick * (F.g == "QB")
    F["U"] = 1.0 - F.D
    F["q2"] = F.q ** 2
    X = pd.concat([F[["lpick", "lpick_qb", "U", "q", "q2", "prior", "prior_q", "no_prior_q"]],
                   pd.get_dummies(F.yr.astype(int), prefix="yr", drop_first=True, dtype=float),
                   pd.get_dummies(F.g, prefix="g", drop_first=True, dtype=float),
                   pd.get_dummies(F.season, prefix="s", drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(F.share0, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": F.pfr_player_id})
    V = m.cov_params()
    print(f"\nUNDRAFTED vs THE PICK LINE (years 2-4, share >= 0.2, measured quality; {len(F)} seasons, "
          f"{int(F.U.sum())} undrafted seasons from {F[F.U == 1].pfr_player_id.nunique()} players)")
    print(f"  drafted slope b = {m.params.lpick:+.4f} per log(pick+10) (se {m.bse.lpick:.4f}); "
          f"undrafted level U = {m.params.U:+.4f} (se {m.bse.U:.4f})")
    for nm, pk in [("flat (vs last pick, ~262)", 262), ("on the line (pick 300)", 300)]:
        c = pd.Series(0.0, index=m.params.index)
        c["U"], c["lpick"] = 1.0, -np.log(pk + 10)
        est = float(c @ m.params)
        se = float(np.sqrt(c @ V @ c))
        print(f"  {nm:28s} undrafted minus line at pick {pk}: {est:+.4f} share (95% {est - 1.96 * se:+.4f} to "
              f"{est + 1.96 * se:+.4f})")
    for pk in (100, 200, 262):
        print(f"    (line at pick {pk}: {m.params.lpick * np.log(pk + 10):+.4f})")
    raise Done


def main():
    rv.quality_table_pff = go.quality_capture
    rv.load_snaps = test
    try:
        rv.build_label(Path("/dev/null"))
    except Done:
        pass


if __name__ == "__main__":
    main()
