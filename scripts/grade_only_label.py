"""Grade-only rookie value label: price each rookie-deal season on the snap share the player's quality earns,
not the share his team gave him (follow-up to scripts/sunk_cost_test.py).

Runs cfdarko's own label builder (src/models/rookie_value.build_label, read only; output written here, never
to ~/cfdarko) with one input changed. Everything else is identical: veteran pay curves price(share, quality),
quality q = within-(position x season) percentile of the shrunk PFF grade, games fraction, rookie costs, the
depth-chart model for pre-2013 seasons (refit to the new season values).

Change: for seasons in years 1-4 of a player's career (season - draft year 0..3) where he played a measurable
role (snap share >= THRESH), share is replaced by share_hat_g(q) = E[share | q] in his pricing group, fit
(isotonic, increasing) on VETERAN seasons (5th year on, share >= THRESH), whose playing time is not tied to a
draft investment. Seasons below THRESH keep their (small) actual value: quality isn't measured there.
Veteran seasons are untouched, so the pay curves (fit on veteran contracts) don't move.

Usage: python scripts/grade_only_label.py [THRESH]   (default 0.2; writes data/derived/labels_gradeonly_t{20}.parquet)
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CF = Path.home() / "cfdarko"
sys.path.insert(0, str(CF))
# cfdarko imports nfl_data_py (not installed here) to fetch caches that already exist; the one live call it makes,
# import_ids(), is served from the cached copy of that same call (data/raw/rookie_value/nflverse_ids.parquet)
_nfl = types.ModuleType("nfl_data_py")
_nfl.import_ids = lambda: pd.read_parquet(CF / "data/raw/rookie_value/nflverse_ids.parquet")
sys.modules.setdefault("nfl_data_py", _nfl)
from src.models import rookie_value as rv  # noqa: E402

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
OUT = Path(__file__).resolve().parents[1] / "data/derived" / f"labels_gradeonly_t{round(THRESH * 100)}.parquet"
CAP = {}


def crosswalk():
    ids = pd.read_parquet(rv.CACHE / "nflverse_ids.parquet")
    pl = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/raw/players.parquet", columns=["pfr_id", "gsis_id"])
    x = pd.concat([pl.dropna(), ids.dropna()]).drop_duplicates("pfr_id")
    return dict(zip(x.pfr_id, x.gsis_id))


_quality = rv.quality_table_pff
_load_snaps = rv.load_snaps


def quality_capture(pos):
    q, neutral = _quality(pos)
    CAP.update(q=q, pos=pos)
    return q, neutral


def snaps_gradeonly():
    s = _load_snaps()
    q, pos = CAP["q"], CAP["pos"]
    p2g = crosswalk()
    bp = pd.read_parquet(rv.PROCESSED_DIR / "bridge_players.parquet", columns=["pfr_id", "draft_year"]).dropna()
    dy = bp.drop_duplicates("pfr_id").set_index("pfr_id").draft_year.astype(int)
    s["share0"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["gsis"] = s.pfr_player_id.map(p2g)
    s["g"] = s.gsis.map(pos)
    s["q"] = [q.get((a, y), np.nan) for a, y in zip(s.gsis, s.season)]
    s["yr"] = s.season - s.pfr_player_id.map(dy)
    ok = s.q.notna() & s.g.notna() & (s.share0 >= THRESH)
    vet = s[ok & (s.yr >= 4)]
    rook = ok & s.yr.between(0, 3)
    s["new"] = s.share0
    stats = {}
    for g, G in vet.groupby("g"):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(G.q, G.share0)
        m = rook & (s.g == g)
        s.loc[m, "new"] = iso.predict(s.loc[m, "q"])
        stats[g] = (len(G), int(m.sum()), round(float(iso.predict([0.1])[0]), 2), round(float(iso.predict([0.5])[0]), 2),
                    round(float(iso.predict([0.9])[0]), 2))
    print(f"\nGRADE-ONLY SHARES (threshold {THRESH}): group: veteran fit seasons, rookie seasons replaced, "
          f"share_hat at q=0.1/0.5/0.9")
    for g, v in stats.items():
        print(f"  {g:5s} {v}")
    r = s[rook]
    print(f"  rookie seasons replaced: {int(rook.sum())}; mean share actual {r.share0.mean():.3f} -> {r.new.mean():.3f}; "
          f"corr {r.share0.corr(r.new):.2f}")
    # build_label takes share = max(offense_pct, defense_pct): write the new share to both sides
    for c in ("offense_pct", "defense_pct"):
        s[c] = np.where(rook, s.new, s[c])
    return s.drop(columns=["share0", "gsis", "g", "q", "yr", "new"])


def main():
    rv.quality_table_pff = quality_capture
    rv.load_snaps = snaps_gradeonly
    rv.build_label(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
