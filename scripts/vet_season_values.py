"""Season values for every player-season with snaps (2013-2025), veterans included, priced like the draft label.

Same harness as scripts/darko_label.py (cfdarko's build_label, read only; darko quality; narrow pick correction):
build_label is patched to stop right after the veteran pay curves are fit and to price every (player, season) in
the snap table instead of only draftees' seasons 1-4:
  value = price_g(snap share, q) x games fraction      (cfdarko units: APY share of cap the market pays that role)
  q     = the label's quality (darko talent percentile within pricing group x season; neutral when unmeasured)
  g     = pricing group from the player's latest OTC position_group
Special-teams-only seasons (st share >= 0.25, no scrimmage snaps) get the position's replacement level, as in the label.
Rookie-deal seasons of drafted players (2011+ classes) use the narrow pick-adjusted share, as in the label; veteran
seasons are untouched by that correction.

Writes data/derived/vet_season_values.parquet: pfr_id, gsis, season, group, share, gfrac, q, value (cfdarko units)
and value_ours (/ outcomes_v2.CF_OVER_OURS_ALL, this project's units, % of cap as a fraction).
Run: .venv/bin/python scripts/vet_season_values.py
"""
import inspect
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import darko_label as dl  # noqa: E402
from outcomes_v2 import CF_OVER_OURS_ALL  # noqa: E402

rv = dl.rv
OUT = Path(__file__).resolve().parents[1] / "data/derived/vet_season_values.parquet"


class Done(Exception):
    pass


def dump(curves, share, gfrac, st, pos, pfr2gsis, qv, repl):
    rows = []
    keys = set(share) | {k for k, v in st.items() if v >= rv.ST_MIN}
    for pfr, y in keys:
        if y < rv.SNAP_FIRST:
            continue
        gs = pfr2gsis.get(pfr)
        g = pos.get(gs) if gs is not None else None
        if g not in curves:
            continue
        sh = share.get((pfr, y), 0.0)
        if sh > 0:
            q = qv(gs, g, y)
            v = curves[g](sh, q) * gfrac[(pfr, y)]
        else:
            q, v = float("nan"), float(repl.get(g, 0.0))
        rows.append((pfr, gs, y, g, sh, gfrac.get((pfr, y), 0.0), q, v))
    d = pd.DataFrame(rows, columns=["pfr_id", "gsis", "season", "group", "share", "gfrac", "q", "value"])
    d["value_ours"] = d.value / CF_OVER_OURS_ALL
    d.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(d):,} player-seasons, {d.pfr_id.nunique():,} players")
    print(d.groupby("group").value_ours.describe().round(4).to_string())
    raise Done


def patched_build_label():
    src = textwrap.dedent(inspect.getsource(rv.build_label))
    anchor = "        curves = pay_curves_wide(s, pfr2gsis, c, qv, qgroups, args.xval_fold, args.xval_k)\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + "        _VET_DUMP(curves, share, gfrac, st, pos, pfr2gsis, qv, repl)\n")
    rv._VET_DUMP = dump
    exec(compile(src, rv.__file__, "exec"), rv.__dict__)


def main():
    patched_build_label()
    dl.go._quality = dl.darko_quality
    dl.pa.OUT = OUT
    try:
        dl.pa.main()
    except Done:
        pass


if __name__ == "__main__":
    main()
