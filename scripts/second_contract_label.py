"""Seasons 5-7 at the same price as the rookie-deal label (complaint #5: value after the rookie deal).

Same harness as scripts/darko_label.py (cfdarko's build_label, read only, narrow pick-attributable correction,
darko quality): build_label's season loop is patched to also record v5, v6, v7 = the value of the player's
seasons 5-7 (snap share x pay curve x games, same as value_y1-4), whatever his contract. value_y1..y5 and
everything else are unchanged (checked against labels_pickadj_narrow_darko).

Writes data/derived/labels_seasons5_7.parquet. Run: .venv/bin/python scripts/second_contract_label.py
"""
import inspect
import sys
import textwrap
from pathlib import Path

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import darko_label as dl  # noqa: E402

rv = dl.rv
OUT = Path(__file__).resolve().parents[1] / "data/derived/labels_seasons5_7.parquet"


def patched_build_label():
    src = textwrap.dedent(inspect.getsource(rv.build_label))
    anchor = '        rec["value_src"] = ('
    assert src.count(anchor) == 1
    extra = ('        for k in (5, 6, 7):\n'
             '            y = dy + k - 1\n'
             '            rec[f"v{k}"] = val(r, y) if FIRST_CLASS <= y <= LAST_SEASON else np.nan\n')
    src = src.replace(anchor, extra + anchor)
    exec(compile(src, rv.__file__, "exec"), rv.__dict__)


def main():
    patched_build_label()
    dl.go._quality = dl.darko_quality
    dl.pa.OUT = OUT
    dl.pa.main()


if __name__ == "__main__":
    main()
