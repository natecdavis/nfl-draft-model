"""NFL coaching staffs 2006-2011 (Wikipedia season pages) and the careers of coaches not already cached,
for the out-of-sample replication on older classes (scripts/old_class_replication.py).

Same sources and parsers as src/coaching.py. Written to separate files so the 2012-2024 data used by the
original tests (nfl_staffs.parquet, coach_careers.parquet) stay unchanged:
  data/derived/nfl_staffs_2006.parquet   staffs 2006-2011
  data/derived/coach_careers_2006.parquet  careers of coaches who appear only in 2006-2011 staffs
"""
import re
import sys

import pandas as pd

sys.path.insert(0, "src")
import coaching as co

YEARS = range(2006, 2012)
OLD = {"LAR": "STL", "LAC": "SDG", "LVR": "OAK"}


def staff_text(t):
    """Older pages head the staff table '==Coaching staff==' (or a level-3 heading): map it to the form
    co.parse_staff reads."""
    if not t or "{{NFL final staff" in t:
        return t
    return re.sub(r"\n=+\s*Coaching staff\s*=+[ \t]*\n", "\n==Staff==\n", t, flags=re.I)


def main():
    abbrs = [a for a in co.TEAMS if a not in OLD] + list(OLD.values()) + ["WAS"]
    abbrs = sorted(set(a for a in abbrs if a not in OLD))
    titles = {(a, y): f"{y} {co.team_name(a, y)} season" for a in abbrs for y in YEARS}
    texts = co.fetch(list(titles.values()), "seasons")
    rows = []
    for (a, y), t in titles.items():
        r = co.parse_staff(staff_text(texts.get(t)))
        if not r:
            print(f"  no staff parsed: {t}")
        rows += [{"fr": co.franchise(a), "year": y, **x} for x in r]
    st = pd.DataFrame(rows)
    print(f"staffs 2006-11: {len(st)} rows, {st.groupby('year').fr.nunique().to_dict()} teams per year, "
          f"{st.page.notna().mean():.0%} linked")
    st.to_parquet(co.OUT / "nfl_staffs_2006.parquet")

    have = set(pd.read_parquet(co.OUT / "coach_careers.parquet").page)
    new = sorted(set(st.page.dropna()) - have)
    print(f"career pages to fetch: {len(new)}")
    ct = co.fetch(new, "coaches")
    car = pd.DataFrame([{"page": p, **x} for p in new for x in co.parse_career(ct.get(p))])
    print(f"careers: {len(car)} rows from {car.page.nunique()} pages")
    car.to_parquet(co.OUT / "coach_careers_2006.parquet")


if __name__ == "__main__":
    main()
