"""NFL coaching staffs 2025-2026 (Wikipedia) and the careers of coaches not already cached, so the 2025 and 2026
draft classes can be flagged for coaching connections (scripts/connected_fantasy_test.py, rookies_2025_26.py).

Same sources and parsers as src/coaching.py and scripts/staffs_pre2012.py. Separate files, earlier data unchanged:
  data/derived/nfl_staffs_2025.parquet    staffs 2025-2026
  data/derived/coach_careers_2025.parquet careers of coaches not in coach_careers(.|_2006).parquet

2025: "2025 {team} season" pages (season finished). 2026: the season page transcludes "Template:{team} staff", the
staff as of the fetch (Nov 2026); used as the 2026 draft-day staff (in-season changes after April are rare).
Any season page whose staff doesn't parse falls back to the template only for 2026.
"""
import re
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
from staffs_pre2012 import staff_text

ABBRS = sorted(set(co.franchise(a) for a in co.TEAMS) | {"WAS"})
TEMPLATE_NAME = {"LAR": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LVR": "Las Vegas Raiders"}


def parse_template(text):
    """'Template:{Team} staff': ';Section' or bold headings, '* Role – [[Name]]' lines."""
    rows, sec = [], None
    for line in (text or "").split("\n"):
        line = line.strip()
        h = re.fullmatch(r";\s*(.+)|'''([^']+)'''", line)  # ';Section' or bold heading
        if h:
            sec = (h.group(1) or h.group(2)).strip()
            continue
        if not line.startswith("*") or not sec:
            continue
        if not re.search(r"coach", sec, re.I) or re.search(r"strength", sec, re.I):
            continue
        body = line.lstrip("* ").strip()
        role, _, who = body.rpartition("–") if "–" in body else body.rpartition("-")
        lk = co.LINK.search(who)
        rows.append({"section": sec, "role": role.strip(),
                     "name": (lk.group(2) or lk.group(1)).strip() if lk else re.sub(r"[\[\]']", "", who).strip(),
                     "page": lk.group(1).strip() if lk else None})
    return rows


def main():
    rows = []
    titles = {(a, y): f"{y} {co.team_name(a, y) if a not in TEMPLATE_NAME else TEMPLATE_NAME[a]} season"
              for a in ABBRS for y in (2025, 2026)}
    texts = co.fetch(list(titles.values()), "seasons")
    tmpl = {a: f"Template:{TEMPLATE_NAME.get(a) or co.team_name(a, 2026)} staff" for a in ABBRS}
    ttext = co.fetch(list(tmpl.values()), "templates")
    for (a, y), t in titles.items():
        r = co.parse_staff(staff_text(texts.get(t)))
        src = "season page"
        if not r and y == 2026:
            r, src = parse_template(ttext.get(tmpl[a])), "template"
        if not r:
            print(f"  no staff parsed: {t}")
        rows += [{"fr": a, "year": y, "src": src, **x} for x in r]
    st = pd.DataFrame(rows)
    print(f"staffs 2025-26: {len(st)} rows; teams per year {st.groupby('year').fr.nunique().to_dict()}; "
          f"rows per team-year {st.groupby(['fr', 'year']).size().median():.0f}; linked {st.page.notna().mean():.0%}; "
          f"source {st.drop_duplicates(['fr', 'year']).groupby(['year', 'src']).size().to_dict()}")
    st.drop(columns="src").to_parquet(co.OUT / "nfl_staffs_2025.parquet")

    have = set(pd.read_parquet(co.OUT / "coach_careers.parquet").page) | \
        set(pd.read_parquet(co.OUT / "coach_careers_2006.parquet").page)
    new = sorted(set(st.page.dropna()) - have)
    print(f"career pages to fetch: {len(new)}")
    ct = co.fetch(new, "coaches")
    car = pd.DataFrame([{"page": p, **x} for p in new for x in co.parse_career(ct.get(p))])
    print(f"careers: {len(car)} rows from {car.page.nunique()} pages")
    car.to_parquet(co.OUT / "coach_careers_2025.parquet")


if __name__ == "__main__":
    main()
