"""Dane Brugler's draft guide ("The Beast", The Athletic), 2019-2023: one row per player.

Text comes from the user's own downloaded guides (data/raw/beast/, extracted once to
data/raw/beast/text/{year}.txt); it stays local (data/ is not in git) and only derived
numbers feed the model.

Per profile: position-group rank, name, school, height, weight, class, age, the guide's
position section, its BACKGROUND / STRENGTHS / WEAKNESSES / SUMMARY text, and the GRADE
(round, plus overall rank for the top ~100). Later-ranked players have a short SUMMARY only.

Run:  python src/beast.py   -> data/derived/beast_profiles.parquet
"""
import re

import pandas as pd

from outlook import OUT, RAW

TEXT = RAW / "beast" / "text"
YEARS = [2019, 2020, 2021, 2022, 2023]
HEADER = re.compile(r"(?m)^\s*(\d{1,3})\.\s*\n?\s*([A-Z][A-Za-z'’.\- ]+?(?: (?:JR|SR|II|III|IV)\.?)?) \| (.+?) {2,}(\d{4}) \| (\d{3}) lbs\. \| (\S+)"
                    r"(.*?)\(age (\d+\.\d+)\)", re.S)
# position-section headings, e.g. "OFFENSIVE GUARDS", "DEFENSIVE LINEMEN", "TIGHT ENDS/FULLBACKS"
SEC_RE = re.compile(r"(?m)^[ \t]*((?:OFFENSIVE |DEFENSIVE |INTERIOR )?(?:QUARTERBACK|RUNNING BACK|WIDE RECEIVER|TIGHT END|"
                    r"TACKLE|GUARD|CENTER|LINEM[AE]N|LINE|EDGE|LINEBACKER|CORNERBACK|SAFET|SPECIALIST)[A-Z/ ]*?)[ \t]*$")


def section(line: str) -> str:
    l = line.upper()
    for key, grp in [("QUARTERBACK", "QB"), ("RUNNING BACK", "RB"), ("WIDE RECEIVER", "WR"), ("TIGHT END", "TE"),
                     ("LINEBACKER", "LB"), ("CORNERBACK", "CB"), ("SAFET", "S"), ("SPECIALIST", "SPEC"),
                     ("EDGE", "ED"), ("OFFENSIVE TACKLE", "T"), ("GUARD", "G"), ("CENTER", "C"),
                     ("DEFENSIVE TACKLE", "IDL"), ("DEFENSIVE LINE", "IDL"), ("INTERIOR", "IDL"), ("TACKLE", "T")]:
        if key in l:
            return grp
    return None


PARTS = ["BACKGROUND", "STRENGTHS", "WEAKNESSES", "SUMMARY", "GRADE"]
ROUND = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7}


def _parts(body: str) -> dict:
    out, idx = {}, sorted((m.start(), m.group(1)) for m in re.finditer(r"\b(" + "|".join(PARTS) + r"):", body))
    for (i, name), nxt in zip(idx, idx[1:] + [(len(body), None)]):
        if name not in out:
            out[name] = re.sub(r"\s+", " ", body[i + len(name) + 1:nxt[0]]).strip()
    return out


def _grade(g: str):
    """'1st-2nd Round (#37 overall)' -> (1.5, 37); 'Priority Free Agent' -> (8, None)."""
    if not g:
        return None, None
    rounds = [ROUND[r] for r in re.findall(r"(\dst|\dnd|\drd|\dth)", g) if r in ROUND]
    rnd = sum(rounds) / len(rounds) if rounds else (8.0 if "Free Agent" in g else None)
    ov = re.search(r"(?:#|No\. )(\d+) overall", g)
    return rnd, int(ov.group(1)) if ov else None


def parse(year: int) -> pd.DataFrame:
    t = open(TEXT / f"{year}.txt").read()
    secs = [(m.start(), section(m.group(1))) for m in SEC_RE.finditer(t) if section(m.group(1))]
    heads = list(HEADER.finditer(t))
    rows = []
    for h, nxt in zip(heads, heads[1:] + [None]):
        body = t[h.end(): nxt.start() if nxt else len(t)]
        p = _parts(body)
        sec = [s for i, s in secs if i < h.start()]
        rnd, ov = _grade(p.get("GRADE", "")[:80])
        rows.append({"year": year, "pos_rank": int(h.group(1)), "name": h.group(2).strip().title(),
                     "school": h.group(3).strip(), "height": h.group(4), "weight": int(h.group(5)),
                     "cls": h.group(6), "age": float(h.group(8)), "section": sec[-1] if sec else None,
                     "background": p.get("BACKGROUND"), "strengths": p.get("STRENGTHS"),
                     "weaknesses": p.get("WEAKNESSES"), "summary": p.get("SUMMARY"),
                     "grade_text": (p.get("GRADE") or "")[:80], "grade_round": rnd, "overall_rank": ov})
    return pd.DataFrame(rows)


def main():
    df = pd.concat([parse(y) for y in YEARS], ignore_index=True)
    df.to_parquet(OUT / "beast_profiles.parquet")
    for y, g in df.groupby("year"):
        print(y, len(g), "profiles |", g.strengths.notna().sum(), "full |", g.summary.notna().sum(), "summary |",
              g.grade_round.notna().sum(), "graded |", g.overall_rank.notna().sum(), "overall-ranked |",
              g.section.value_counts().to_dict())
    return df


if __name__ == "__main__":
    main()
