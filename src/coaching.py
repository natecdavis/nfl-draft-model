"""Coaching-staff connections between NFL teams and the college players they draft.

Sources (Wikipedia, CC BY-SA; raw wikitext cached under data/raw/wiki/, not in git):
  NFL staffs  - the "{{NFL final staff}}" template on each "{year} {team} season" page: every coach
                in the Head / Offensive / Defensive / Special Teams sections (strength staff and front
                office left out). The season-Y staff is the one in place for the April draft of year Y.
  Careers     - each linked coach's own page, infobox field `past_coaching`: one line per job,
                "* [[Team]] (years)<br>Role", sub-roles on "**" lines with their own years.
Coaches without their own page (mostly quality-control and assistant position coaches) are missed.

A pick is "connected" when someone on the drafting team's staff coached at the player's final
college during his college years (the 5 seasons before the draft).
"""
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/wiki")
OUT = Path("data/derived")
UA = "nfl-draft-research/0.1 (personal research script; python urllib)"
API = "https://en.wikipedia.org/w/api.php"

TEAMS = {
    "IND": "Indianapolis Colts", "CLE": "Cleveland Browns", "MIN": "Minnesota Vikings", "JAX": "Jacksonville Jaguars",
    "DAL": "Dallas Cowboys", "TAM": "Tampa Bay Buccaneers", "MIA": "Miami Dolphins", "CAR": "Carolina Panthers",
    "BUF": "Buffalo Bills", "KAN": "Kansas City Chiefs", "PHI": "Philadelphia Eagles", "ARI": "Arizona Cardinals",
    "SEA": "Seattle Seahawks", "NYJ": "New York Jets", "CIN": "Cincinnati Bengals", "CHI": "Chicago Bears",
    "TEN": "Tennessee Titans", "NWE": "New England Patriots", "DET": "Detroit Lions", "PIT": "Pittsburgh Steelers",
    "HOU": "Houston Texans", "GNB": "Green Bay Packers", "SFO": "San Francisco 49ers", "NYG": "New York Giants",
    "BAL": "Baltimore Ravens", "DEN": "Denver Broncos", "ATL": "Atlanta Falcons", "NOR": "New Orleans Saints",
    "STL": "St. Louis Rams", "LAR": "Los Angeles Rams", "SDG": "San Diego Chargers", "LAC": "Los Angeles Chargers",
    "OAK": "Oakland Raiders", "LVR": "Las Vegas Raiders",
}


def team_name(abbr: str, year: int) -> str:
    if abbr == "WAS":
        return "Washington Redskins" if year <= 2019 else "Washington Football Team" if year <= 2021 else "Washington Commanders"
    return TEAMS[abbr]


def franchise(abbr: str) -> str:
    return {"STL": "LAR", "SDG": "LAC", "OAK": "LVR"}.get(abbr, abbr)


def fetch(titles: list[str], sub: str) -> dict[str, str | None]:
    """Wikitext for each title (redirects followed), cached one JSON file per title."""
    d = RAW / sub
    d.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for t in titles:
        f = d / (re.sub(r"[^\w\-.()]+", "_", t) + ".json")
        if f.exists():
            out[t] = json.loads(f.read_text())["text"]
        else:
            todo.append(t)
    def save(t, text):
        out[t] = text
        (d / (re.sub(r"[^\w\-.()]+", "_", t) + ".json")).write_text(json.dumps({"title": t, "text": text}))

    # the API returns page text only up to a size limit per request; pages left without text
    # (and not reported missing) go back into the queue
    while todo:
        batch, todo = todo[:50], todo[50:]
        q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                    "format": "json", "formatversion": "2", "redirects": "1", "titles": "|".join(batch)})
        for attempt in range(6):
            try:
                req = urllib.request.Request(API, data=q.encode(), headers={"User-Agent": UA})
                js = json.loads(urllib.request.urlopen(req, timeout=60).read())["query"]
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    raise
                time.sleep(int(e.headers.get("Retry-After") or 0) or 15 * (attempt + 1))
        back = {}
        for k in ("normalized", "redirects"):
            for r in js.get(k, []):
                back[r["to"]] = back.get(r["from"], r["from"])
        got = {}
        for pg in js["pages"]:
            src = pg["title"]
            while src in back:
                src = back[src]
            if "revisions" in pg:
                got[src] = pg["revisions"][0]["slots"]["main"]["content"]
            elif pg.get("missing") or pg.get("invalid"):
                got[src] = None
        retry = []
        for t in batch:
            if t in got:
                save(t, got[t])
            else:
                retry.append(t)
        if len(retry) == len(batch):  # nothing came back: fetch one at a time
            for t in retry[:1]:
                save(t, fetch_one(t))
            retry = retry[1:]
        todo = retry + todo
        time.sleep(3)
    return out


def fetch_one(t: str) -> str | None:
    q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                "format": "json", "formatversion": "2", "redirects": "1", "titles": t})
    js = json.loads(urllib.request.urlopen(urllib.request.Request(API, data=q.encode(), headers={"User-Agent": UA}),
                                           timeout=60).read())["query"]
    pg = js["pages"][0]
    return pg["revisions"][0]["slots"]["main"]["content"] if "revisions" in pg else None


LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")


def parse_staff(text: str) -> list[dict]:
    i = text.find("{{NFL final staff") if text else -1
    if i >= 0:
        block = text[i:text.find("\n}}", i)]
        head = r"\|\s*([^=]+?)\s*="
    else:  # older pages: a plain table under ==Staff== with '''Section''' headings
        i = text.find("==Staff==") if text else -1
        if i < 0:
            return []
        j = text.find("\n==", i + 10)
        block = text[i:j if j > 0 else None]
        head = r"'''([^']+)'''"
    rows, section = [], None
    for line in block.split("\n"):
        m = re.match(head, line.strip())
        if m:
            section = m.group(1)
            continue
        if not line.startswith("*") or section is None:
            continue
        if not re.search(r"coach", section, re.I) or re.search(r"strength", section, re.I):
            continue
        body = line.lstrip("* ").strip()
        role, _, who = body.rpartition("–") if "–" in body else body.rpartition("-")
        lk = LINK.search(who)
        rows.append({"section": section, "role": role.strip(), "name": (lk.group(2) or lk.group(1)).strip() if lk else
                     re.sub(r"[\[\]]", "", who).strip(), "page": lk.group(1).strip() if lk else None})
    return rows


def _years(s: str) -> list[int]:
    s = re.sub(r"\{\{(?:nfly|NFL Year|nfl year|cfb link[^}]*)\|?", " ", s)
    now = 2026
    ys = []
    for a, b in re.findall(r"((?:19|20)\d\d)\s*(?:[–\-|]|&ndash;|to)\s*((?:19|20)\d\d|present)", s):
        ys += list(range(int(a), (now if b == "present" else int(b)) + 1))
    s2 = re.sub(r"((?:19|20)\d\d)\s*(?:[–\-|]|&ndash;|to)\s*((?:19|20)\d\d|present)", " ", s)
    ys += [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d\d)(?!\d)", s2)]
    return sorted(set(ys))


def parse_career(text: str) -> list[dict]:
    """One row per (team string, year) from the infobox past_coaching field."""
    if not text:
        return []
    m = re.search(r"\|\s*past_coaching\s*=(.*?)(?=\n\s*\|\s*\w+\s*=|\n\}\})", text, re.S)
    if not m:  # {{Infobox college coach}}: coach_years{n} / coach_team{n}
        rows = []
        yrs = dict(re.findall(r"\|\s*coach_years(\d+)\s*=\s*([^\n|]*(?:\{\{[^}]*\}\}[^\n|]*)?)", text))
        for n, team in re.findall(r"\|\s*coach_team(\d+)\s*=\s*(.+)", text):
            lk = LINK.search(team)
            target = lk.group(1).strip() if lk else re.split(r"\s*\(", team)[0].strip()
            disp = (lk.group(2) or lk.group(1)).strip() if lk else target
            for y in _years(yrs.get(n, "")):
                rows.append({"target": target, "display": disp, "year": y})
        return rows
    rows, team = [], None
    for line in m.group(1).split("\n"):
        line = line.strip()
        if line.startswith("**"):
            if team:
                for y in _years(line):
                    rows.append({**team, "year": y})
            continue
        if not line.startswith("*"):
            continue
        body = re.sub(r"\{\{ubl\|", "", line.lstrip("* "))
        head = re.split(r"<br\s*/?>", body)[0]
        lk = LINK.search(head)
        if lk:
            target, disp = lk.group(1).strip(), (lk.group(2) or lk.group(1)).strip()
        else:
            disp = re.split(r"\s*\(", head)[0].strip()
            target = disp
        team = {"target": target, "display": disp}
        for y in _years(head) or _years(body):  # some pages put the years after the role
            rows.append({**team, "year": y})
    return rows


# ---- colleges: one key per program, shared by coach careers and draft records
ALIAS = {  # draft-record college name (normalized) -> Wikipedia program article
    "boston col": "Boston College Eagles football", "central florida": "UCF Knights football",
    "ala birmingham": "UAB Blazers football", "texas san antonio": "UTSA Roadrunners football",
    "texas el paso": "UTEP Miners football", "middle tenn st": "Middle Tennessee Blue Raiders football",
    "la monroe": "Louisiana–Monroe Warhawks football", "mississippi": "Ole Miss Rebels football",
    "s f austin": "Stephen F. Austin Lumberjacks football", "se missouri st": "Southeast Missouri State Redhawks football",
    "miami fl": "Miami Hurricanes football", "miami oh": "Miami RedHawks football",
    "louisiana lafayette": "Louisiana Ragin' Cajuns football", "southern california": "USC Trojans football",
    "louisiana st": "LSU Tigers football", "north carolina st": "NC State Wolfpack football",
    "brigham young": "BYU Cougars football", "texas christian": "TCU Horned Frogs football",
    "southern methodist": "SMU Mustangs football", "nevada las vegas": "UNLV Rebels football",
    "pittsburgh": "Pittsburgh Panthers football",
}
AMBIGUOUS = {"miami"}  # unlinked "Miami" in a coach's career: FL or OH unknown, dropped


def norm(x: str) -> str:
    x = str(x).lower().replace("&", "and")
    x = re.sub(r"\bfootball\b|\bthe\b", "", x)
    x = re.sub(r"\bstate\b", "st", x)
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    return " ".join(x.split())


def _clean_target(t: str) -> str:
    t = re.sub(r"^(?:19|20)\d\d\s+", "", t)
    return re.sub(r"\s+(?:team|season)$", "", re.sub(r"[Ff]ootball", "football", t))


def college_keys(careers: pd.DataFrame):
    """Returns (careers with a 'key' column, function: draft-record college -> key)."""
    c = careers.copy()
    c["t"] = c.target.map(_clean_target)
    c["dn"] = c.display.map(norm)
    linked = c[c.t != c.display]
    # each article's key = its most common display name; an unlinked name maps through the article it usually links to
    canon = linked.groupby("t").dn.agg(lambda s: s.value_counts().index[0])
    dn_to_t = linked[linked.dn.str.match(r"^\D")].groupby("dn").t.agg(lambda s: s.value_counts().index[0])
    def key_dn(dn):
        if dn in AMBIGUOUS:
            return None
        return canon.get(dn_to_t[dn], dn) if dn in dn_to_t.index else dn
    c["key"] = [canon.get(t) if t in canon.index and t != disp else key_dn(dn)
                for t, disp, dn in zip(c.t, c.display, c.dn)]
    def college(name):
        n = norm(name)
        if n in ALIAS:
            t = ALIAS[n]
            return canon.get(t, norm(t))
        return key_dn(n)
    return c, college


def connections(years=range(2012, 2025), window=5) -> pd.DataFrame:
    """Every (NFL franchise, draft year, coach, college) where the coach was at that college in the
    `window` seasons before the draft."""
    staffs = pd.read_parquet(OUT / "nfl_staffs.parquet")
    car, _ = college_keys(pd.read_parquet(OUT / "coach_careers.parquet"))
    car = car.dropna(subset=["key"])
    st = staffs.dropna(subset=["page"]).drop_duplicates(["fr", "year", "page"])
    m = st.merge(car[["page", "key", "year"]].rename(columns={"year": "cyear"}), on="page")
    m = m[(m.cyear >= m.year - window) & (m.cyear <= m.year - 1)]
    g = m.groupby(["fr", "year", "key", "page", "name", "role"]).cyear.agg(["min", "max"]).reset_index()
    return g.rename(columns={"min": "first", "max": "last"})
