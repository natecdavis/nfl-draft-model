"""College football press-conference transcripts from ASAP Sports (asapsports.com FastScripts archive).

Football category (id 1), calendar years -> events (bowls, conference title games, kickoff games; media days
transcribe coaches only) -> event days -> interviews ('NAME: answer' turns after 'Q.' questions).
Only interviews whose interviewee name matches a board or drafted player (normalized name, draft class 0-5 years
after the event year) are fetched. Polite: one request every 2 s, every page cached under data/raw/asap/
(no robots.txt on the site; generic research User-Agent).
"""
import html
import re
import time
import urllib.request
from pathlib import Path

RAW = Path("data/raw/asap")
BASE = "http://www.asapsports.com/"
UA = "nfl-draft-research/0.1 (personal research script; python urllib)"
DELAY = 2.0
_last = [0.0]


def get(url: str, name: str) -> str:
    f = RAW / name
    if f.exists():
        return f.read_text(encoding="utf-8", errors="replace")
    wait = DELAY - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            raw = urllib.request.urlopen(req, timeout=60).read()
            break
        except Exception:
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(url)
    _last[0] = time.time()
    txt = raw.decode("utf-8", errors="replace")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(txt, encoding="utf-8")
    return txt


def links(page: str, kind: str):
    """(url, label) pairs for show_<kind>.php links."""
    out = []
    for u, lab in re.findall(r'href="(http://www\.asapsports\.com/show_' + kind + r'\.php\?[^"]+)"[^>]*>([^<]*)', page):
        out.append((html.unescape(u), html.unescape(lab).strip()))
    return out


def index(years) -> list[dict]:
    """Every interview link in the football category for the given calendar years."""
    rows = []
    for y in years:
        yp = get(f"{BASE}show_year.php?category=1&year={y}", f"year/{y}.html")
        for eu, title in links(yp, "events"):
            eid = re.search(r"event_id=(\d+)", eu).group(1)
            ep = get(eu, f"event/{eid}.html")
            for du, _ in links(ep, "event"):
                did = re.search(r"event_id=(\d+)", du).group(1)
                date = re.search(r"date=([\d-]+)", du)
                dp = get(du, f"day/{did}.html")
                for iu, who in links(dp, "interview"):
                    rows.append({"year": y, "event": title, "date": date.group(1) if date else None,
                                 "id": int(re.search(r"id=(\d+)", iu).group(1)), "who": who})
    return rows


def interview(iid: int) -> str:
    return get(f"{BASE}show_interview.php?id={iid}", f"interview/{iid}.html")


def turns(page: str) -> list[tuple[str, str]]:
    """[(speaker, text)] with speaker 'Q' for questions; continuation lines join the previous turn."""
    body = re.sub(r"(?i)<br\s*/?>|</p>", "\n", page)
    body = html.unescape(re.sub(r"<[^>]+>", "", body))
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    start = next((i for i, l in enumerate(lines) if re.match(r"^(Q\.|[A-Z][A-Z .'\-]+:)", l)), None)
    if start is None:
        return []
    out = []
    for l in lines[start:]:
        if re.match(r"^(FastScripts|ASAP|©|Copyright)", l, re.I):
            break
        m = re.match(r"^Q\.\s*(.*)", l)
        s = re.match(r"^([A-Z][A-Z .'\-]{2,40}):\s*(.*)", l)
        if m:
            out.append(("Q", m.group(1)))
        elif s:
            out.append((s.group(1).strip(), s.group(2)))
        elif out:
            out[-1] = (out[-1][0], out[-1][1] + " " + l)
    return out
