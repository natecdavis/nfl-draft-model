"""Draft-night trades for drafts nflverse's trade log doesn't cover yet, from the trade notes
on Wikipedia's "<year> NFL draft" page (CC BY-SA; saved locally, not redistributed).

Each traded pick carries a note; draft-day deals are marked "(D)" and described once in full:
  "Cleveland traded a first-round selection (6th overall) to Kansas City in exchange for
   first-, third- and fifth round selections (9th, 74th and 148th overall)."
Parsed into the nflverse trades format (one row per asset), then checked: every pick that
changed hands must end up with the team that actually used it.

Run:  python src/wiki_trades.py 2026   -> data/raw/trades_wiki_2026.parquet
"""
import re
import sys
import urllib.request

import pandas as pd

from outlook import RAW

TEAMS = {"Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF", "Carolina": "CAR",
         "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
         "Detroit": "DET", "Green Bay": "GB", "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
         "Kansas City": "KC", "Las Vegas": "LV", "LA Chargers": "LAC", "LA Rams": "LA", "Miami": "MIA",
         "Minnesota": "MIN", "New England": "NE", "New Orleans": "NO", "NY Giants": "NYG", "NY Jets": "NYJ",
         "Philadelphia": "PHI", "Pittsburgh": "PIT", "San Francisco": "SF", "Seattle": "SEA", "Tampa Bay": "TB",
         "Tennessee": "TEN", "Washington": "WAS"}
ROUND = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7}
TEAM_RE = "(?:the )?(" + "|".join(sorted(TEAMS, key=len, reverse=True)) + ")"
DEAL = re.compile(r"'''(?:No\. \d+: )?[^']*\(D\)\.'''\s*" + r"(?:The )?" + TEAM_RE + r" traded (.+?) to " + TEAM_RE
                  + r" in exchange for (.+?)\.(?=<ref)(<ref[^>]*>.*?date=([A-Z][a-z]+ \d+, \d{4}))?", re.S)


def fetch(year: int):
    f = RAW / "wiki" / f"{year}_NFL_draft.txt"
    if not f.exists():
        f.parent.mkdir(exist_ok=True)
        url = f"https://en.wikipedia.org/w/index.php?title={year}_NFL_draft&action=raw"
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-draft-model research script"})
        f.write_text(urllib.request.urlopen(req, timeout=60).read().decode())
    return f.read_text()


def assets(text: str, year: int, rnd_of: dict) -> list[dict]:
    """Picks (this year's by number, later years by round) and players in one side of a deal."""
    out = []
    for grp in re.findall(r"\(([^)]*overall)\)", text):
        for n in re.findall(r"(\d+)(?:st|nd|rd|th)", grp):
            out.append({"pick_season": year, "pick_number": int(n), "pick_round": rnd_of.get(int(n))})
    for fy, words in re.findall(r"(\d{4}) ((?:(?:first|second|third|fourth|fifth|sixth|seventh)[- ,and]*)+)round",
                                re.sub(r"\(.*?\)", "", text)):
        for w in re.findall("|".join(ROUND), words):
            out.append({"pick_season": int(fy), "pick_number": None, "pick_round": ROUND[w]})
    for name in re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", text):
        out.append({"pick_season": None, "pick_number": None, "pick_round": None,
                    "pfr_name": name[1] or re.sub(r" \(.*\)$", "", name[0])})
    return out


def parse(year: int) -> pd.DataFrame:
    raw = fetch(year)
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dp = dp[dp.season == year]
    rnd_of = dict(zip(dp.pick, dp["round"]))
    rows, seen = [], set()
    for m in DEAL.finditer(raw):
        a, give, b, get, _, date = m.groups()
        key = (a, give, b, get)
        if key in seen:
            continue
        seen.add(key)
        tid = 900000 + year % 100 * 1000 + len(seen)  # clear of nflverse ids
        when = pd.to_datetime(date).date() if date else None
        for frm, to, side in [(a, b, give), (b, a, get)]:
            for x in assets(side, year, rnd_of):
                rows.append({"trade_id": tid, "season": year, "trade_date": when, "gave": TEAMS[frm],
                             "received": TEAMS[to], "conditional": 0, "pfr_id": None, "pfr_name": None, **x})
    df = pd.DataFrame(rows)
    check(df, dp, raw)
    return df


def check(df, dp, raw):
    """Every traded pick's last receiver = the team that used it; every '(D)' note parsed."""
    n_notes = len(set(re.findall(r"'''(?:No\. \d+: )?[^']*\(D\)\.'''\s*(?!\s*See )([^<]{20,}?)\.<ref", raw)))
    n_deals = df.trade_id.nunique()
    print(f"draft-day deals described: {n_notes}, parsed: {n_deals}")
    from review import team
    used = dict(zip(dp.pick, dp.team.map(team)))
    cur = df[df.pick_number.notna()]
    bad = []
    for k, g in cur.groupby("pick_number"):
        # a pick can move more than once that night; it must end where it was used
        holders = set(g.received) - set(g.gave)
        final = holders.pop() if len(holders) == 1 else (g.received.iloc[-1])
        if final != used.get(int(k)) and used.get(int(k)) not in set(g.received):
            bad.append((int(k), list(zip(g.gave, g.received)), used.get(int(k))))
    print(f"picks that changed hands: {cur.pick_number.nunique()}, not ending with the team that used them: {len(bad)}")
    for b in bad:
        print("   ", b)
    assert n_deals == n_notes and not bad, "parse check failed"


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    d = parse(y)
    d.to_parquet(RAW / f"trades_wiki_{y}.parquet")
    print(f"saved {len(d)} rows, {d.trade_id.nunique()} trades")
