"""Pilot: how often do GMs / head coaches name draftable prospects in pre-draft press conferences?

Feasibility check for a voice/text "tell" test (does how a team talks about a prospect predict whether it
takes him?). Sources: YouTube videos on each team's official channel (plus NFL / NFL Network for combine
pressers), uploaded Feb 1 - draft eve, for drafts 2023-2025. Auto-captions only (no audio yet).

Steps (each cached under data/raw/presser_yt/):
  search   ytsearch per team x year x event, keep official-channel videos with presser-like titles
  meta     full metadata (upload date) for candidates, keep the pre-draft window
  subs     English auto-captions (.vtt) for kept videos
  scan     prospect-name mentions per video (board top-320 + drafted players of that class)
Output: reports/presser_pilot.csv (one row per video) and reports/presser_pilot_mentions.csv.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/presser_yt")
YEARS = [2023, 2024, 2025]
DRAFT_DAY = {2023: "20230427", 2024: "20240425", 2025: "20250424"}
QUERIES = ["{team} pre-draft press conference {year}", "{team} general manager combine press conference {year}",
           "{team} head coach combine press conference {year}"]
TITLE_OK = re.compile(r"pre-?\s?draft|combine|luncheon|press conference|presser|media availability|meets the media"
                      r"|speaks|talks|news conference", re.I)
TITLE_BAD = re.compile(r"gets the call|highlights|mic'?d|rookie minicamp|post-?draft|day [23]|round|pick|trade|"
                       r"welcome|first look|live:? draft|reacts", re.I)
YTDLP = ["uvx", "yt-dlp", "--quiet", "--no-warnings"]


def run(args, timeout=300):
    return subprocess.run(YTDLP + args, capture_output=True, text=True, timeout=timeout).stdout


def search(teams):
    f = RAW / "search.parquet"
    if f.exists():
        return pd.read_parquet(f)
    part = RAW / "search_parts"
    part.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        for _, t in teams.iterrows():
            pf = part / f"{year}_{t.team_abbr}.parquet"
            if pf.exists():
                continue
            rows = []
            for q in QUERIES:
                q = q.format(team=t.team_name, year=year)
                out = run(["--flat-playlist", "--print", "%(id)s\t%(title)s\t%(channel)s\t%(duration)s",
                           f"ytsearch15:{q}"])
                for line in out.splitlines():
                    p = line.split("\t")
                    if len(p) == 4:
                        rows.append({"year": year, "team": t.team_abbr, "team_name": t.team_name, "query": q,
                                     "id": p[0], "title": p[1], "channel": p[2], "duration": p[3]})
                time.sleep(1)
            pd.DataFrame(rows).to_parquet(pf)
            print(year, t.team_abbr, len(rows), flush=True)
    df = pd.concat([pd.read_parquet(x) for x in sorted(part.glob("*.parquet"))], ignore_index=True)
    df.to_parquet(f)
    return df


def candidates(s):
    s = s.copy()
    official = s.channel == s.team_name
    league = s.channel.isin(["NFL", "NFL Network", "NFL on ESPN"]) & s.title.str.contains(r"combine", case=False)
    league &= s.apply(lambda r: r.team_name.split()[-1].lower() in r.title.lower(), axis=1)
    dur = pd.to_numeric(s.duration, errors="coerce").fillna(0)
    keep = (official | league) & s.title.str.contains(TITLE_OK) & ~s.title.str.contains(TITLE_BAD) & (dur >= 240)
    return s[keep].drop_duplicates(["year", "team", "id"])


def meta(ids):
    f = RAW / "meta.json"
    m = json.loads(f.read_text()) if f.exists() else {}
    for i, vid in enumerate(ids):
        if vid in m:
            continue
        out = run(["--skip-download", "--print", "%(upload_date)s", f"https://www.youtube.com/watch?v={vid}"])
        m[vid] = out.strip()
        if i % 25 == 0:
            f.write_text(json.dumps(m))
            print("meta", i, len(ids), flush=True)
        time.sleep(1)
    f.write_text(json.dumps(m))
    return m


def subs(ids):
    d = RAW / "subs"
    d.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(ids):
        if list(d.glob(f"{vid}*")):
            continue
        run(["--skip-download", "--write-auto-subs", "--sub-langs", "en", "--sub-format", "vtt",
             "-o", str(d / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"])
        if not list(d.glob(f"{vid}*")):
            (d / f"{vid}.none").write_text("")
        if i % 25 == 0:
            print("subs", i, len(ids), flush=True)
        time.sleep(1)


def vtt_text(path: Path) -> str:
    """Plain caption text: drop cue timing / inline word tags, de-duplicate rolling lines."""
    seen, out = set(), []
    for line in path.read_text(errors="replace").splitlines():
        if "-->" in line or not line.strip() or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return " ".join(out)


def norm(s: str) -> str:
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def prospects(year):
    picks = pd.read_parquet("data/raw/draft_picks.parquet")
    pk = picks[picks.season == year][["pfr_player_name", "team", "pick", "position"]]
    pk.columns = ["name", "drafted_by", "pick", "pos"]
    if year <= 2024:
        h = pd.read_csv("data/raw/hasan.csv")
        h = h[h.year == year]
    else:
        h = pd.read_csv("data/raw/hasan_2025_26.csv")
        h = h[h.year == year] if "year" in h.columns else h
    b = h[["name", "rank"]].assign(name=lambda x: x.name.astype(str))
    pk["key"], b["key"] = pk.name.map(norm), b.name.map(norm)
    p = pk.merge(b[["key", "rank"]], on="key", how="outer")
    p["name"] = p.name.fillna(p.key)
    p = p[p.key.str.len() > 0].drop_duplicates("key")
    p["last"] = p.key.str.split().str[-1]
    p["first"] = p.key.str.split().str[0]
    return p


def common_surnames():
    """Surnames that are English words, or shared with many NFL players: unsafe to match alone."""
    words = set()
    wf = Path("/usr/share/dict/words")
    if wf.exists():
        words = {w.strip().lower() for w in wf.read_text().splitlines() if w.strip().islower()}
    pl = pd.read_parquet("data/raw/draft_picks.parquet")
    last = pl.pfr_player_name.dropna().map(norm).str.split().str[-1].value_counts()
    return words | set(last[last >= 6].index)


def scan(kept):
    unsafe = common_surnames()
    rows = []
    for year, g in kept.groupby("year"):
        p = prospects(year)
        dup_last = set(p["last"].value_counts()[lambda v: v > 1].index)
        for _, v in g.iterrows():
            f = RAW / "subs" / f"{v.id}.en.vtt"
            if not f.exists():
                continue
            txt = " " + norm(vtt_text(f)) + " "
            for _, r in p.iterrows():
                full = txt.count(f" {r.key} ")
                lastn = txt.count(f" {r['last']} ")
                safe_last = len(r["last"]) >= 4 and r["last"] not in unsafe and r["last"] not in dup_last
                if full or (safe_last and lastn):
                    rows.append({"year": year, "team": v.team, "id": v.id, "title": v.title, "prospect": r["name"],
                                 "board_rank": r["rank"], "pos": r.pos, "drafted_by": r.drafted_by, "pick": r.pick,
                                 "full_mentions": full, "last_mentions": lastn if safe_last else 0,
                                 "match": "full" if full else "last"})
    return pd.DataFrame(rows)


def main():
    teams = pd.read_parquet("data/raw/teams.parquet")
    teams = teams[~teams.team_abbr.isin(["LA", "OAK", "SD", "STL"])].drop_duplicates("team_abbr")
    pfr = {"GB": "GNB", "KC": "KAN", "LV": "LVR", "NO": "NOR", "NE": "NWE", "SF": "SFO", "TB": "TAM"}
    teams["team_abbr"] = teams.team_abbr.replace(pfr)  # draft_picks uses PFR codes
    assert len(teams) == 32, len(teams)
    s = search(teams)
    c = candidates(s)
    print("search rows", len(s), "candidates", len(c))
    m = meta(c.id.unique().tolist())
    c["upload"] = c.id.map(m)
    c["in_window"] = c.apply(lambda r: f"{r.year}0201" <= str(r.upload) < DRAFT_DAY[r.year], axis=1)
    kept = c[c.in_window].copy()
    print("in window", len(kept))
    subs(kept.id.unique().tolist())
    kept["has_subs"] = kept.id.map(lambda i: (RAW / "subs" / f"{i}.en.vtt").exists())
    kept["words"] = kept.id.map(lambda i: len(vtt_text(RAW / "subs" / f"{i}.en.vtt").split())
                                if (RAW / "subs" / f"{i}.en.vtt").exists() else 0)
    men = scan(kept)
    Path("reports").mkdir(exist_ok=True)
    kept.to_csv("reports/presser_pilot.csv", index=False)
    men.to_csv("reports/presser_pilot_mentions.csv", index=False)
    print("videos", len(kept), "with subs", kept.has_subs.sum(), "mention rows", len(men))


if __name__ == "__main__":
    sys.exit(main())
