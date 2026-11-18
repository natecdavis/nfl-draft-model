"""Which college position coaches would have been worth pairing with which NFL teams before the 2026 draft?

Ex ante (information available before the draft):
  board     Hasan consensus 2026 (data/raw/hasan_2025_26.csv), top N prospects
  staffs    each prospect's school's 2025 coaching staff: Wikipedia "2025 {program} football team" (Coaching staff
            section), else the program page as it stood on 2025-10-15 (mid-season; a January 2026 snapshot already showed new staffs at
            schools that changed head coaches, e.g. LSU)
  coach     the staff member whose title names the prospect's position (coordinators count when the title names it)
  picks     each team's picks at the start of draft night (review.draft_night: before draft-night trades)
  chance    P(team takes the prospect) = sum over its picks of P(he goes at that pick), from the project's availability
            model (draftday.drift / spread: log(pick + 10) ~ normal around log(rank + 10) + drift)
  value     chance x (edge - 1) x expected value at that pick; edge from scripts/contact_value.py, measured on connected
            picks 2006-22: x1.15 rounds 1-2, x1.05 rounds 3-7 (role-matched / recent links ran x1.2-1.27, so this is
            conservative); expected value at a pick from the leave-one-class-out slot model (2006-22 drafted),
            smoothed by pick; $ at the 2025 cap
Summed per (coach, team). Ex post: 2026 NFL staffs (Wikipedia "2026 {team} season") - which of these coaches were
actually hired, and by whom.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import draftday as dd
import review as rv
from old_class_replication3 import base

OUT = Path("data/derived")
RAW = Path("data/raw/wiki")
TOP = 120
SCRATCH = None  # directory holding coaches_*.jsonl (web-looked-up staffs); set from argv
CAP = 279.2
POS_KW = {
    "QB": [r"quarterback"], "RB": [r"running back"], "WR": [r"wide receiver|receivers"], "TE": [r"tight end"],
    "OT": [r"offensive line|o-line|tackles"], "OC": [r"offensive line|o-line|interior|cent"], "OG": [r"offensive line|o-line|interior"], "C": [r"offensive line|o-line|interior|cent"],
    "IOL": [r"offensive line|o-line|interior"],
    "EDGE": [r"defensive end|edge|outside linebacker|pass rush|defensive line"], "ED": [r"defensive end|edge|outside linebacker|pass rush|defensive line"],
    "DL1T": [r"defensive line|defensive tackle|interior"], "DL3T": [r"defensive line|defensive tackle|interior"],
    "DL5T": [r"defensive line|defensive end|defensive tackle"], "DL": [r"defensive line|defensive tackle"],
    "LB": [r"(?<!outside )linebacker|inside linebacker"], "ILB": [r"linebacker"],
    "CB": [r"cornerback|defensive back|secondary|\bdbs?\b|nickel"], "S": [r"safet|defensive back|secondary|\bdbs?\b"],
}


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(co.API, data=q.encode(), headers={"User-Agent": co.UA})
    for attempt in range(5):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(15 * (attempt + 1))


def cached(name, fn):
    f = RAW / "pairing" / (re.sub(r"[^\w\-.()]+", "_", name) + ".json")
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return json.loads(f.read_text())["v"]
    v = fn()
    f.write_text(json.dumps({"v": v}))
    time.sleep(1)
    return v


ALIAS = {"Miami (FL)": "Miami Hurricanes", "NDSU": "North Dakota State Bison", "Pitt": "Pittsburgh Panthers",
         "USC": "USC Trojans", "UCF": "UCF Knights", "UConn": "UConn Huskies", "LSU": "LSU Tigers", "TCU": "TCU Horned Frogs",
         "Ole Miss": "Ole Miss Rebels", "Georgia State": "Georgia State Panthers", "Georgia Tech": "Georgia Tech Yellow Jackets",
         "Florida": "Florida Gators", "Georgia": "Georgia Bulldogs", "Washington": "Washington Huskies",
         "Michigan": "Michigan Wolverines", "Arizona": "Arizona Wildcats", "Iowa": "Iowa Hawkeyes", "Texas": "Texas Longhorns"}


def season_title(school):
    q = ALIAS.get(school, school)
    def go():
        js = api({"action": "query", "list": "search", "srsearch": f"2025 {q} football team", "srlimit": 10})
        first = q.split()[0].lower()
        for r in js["query"]["search"]:
            t = r["title"]
            if re.match(r"^2025 .+ football team$", t) and t.lower().startswith(f"2025 {q.lower()}") or \
                    (re.match(r"^2025 .+ football team$", t) and t.split()[1].lower() == first and len(q.split()) == 1
                     and len(t.split()) == 5):
                return t
        return None
    return cached(f"search_{school}", go)


def old_revision(title, ts="2025-10-15T00:00:00Z"):  # mid-season: before LSU / Florida firings, after preseason changes
    def go():
        js = api({"action": "query", "prop": "revisions", "titles": title, "rvprop": "content", "rvslots": "main",
                  "rvstart": ts, "rvdir": "older", "rvlimit": 1, "redirects": 1})
        pg = js["query"]["pages"][0]
        return pg["revisions"][0]["slots"]["main"]["content"] if "revisions" in pg else None
    return cached(f"rev_{title}_{ts[:10]}", go)


def parse_staff_table(text):
    """Rows of (name, title) from the first coaching-staff table (not 'staff changes')."""
    if not text:
        return []
    for m in re.finditer(r"(?im)^=+\s*(?:current\s+)?coaching staff\s*=+\s*$|\|\+[^\n]*coaching staff", text):
        seg = text[m.end(): m.end() + 15000]
        j = seg.find("{|")
        if j < 0 or j > 3000:
            continue
        k = seg.find("\n|}", j)
        tbl = seg[j: k if k > 0 else None]
        rows = []
        for row in re.split(r"\n\|-[^\n]*", tbl)[1:]:  # separators can carry attributes: |- align="center";
            cells = [c.strip() for c in re.split(r"\|\||\n\|", row) if c.strip()]
            cells = [re.sub(r'^(?:\s*[\w-]+\s*=\s*"[^"]*"\s*;?)+\s*\|', "", c).strip() for c in cells
                     if not c.startswith("!") and not c.startswith("{{")]
            cells = [c for c in cells if c]
            if len(cells) < 2:
                continue
            nm = co.LINK.search(cells[0])
            name = (nm.group(2) or nm.group(1)) if nm else re.sub(r"<[^>]+>|\[|\]|'''", "", cells[0]).strip()
            title = re.sub(r"<ref.*?(</ref>|/>)|\[\[(?:[^|\]]*\|)?([^\]]*)\]\]|<[^>]+>", lambda x: x.group(2) or "", cells[1]).strip()
            if name and title and len(name) < 50:
                rows.append({"coach": re.sub(r"\s*\(.*\)$", "", name).strip(), "title": title})
        if len(rows) >= 5:
            return rows
    return []


def staff_for(school):
    t = season_title(school)
    rows, src = [], None
    if t:
        rows = parse_staff_table(co.fetch([t], "college_seasons").get(t))
        src = t
    if len(rows) < 5 and t:
        prog = re.sub(r"^2025 (.+) football team$", r"\1 football", t)
        rows = parse_staff_table(old_revision(prog))
        src = prog + " (2025-10-15)"
    return rows, src


def expected_by_pick():
    v = pd.read_parquet(OUT / "contact_value.parquet")
    g = v.groupby("pick").exp_slot.mean()
    picks = np.arange(1, 263)
    lp = np.log(g.index.values + 10)
    coef = np.polyfit(lp, np.log(g.values.clip(1e-4)), 3)
    return dict(zip(picks, np.exp(np.polyval(coef, np.log(picks + 10)))))


def pick_probs(rank, picks_all):
    mu = np.log(rank + 10) + dd.drift(rank)
    sd = dd.spread(rank)
    edges = np.log(np.append(picks_all - 0.5, picks_all[-1] + 0.5) + 10)
    p = np.diff(norm.cdf(edges, mu, sd))
    return p  # residual mass = undrafted


def main():
    picks, _ = rv.draft_night(2026)
    picks["owner"] = picks.original_team.map(co.franchise)
    pk = picks.sort_values("pick")
    allp = pk.pick.values
    ev = expected_by_pick()
    h = pd.read_csv("data/raw/hasan_2025_26.csv")
    h = h[(h.year == 2026) & (h["rank"] <= TOP)].copy()

    staffs, srcs = {}, {}
    for s in sorted(h.school.unique()):
        staffs[s], srcs[s] = staff_for(s)
    # schools without a parsable Wikipedia staff: position coaches looked up by hand (web sources, one per row)
    web = {sc for sc in staffs if not staffs[sc]}
    for f in (sorted(Path(SCRATCH).glob("coaches_*.jsonl")) if SCRATCH else []):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("coach") and r["school"] in web:
                staffs[r["school"]].append({"coach": r["coach"], "title": r["title"], "pos": r["pos"],
                                            "source": r.get("source"), "confidence": r.get("confidence")})
                srcs[r["school"]] = "web lookup"
    miss = [s for s in staffs if not staffs[s]]
    print(f"top {TOP} prospects from {h.school.nunique()} schools; staff found for {h.school.nunique() - len(miss)}; "
          f"missing: {', '.join(miss)}")

    rows = []
    for _, r in h.iterrows():
        kws = POS_KW.get(r.pos, [])
        coaches = [c for c in staffs[r.school] if (c["pos"] == r.pos if c.get("pos") else
                                                   any(re.search(k, c["title"], re.I) for k in kws))]
        # prefer pure position coaches; keep coordinators when their title names the position
        probs = pick_probs(r["rank"], allp)
        for c in coaches:
            for tm, g in pk.groupby("owner"):
                idx = np.searchsorted(allp, g.pick.values)
                p = probs[idx]
                edge = np.where(g["round"].values <= 2, 0.15, 0.05)
                val = (p * edge * np.array([ev[x] for x in g.pick.values])).sum() * CAP
                rows.append({"team": tm, "coach": c["coach"], "title": c["title"], "school": r.school,
                             "prospect": r["name"], "pos": r.pos, "rank": r["rank"], "p_team": p.sum(), "value": val})
    d = pd.DataFrame(rows)
    pair = d.groupby(["team", "coach", "title", "school"]).agg(
        value=("value", "sum"), prospects=("prospect", lambda s: ", ".join(dict.fromkeys(s))),
        p_any=("p_team", "sum")).reset_index().sort_values("value", ascending=False)
    pair.to_parquet(OUT / "coach_pairing_2026.parquet")
    d.to_parquet(OUT / "coach_pairing_2026_detail.parquet")

    print("\nTOP COACH-TEAM PAIRS (expected draft value of the connection, $M, 2025 cap)")
    print(pair.head(25).assign(value=lambda x: x.value.round(2), p_any=lambda x: x.p_any.round(2)).to_string(index=False))
    print("\nBEST COACH PER TEAM, greedy (each coach to one team, highest values first)")
    used, taken, out = set(), set(), []
    for _, r in pair.iterrows():
        if r.coach in used or r.team in taken:
            continue
        used.add(r.coach); taken.add(r.team); out.append(r)
    g = pd.DataFrame(out)
    print(g.assign(value=lambda x: x.value.round(2)).to_string(index=False))
    tot = pair.groupby("coach").value.max().sort_values(ascending=False)
    print("\nMOST VALUABLE COACHES (best single team)\n" + tot.head(15).round(2).to_string())

    # ex post: 2026 NFL staffs
    abbrs = sorted(set(co.franchise(a) for a in list(co.TEAMS) + ["WAS"]) - {"STL", "SDG", "OAK"})
    nm = {"LAR": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LVR": "Las Vegas Raiders"}
    # current-season pages transclude "Template:{Team} staff" (staff as of now, Nov 2026)
    titles = {a: f"Template:{nm.get(a) or co.team_name(a, 2026)} staff" for a in abbrs}
    tx = co.fetch(list(titles.values()), "templates")
    rows = []
    for a, t in titles.items():
        sec = None
        for line in (tx.get(t) or "").split("\n"):
            if line.startswith(";"):
                sec = line[1:].strip()
            elif line.startswith("*") and sec and "coach" in sec.lower():
                role, _, who = line.lstrip("* ").rpartition("–")
                lk = co.LINK.search(who)
                name = (lk.group(2) or lk.group(1)) if lk else re.sub(r"[\[\]']", "", who).strip()
                rows.append({"team": a, "section": sec, "role": role.strip(), "name": name.strip()})
    st = pd.DataFrame(rows)
    import nflreadpy as nr
    nr_picks = nr.load_draft_picks().to_pandas().query("season == 2026")
    nr_picks["cschool"] = nr_picks.college
    print(f"\n2026 NFL staffs parsed: {st.team.nunique()} teams")
    st["k"] = st.name.map(lambda n: base(n)[0])
    cand = d.drop_duplicates(["coach", "school", "title"])[["coach", "school", "title"]].assign(k=lambda x: x.coach.map(lambda n: base(n)[0]))
    hired = cand.merge(st[["k", "team", "role"]], on="k")
    print("college position coaches of top prospects who were on a 2026 NFL staff:")
    to_pfr = {v: k for k, v in rv.PFR_TEAM.items() if k in co.TEAMS}
    for _, r in hired.iterrows():
        dp = nr_picks[nr_picks.team.map(co.franchise) == r.team]  # nflverse codes match the staff side (TAM, NWE)
        dr = dp
        # d.team uses review codes (GB, KC, TB...), the staff side PFR codes (GNB, KAN, TAM...)
        own = d[(d.coach == r.coach) & (d.team.map(to_pfr).fillna(d.team) == r.team)].drop_duplicates("prospect")
        took = [p for p in own.prospect if base(p)[0] in set(dr.pfr_player_name.map(lambda n: base(n)[0]))]
        sch = r.school.replace("Miami (FL)", "Miami").split(" (")[0]
        same = dr[dr.college.str.startswith(sch.replace("State", "St."))][["pick", "pfr_player_name", "position"]].values.tolist()
        print(f"   {r.coach} ({r.school} {r.title}) -> {r.team} {r.role}; drafted his top-{TOP} prospects: {took or 'none'}; "
              f"any picks from {r.school}: {same or 'none'}")


if __name__ == "__main__":
    SCRATCH = sys.argv[1] if len(sys.argv) > 1 else None
    main()
