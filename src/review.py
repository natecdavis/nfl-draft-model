"""Past drafts, pick by pick: what teams did vs what the model would have recommended.

For each draft 2014-2026 (Arif Hasan's consensus board: ~/cfdarko/hasan.csv for 2014-2024, his public
Wide Left sheets for 2025-2026, saved to data/raw/hasan_2025_26.csv; 2026 draft-night trades aren't
in nflverse yet, so 2026's come from Wikipedia's draft page via src/wiki_trades.py):

  Pick ownership    who held each pick when the draft began (draft-night trades undone,
                    trades before draft night kept) vs the team that made the pick
  Team state        as of draft day:
                      strength  = prior season's rating, shrunk for luck, carried forward
                      QB        = the team's week-1 starter that season unless he's a rookie
                                  from this draft (so March moves like Cousins->ATL count),
                                  else the prior season's main QB; status from his age,
                                  contract and draft round, same rules as teams.py
  Recommendation    at each pick, for the team on the clock: the available board player
                    (by consensus rank) with the most title value for that team, given what
                    it had already taken that night; same engine as the draft-day app,
                    through precomputed value tables per team situation
  Actual pick       the same value for the player the team chose (players off the board
                    are treated as ranked just past it)
  Draft-night trades  scored for both teams with each team's own situation
  Outcomes (2014-2021 drafts)  star / starter / bust for the actual pick and for the
                    recommended player (wherever he was drafted)

Run:  python src/review.py [first_year last_year]   -> data/derived/review_picks.parquet etc.
"""
import sys

import os

import numpy as np
import pandas as pd

from draftday import LEVELS, NEED_STEP, DraftEngine
from outlook import GROUPS, OUT, RAW
from public_board import _norm
from simulate import league_params, load_schedules
from trade import Context, Pick, evaluate

YEARS = list(range(2014, 2027))
# nflverse's trade log ends 2026-04-17, before the 2026 draft: those draft-night trades come
# from Wikipedia's draft page (src/wiki_trades.py), used only for drafts nflverse lacks
DRAFT_START = {2014: "2014-05-08", 2015: "2015-04-30", 2016: "2016-04-28", 2017: "2017-04-27",
               2018: "2018-04-26", 2019: "2019-04-25", 2020: "2020-04-23", 2021: "2021-04-29",
               2022: "2022-04-28", 2023: "2023-04-27", 2024: "2024-04-25", 2025: "2025-04-24",
               2026: "2026-04-23"}
PFR_TEAM = {"GNB": "GB", "KAN": "KC", "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB", "LAR": "LA",
            "LVR": "LV", "OAK": "LV", "SDG": "LAC", "STL": "LA", "SD": "LAC"}
DRAFT_POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "T", "OT": "T", "G": "G",
             "OL": "G", "C": "C", "DE": "ED", "OLB": "ED", "DT": "IDL", "NT": "IDL", "DL": "IDL", "LB": "LB",
             "ILB": "LB", "CB": "CB", "DB": "CB", "S": "S", "SAF": "S", "FS": "S", "K": "SPEC", "P": "SPEC",
             "LS": "SPEC"}
HASAN_POS = {"QB": "QB", "RB": "RB", "RBF": "RB", "RBC": "RB", "FB": "RB", "WR": "WR", "WRF": "WR", "WRS": "WR",
             "SE": "WR", "FL": "WR", "SL": "WR", "SB": "WR", "TE": "TE", "OT": "T", "OG": "G", "OL": "G",
             "IL": "G", "OC": "C", "EDGE": "ED", "ED": "ED", "ER": "ED", "OLB": "ED", "DL1T": "IDL",
             "DL3T": "IDL", "DL5T": "IDL", "ID": "IDL", "IDL": "IDL", "3T": "IDL", "5T": "IDL", "NT": "IDL",
             "LB": "LB", "ILB": "LB", "OB": "LB", "OFB": "LB", "MB": "LB", "CB": "CB", "NCB": "CB", "S": "S",
             "SS": "S", "FS": "S", "PK": "SPEC", "K": "SPEC", "P": "SPEC"}
OFF_BOARD_RANK = 330
team = lambda s: PFR_TEAM.get(s, s)


def load_trades() -> pd.DataFrame:
    """nflverse's trade log, plus Wikipedia-parsed draft-night trades for any draft it
    doesn't cover yet (no trades dated draft week)."""
    t = pd.read_parquet(RAW / "trades.parquet")
    dates = pd.to_datetime(t.trade_date)
    for f in sorted(RAW.glob("trades_wiki_*.parquet")):
        y = int(f.stem.split("_")[-1])
        start = pd.Timestamp(DRAFT_START[y])
        if not ((dates >= start) & (dates <= start + pd.Timedelta(days=3))).any():
            t = pd.concat([t, pd.read_parquet(f)], ignore_index=True)
    return t


# ---------- 1. picks and draft-night ownership ----------

def draft_night(year: int) -> pd.DataFrame:
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    d = dp[dp.season == year][["pick", "round", "team", "pfr_player_name", "pfr_player_id", "position"]].copy()
    d["team"] = d.team.map(team)
    out = pd.read_parquet(OUT / "pick_outcomes.parquet")
    grp_known = dict(zip(out[out.draft_year == year].pick, out[out.draft_year == year].grp))
    d["grp"] = [grp_known.get(p) or DRAFT_POS.get(pos, "SPEC") for p, pos in zip(d.pick, d.position)]
    d.loc[~d.grp.isin(GROUPS), "grp"] = "SPEC"

    t = load_trades()
    for c in ["gave", "received"]:
        t[c] = t[c].map(team)
    t["date"] = pd.to_datetime(t.trade_date)
    start = pd.Timestamp(DRAFT_START[year])
    night = t[(t.date >= start) & (t.date <= start + pd.Timedelta(days=3))]
    moved = night[(night.pick_season == year) & night.pick_number.notna()].sort_values("date")
    first_giver = moved.groupby("pick_number").gave.first()
    # owner when the draft began = the root of the pick's draft-night chain: a team that gave it and never received
    # it that night. The first giver by date can be a later link when a pick moves twice the same day (trade ids
    # aren't in time order within a day): 66 picks 2014-2026 were mislabeled until Nov 2026 (63 chains, 3 loops).
    # A pick that went round in a loop (A -> B -> A) has no such root: then the team that made the pick.
    root = {}
    for k, g in moved.groupby("pick_number"):
        r = set(g.gave) - set(g.received)
        if len(r) == 1:
            root[k] = r.pop()
    d["original_team"] = [root.get(p, tm) for p, tm in zip(d.pick, d.team)]
    d["traded_on_night"] = d.pick.isin(first_giver.index) & (d.original_team != d.team)
    return d.reset_index(drop=True), night


def board(year: int, picks: pd.DataFrame) -> pd.DataFrame:
    h = pd.concat([pd.read_csv(RAW / "hasan.csv", encoding="utf-8-sig"),
                   pd.read_csv(RAW / "hasan_2025_26.csv")], ignore_index=True)
    h = h[(h.year == year) & (h["rank"] <= 320)].copy()  # same depth as the 2014-24 boards
    h["grp"] = h.pos.map(HASAN_POS).fillna("SPEC")
    h["key"] = h.name.str.replace(r"\(.*?\)", "", regex=True).map(_norm)  # "Kevin White (WR)"
    keys = dict(zip(picks.pfr_player_name.map(_norm), picks.pick))
    h["drafted_at"] = h.key.map(keys)
    # Two board players with the same name (Kevin White WR / Kevin White CB): keep the one
    # whose position matches the draft record
    dgrp = dict(zip(picks.pick, picks.grp))
    dupe = h.drafted_at.notna() & h.drafted_at.duplicated(keep=False)
    for pk_ in h.loc[dupe, "drafted_at"].unique():
        rows = h.index[h.drafted_at == pk_]
        keep = [i for i in rows if HASAN_POS.get(h.at[i, "pos"]) == dgrp.get(pk_)][:1] or list(rows[:1])
        h.loc[[i for i in rows if i not in keep], "drafted_at"] = np.nan
    # Fallback for nicknames ("Alvin" vs "Bud" Dupree): same last name, similar school,
    # same position group, not already matched
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    college = dict(zip(dp[dp.season == year].pick, dp[dp.season == year].college))
    left = picks[~picks.pick.isin(h.drafted_at.dropna())].assign(
        names=lambda d: d.pfr_player_name.map(lambda n: set(_norm(n).split()[1:]) if _norm(n) else set()))
    from difflib import SequenceMatcher
    sim = lambda a, b: SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    for i in h.index[h.drafted_at.isna()]:
        last = h.at[i, "key"].split()[-1] if h.at[i, "key"] else ""
        # last name among the draft name's surname parts ("Tryon" in "Tryon Shoyinka")
        c = left[left.names.map(lambda ns: last in ns) & ~left.pick.isin(h.drafted_at.dropna())]
        same_school = c[[sim(college.get(p, ""), h.at[i, "school"]) >= 0.5 for p in c.pick]]
        same_pos = c[c.grp == h.at[i, "grp"]]
        pick_ = (same_school if len(same_school) == 1 else same_pos if len(same_pos) == 1 else None)
        if pick_ is not None:
            h.at[i, "drafted_at"] = pick_.pick.iloc[0]
    # a drafted player's group comes from the draft record (more reliable)
    # grp_label: the board's own position (known before the draft; what decisions use, via
    # positions.blended). grp: for a drafted player, the position his team listed (known only once
    # he's drafted; used for what he actually delivered)
    h["grp_label"] = h.grp
    draft_grp = dict(zip(picks.pick, picks.grp))
    rec = h.drafted_at.map(draft_grp)
    use = h.drafted_at.notna() & rec.isin(GROUPS)  # a record without a usable position keeps the board's
    h.loc[use, "grp"] = rec[use]
    return h.reset_index(drop=True)


# ---------- 2. team state at draft time ----------

class States:
    def __init__(self):
        self.ratings = league_params(load_schedules())["ratings"]
        s = pd.concat([pd.read_parquet(RAW / "snap_counts.parquet"),
                       pd.read_parquet(RAW / "snap_counts_2026.parquet")], ignore_index=True)
        self.snaps = s[(s.position == "QB") & (s.game_type == "REG")].assign(team=lambda x: x.team.map(team))
        pl_ = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
        self.players = pl_.set_index("pfr_id")
        self.contracts = pd.read_parquet(RAW / "contracts.parquet")
        # QB play: EPA per dropback by season (nflverse), and the league median each season
        ps = pd.read_parquet(RAW / "player_stats.parquet")
        ps = ps[(ps.position == "QB") & (ps.season_type == "REG")].copy()
        ps["dropbacks"] = ps.attempts + ps.sacks_suffered
        ps["epa_db"] = ps.passing_epa / ps.dropbacks.clip(lower=1)
        q = ps[ps.dropbacks >= 200]
        self.qb_med = q.groupby("season").epa_db.median()
        self.qb_play = {(p, y): (e, d) for p, y, e, d in zip(ps.player_id, ps.season, ps.epa_db, ps.dropbacks)}
        self.trades = load_trades().assign(
            date=lambda d: pd.to_datetime(d.trade_date), gave=lambda d: d.gave.map(team))
        tms = pd.read_parquet(RAW / "teams.parquet")
        nick = dict(zip(tms.team_nick, tms.team_abbr))
        self.contracts["team_code"] = self.contracts.team.map(lambda x: nick.get(str(x).split("/")[-1], x))

    def qb(self, tm: str, year: int) -> dict:
        """The team's QB as of draft day and its QB situation (same rules as teams.py)."""
        cand = None
        wk = self.snaps[(self.snaps.season == year) & (self.snaps.team == tm)]
        if len(wk):
            w1 = wk[wk.week == wk.week.min()].sort_values("offense_snaps").iloc[-1]
            dy = self.players.draft_year.get(w1.pfr_player_id)
            if not (pd.notna(dy) and int(dy) == year):  # a rookie from this draft doesn't count
                cand = w1.pfr_player_id, w1.player
        if cand is None:
            prev = self.snaps[(self.snaps.season == year - 1) & (self.snaps.team == tm)]
            if len(prev):
                g = prev.groupby(["pfr_player_id", "player"]).offense_snaps.sum().sort_values()
                cand = g.index[-1]
                # If last season's QB played for another team this season, he'd left by the
                # draft (e.g. Darnold, traded NYJ -> CAR in April 2021): no starter.
                if self._left(cand[0], tm, year):
                    return {"qb": f"none ({cand[1]} left)", "status": "no starter", "qb_from": 1, "shift": {}}
        if cand is None:
            return {"qb": "?", "status": "unknown", "qb_from": None, "shift": {}}
        pid, name = cand
        p = self.players.loc[pid] if pid in self.players.index else None
        otc = pd.to_numeric(p.otc_id, errors="coerce") if p is not None else np.nan
        # Contracts known on draft day: anything from earlier years, plus a deal signed this
        # year with this team (March free agency). A deal with another team later that year
        # (e.g. after a release) didn't exist yet.
        c = self.contracts[self.contracts.otc_id == otc]
        k = c[(c.year_signed < year) | ((c.year_signed == year) & (c.team_code.map(team) == tm))].sort_values("year_signed")
        latest = k.iloc[-1] if len(k) else None
        rnd = p.draft_round if p is not None else np.nan
        dyear = p.draft_year if p is not None else np.nan
        birth = pd.to_datetime(p.birth_date, errors="coerce") if p is not None else pd.NaT
        age = (pd.Timestamp(f"{year}-09-01") - birth).days / 365.25 if pd.notna(birth) else np.nan
        # a rookie deal runs at most 5 seasons; past that, his later deal is just missing from the data
        on_rookie = (latest is not None and latest.year_signed == latest.draft_year
                     and pd.notna(dyear) and year - dyear <= 4)
        # no contract in the data at all, but drafted within the rookie window (WAS 2020, Haskins: was "veteran")
        on_rookie = on_rookie or (latest is None and pd.notna(dyear) and pd.notna(rnd) and 1 <= year - dyear <= 4)
        pay = latest.apy_cap_pct if latest is not None and not (
            latest.year_signed == latest.draft_year and not on_rookie) else np.nan
        shift, qb_from = {}, None
        if on_rookie and rnd == 1:
            status = "rookie deal (1st round)"
            s_idx = int(dyear) + 4 - year + 1
            if 1 <= s_idx <= 5:
                shift = {s_idx: -3.0}
        elif on_rookie:
            status, qb_from = "rookie deal (later round)", 2
        elif pd.notna(age) and age >= 36:
            status, qb_from = f"veteran, {age:.0f}", int(np.clip(39 - age, 1, 4))
        elif pd.notna(pay) and pay < 0.07:
            status, qb_from = "stopgap veteran", 1
        else:
            status = "paid veteran" if pd.notna(pay) else "veteran (contract not in data)"
            play = self._play(p.gsis_id if p is not None else None, year)
            if play is not None and play < self.qb_med.get(year - 1, 0):
                status, qb_from = status + ", below-average play", 2
        # seasons left on his deal after this one (known on draft day; review.p_need)
        # Negative = his latest deal in the data ended before this season (current deal missing, usually an old
        # QB going year to year: Brady 2022, Rivers 2019). p_need counts these with "expiring" on purpose.
        left = (int(latest.year_signed + latest.years - 1 - year)
                if latest is not None and pd.notna(latest.years) and pd.notna(latest.year_signed) else None)
        return {"qb": name, "status": status, "qb_from": qb_from, "shift": shift, "qb_left": left}

    def _play(self, gsis, year):
        """EPA per dropback last season (or the one before if he barely played), else None."""
        for y in (year - 1, year - 2):
            e = self.qb_play.get((gsis, y))
            if e is not None and e[1] >= 200:
                return e[0] - self.qb_med.get(y, 0) + self.qb_med.get(year - 1, 0)
        return None

    def _left(self, pfr_id, tm, year) -> bool:
        """Had last season's QB left the team by draft day? Yes if traded away before the
        draft, or signed with another team that year without being traded away later."""
        start = pd.Timestamp(DRAFT_START[year])
        t = self.trades[(self.trades.pfr_id == pfr_id) & (self.trades.gave == tm)]
        if ((t.date < start) & (t.date > pd.Timestamp(f"{year - 1}-08-01"))).any():
            return True
        if (t.date >= start).any():
            return False  # traded away after the draft: still ours on draft day
        p = self.players.loc[pfr_id] if pfr_id in self.players.index else None
        otc = pd.to_numeric(p.otc_id, errors="coerce") if p is not None else np.nan
        c = self.contracts[(self.contracts.otc_id == otc) & (self.contracts.year_signed == year)]
        return bool(len(c) and (c.team_code.map(team) != tm).all())

    def state(self, tm: str, year: int) -> dict:
        r = self.ratings.get(year - 1, pd.Series(dtype=float)).get(tm, 0.0)
        q = self.qb(tm, year)
        start = round(0.6 * 0.73 * r * 2) / 2  # to the nearest half point (shared value tables)
        return {"strength_prior": r, "start": start, **q, "p_need": p_need(q["status"], q["qb_from"], q.get("qb_left"))}


# ---------- 3. value tables ----------

T_GRID = np.unique(np.round(np.exp(np.linspace(np.log(11), np.log(OFF_BOARD_RANK + 10), 22)) - 10).astype(int))
K_GRID = np.unique(np.round(np.exp(np.linspace(np.log(11), np.log(272), 22)) - 10).astype(int))


class Table:
    """Titles for (position, need level, effective talent, pick) on a grid, for one situation.
    Level 0-2 = players the team already took at the position this draft (draftday.NEED_STEP: measured
    x1 / x0.95 / x0.88 all seasons; "early" = starter/rotation/depth); for QBs 0 = fills the need, 1-2 = filled.
    The grid is over talent already shaded for a fall; lookup() applies the shading once (the grid
    used to be built with player_value(), which shades again: fixed Nov 2026)."""
    def __init__(self, sit: dict, ctx: Context, fall_weight=0.85):  # same default as FALL_W below
        self.eng = DraftEngine(sit, ctx=ctx, fall_weight=fall_weight)
        e = self.eng
        self.v = {}
        for g in GROUPS:
            arr = np.zeros((3, len(T_GRID), len(K_GRID)))
            for li in range(3):
                for a, t in enumerate(T_GRID):
                    for b, k in enumerate(K_GRID):
                        pts = (e.player_points(g, t, k, qb_filled=li > 0) if g == "QB"
                               else e.player_points(g, t, k, nth=li) if NEED_STEP == "measured"
                               else e.player_points(g, t, k, LEVELS[li]))
                        arr[li, a, b] = e.titles(pts)
            self.v[g] = arr

    def lookup(self, g, level, rank, pick, p_need=1.0) -> float:
        """p_need: chance the team's QB need is real (QBs at level 0 only): the value is
        p x (fills the need) + (1 - p) x (need already filled)."""
        if g not in self.v:
            return -1.0
        t = self.eng.effective_talent(min(rank, OFF_BOARD_RANK), pick)
        x, y = np.log(T_GRID + 10), np.log(K_GRID + 10)
        lt, lk = np.log(t + 10), np.log(pick + 10)
        i = int(np.clip(np.searchsorted(x, lt) - 1, 0, len(x) - 2)); j = int(np.clip(np.searchsorted(y, lk) - 1, 0, len(y) - 2))
        fx = np.clip((lt - x[i]) / (x[i + 1] - x[i]), 0, 1); fy = np.clip((lk - y[j]) / (y[j + 1] - y[j]), 0, 1)
        at = lambda A: float((1 - fx) * (1 - fy) * A[i, j] + fx * (1 - fy) * A[i + 1, j] + (1 - fx) * fy * A[i, j + 1] + fx * fy * A[i + 1, j + 1])
        if g == "QB" and level == 0 and p_need < 1.0:
            return p_need * at(self.v[g][0]) + (1 - p_need) * at(self.v[g][1])
        return at(self.v[g][level])


QB_NEED_CONTRACT = os.environ.get("QB_NEED_CONTRACT", "split") == "split"  # "off": one rate per veteran flag


def p_need(status: str, qb_from, left=None) -> float:
    """Chance a flagged QB need is real: share of such team-years whose main QB the next season was someone else
    (README, 'Why the model's round-1 picks start less'). The two veteran flags (age 36+, paid veteran after a
    below-average season) split by his contract (`left` = seasons left after this one, known on draft day):
    under contract next year 0.39 (18 of 46), expiring 0.68 (30 of 44; 2014-2024; README 'QB need while the
    veteran is under contract', scripts/qb_need_contract.py). "Expiring" includes the 12 team-years whose latest
    deal in the data had already ended (left < 0: current deal missing); 10 of those 12 changed QB, against 20
    of 32 with a deal known to expire, so they are kept with expiring rather than split off (checked Nov 17
    2026). No deal in the data at all (left is None): the old pooled rates."""
    if qb_from is None or (isinstance(qb_from, float) and np.isnan(qb_from)):
        return 1.0  # no need flagged: the tables already give a QB no playing time
    s = str(status)
    for prefix, p in [("no starter", 1.0), ("stopgap", 0.79), ("rookie deal (later", 0.62)]:
        if s.startswith(prefix):
            return p
    vet = "below-average" in s or s.startswith("veteran, ")
    if vet and QB_NEED_CONTRACT and left is not None and not (isinstance(left, float) and np.isnan(left)):
        return 0.39 if left >= 1 else 0.68
    if "below-average" in s:
        return 0.54
    if s.startswith("veteran, "):
        return 0.52
    return 1.0


# ---------- 4. the review ----------

def engine_for(year, tables=None):
    """Point the outcome engine at draft `year` with that draft class left out (classes the engine was
    fit on: through prospect_model.LAST_SEASON_PLAYED), so past drafts aren't valued with their own
    careers (scripts/holdout_review.py, README 'Holdout check'). Later drafts use the full engine.
    Clears `tables` (value tables built with another engine) when the engine changes."""
    import outcomes_v2 as o
    import prospect_model as pm
    want = ("lodo", int(year)) if REVIEW_HOLDOUT and int(year) <= pm.LAST_SEASON_PLAYED else None
    if o.HOLDOUT != want:
        o.HOLDOUT, o._MODEL = want, None
        if tables is not None:
            tables.clear()
    return want


FALL_W = float(os.environ.get("REVIEW_FALL_WEIGHT", 0.85))  # fall shading (tests: scripts/holdout_review.py)
TRADE_VALUE = os.environ.get("TRADE_VALUE", "clock")  # draft-night trade verdicts (score_trades)
REVIEW_HOLDOUT = True  # leave-one-draft-out engine for past drafts (review, simulator)


def review_year(year, states, ctx, tables):
    engine_for(year, tables)
    picks, night = draft_night(year)
    b = board(year, picks)
    teams_st = {tm: states.state(tm, year) for tm in sorted(set(picks.team) | set(picks.original_team))}

    def table_for(tm):
        s = teams_st[tm]
        key = (s["start"], s["qb_from"], tuple(sorted(s["shift"].items())))
        if key not in tables:
            tables[key] = Table({"start": s["start"], "shift": dict(key[2]), "qb_from": key[1]}, ctx, fall_weight=FALL_W)
        return tables[key]

    from positions import blended, weights
    W = weights(year)
    taken_count = {}
    rows = []
    for _, p in picks.sort_values("pick").iterrows():
        tm, k = p.team, int(p.pick)
        tab = table_for(tm)
        counts = taken_count.setdefault(tm, {})
        lvl = lambda g: min(counts.get(g, 0), 2)  # each player taken at a position lowers the need
        avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
        pn = teams_st[tm]["p_need"]
        vals = np.array([blended(tab.lookup, lab, g, lvl(g), r, k, pn, W)
                         for lab, g, r in zip(avail.pos, avail.grp_label, avail["rank"])])
        best = avail.iloc[int(np.argmax(vals))] if len(avail) else None
        cons = avail.sort_values("rank").iloc[0] if len(avail) else None
        on_board = b[b.drafted_at == k]
        rank_actual = int(on_board["rank"].iloc[0]) if len(on_board) else OFF_BOARD_RANK
        v_actual = tab.lookup(p.grp, lvl(p.grp), rank_actual, k, pn) if p.grp != "SPEC" else np.nan
        rows.append({"year": year, "pick": k, "round": p["round"], "team": tm, "original_team": p.original_team,
                     "traded_on_night": p.traded_on_night, "player": p.pfr_player_name, "grp": p.grp,
                     "board_rank": rank_actual if rank_actual < OFF_BOARD_RANK else np.nan,
                     "value_actual": v_actual * 100,
                     "rec_player": best["name"] if best is not None else None,
                     "rec_grp": best.grp if best is not None else None,
                     "rec_rank": best["rank"] if best is not None else np.nan,
                     "rec_drafted_at": best.drafted_at if best is not None else np.nan,
                     "value_rec": vals.max() * 100 if len(vals) else np.nan,
                     "best_available_rank": cons["rank"] if cons is not None else np.nan,
                     "best_available_drafted_at": cons.drafted_at if cons is not None else np.nan,
                     "best_available_grp": cons.grp if cons is not None else None,
                     "best_available": cons["name"] if cons is not None else None})
        if p.grp != "SPEC":
            counts[p.grp] = counts.get(p.grp, 0) + 1
    df = pd.DataFrame(rows)
    df["value_gap"] = df.value_rec - df.value_actual
    st = pd.DataFrame(teams_st).T.reset_index(names="team").assign(year=year)
    return df, st, night


def score_trades(year, night, states_df, ctx, picks_df):
    """Draft-night trades made only of picks, scored for both teams (titles over 5 seasons).
    TRADE_VALUE "clock" (default, Nov 2026): the trade's earliest pick this year, the one on the clock when the deal
    is made, is valued for the team that gets it as the player taken there (his board rank shaded toward the pick,
    and position: known at that moment); the team giving it up keeps today's typical-player value (its alternative
    was its own choice there, not the buyer's target). Other picks this year as a typical player at the slot (a
    received pick as a QB if a QB was taken there). For trade-ups this predicts how trades turned out better than
    slot values (scripts/trade_target.py). "slot": the old rule."""
    pos_at = dict(zip(picks_df.pick, picks_df.grp))
    b = board(year, picks_df)
    on_board = {int(a): (float(r), g) for a, r, g in zip(b.drafted_at, b["rank"], b.grp_label) if pd.notna(a)}
    eng = DraftEngine.__new__(DraftEngine)
    eng.fall_weight = FALL_W
    sit = {r.team: {"start": r.start, "shift": r.shift, "qb_from": r.qb_from} for r in states_df.itertuples()}
    rows = []
    for tid, g in night.groupby("trade_id"):
        if g.pick_season.isna().any():
            continue  # involves a player
        teams_ = sorted(set(g.gave) | set(g.received))
        if len(teams_) != 2:
            continue
        cur = g[(g.pick_season == year) & g.pick_number.notna()]
        clock = int(cur.pick_number.min()) if TRADE_VALUE == "clock" and len(cur) else None
        for tm in teams_:
            def pk(rows_, receiving):
                out = []
                for r in rows_.itertuples():
                    yr = int(r.pick_season - year)
                    if yr == 0 and pd.notna(r.pick_number):
                        k = int(r.pick_number)
                        rk, gl = on_board.get(k, (float(OFF_BOARD_RANK), pos_at.get(k)))
                        if receiving and k == clock and gl in GROUPS:
                            out.append(Pick(k, pos=gl, talent=eng.effective_talent(min(rk, OFF_BOARD_RANK), k)))
                            continue
                        is_qb = receiving and pos_at.get(k) == "QB"
                        out.append(Pick(k, pos="QB" if is_qb else "nonqb"))
                    elif 0 < yr <= 2 and pd.notna(r.pick_round):
                        src_start = sit.get(r.gave, {"start": 0})["start"]
                        out.append(Pick(rnd=int(r.pick_round), year=yr, src={"start": src_start}))
                return out
            give, get = pk(g[g.gave == tm], False), pk(g[g.received == tm], True)
            if not give or not get or tm not in sit:
                continue
            r = evaluate(ctx, sit[tm], give, get, n=4000, seed=int(tid))
            rows.append({"year": year, "trade_id": tid, "team": tm,
                         "gave": ", ".join(_desc(p) for p in give), "got": ", ".join(_desc(p) for p in get),
                         "titles_change": r["change"] * 100})
    return pd.DataFrame(rows)


def _desc(p: Pick) -> str:
    return f"#{p.number}" + (" (QB)" if p.pos == "QB" else "") if p.year == 0 else f"{p.year} yr later R{p.rnd}"


def add_outcomes(df):
    out = pd.read_parquet(OUT / "pick_outcomes.parquet")[["draft_year", "pick", "tier"]]
    t = {(y, p): str(tr) for y, p, tr in zip(out.draft_year, out.pick, out.tier)}
    df["outcome_actual"] = [t.get((y, p)) for y, p in zip(df.year, df.pick)]
    df["outcome_rec"] = [t.get((y, int(p))) if pd.notna(p) else ("undrafted" if y <= 2021 else None)
                         for y, p in zip(df.year, df.rec_drafted_at)]
    return df


def main(first=YEARS[0], last=YEARS[-1]):
    import time
    ctx, states, tables = Context(), States(), {}
    allp, alls, allt = [], [], []
    for y in range(first, last + 1):
        t0 = time.time()
        df, st, night = review_year(y, states, ctx, tables)
        tr = score_trades(y, night, st, ctx, draft_night(y)[0])
        allp.append(df); alls.append(st); allt.append(tr)
        print(f"{y}: {len(df)} picks, {len(tr) // 2} draft-night pick trades, "
              f"{len(tables)} situation tables so far, {time.time() - t0:.0f}s", flush=True)
    picks = add_outcomes(pd.concat(allp, ignore_index=True))
    picks = add_realized_v2(picks, pd.concat(alls, ignore_index=True), ctx)
    st = pd.concat(alls, ignore_index=True)
    st["shift"] = st["shift"].astype(str)
    tr = pd.concat(allt, ignore_index=True)
    # keep other years already saved (so a subset of years can be added or redone)
    done = set(range(first, last + 1))
    for name, new in [("review_picks", picks), ("review_states", st), ("review_trades", tr)]:
        f = OUT / f"{name}.parquet"
        old = pd.read_parquet(f) if f.exists() else new.iloc[:0]
        out = pd.concat([old[~old.year.isin(done)], new], ignore_index=True).sort_values(
            [c for c in ["year", "pick", "trade_id", "team"] if c in new.columns])
        out.reset_index(drop=True).to_parquet(f)
    print("saved")



UDFA_CACHE = OUT / "udfa_pre2016_labels.parquet"


def udfa_pre2016_labels() -> pd.DataFrame:
    """Season values for undrafted players of the 2014-2015 classes, whom cfdarko doesn't label (its undrafted
    labels start with the 2016 class; before this every one of them counted as zero, e.g. La'el Collins, board
    #16 in 2015, a starting tackle). This project's measure, as in value.py: snap share in seasons 1-4 x his
    market price (best contract signed after his first, share of cap; veteran minimum if none), converted to
    cfdarko's units with the bridge's overall factor (outcomes_v2.CF_OVER_OURS_ALL, scripts/units_check.py).
    Columns as cfdarko's labels (index pfr_id; draft_year, player_name, is_udfa, value_y1-4)."""
    if UDFA_CACHE.exists():
        return pd.read_parquet(UDFA_CACHE)
    import polars as pl
    from outcomes_v2 import CF_OVER_OURS_ALL
    from outlook import season_shares
    P = pd.read_parquet(RAW / "players.parquet")
    P = P[P.draft_round.isna() & P.rookie_season.isin([2014, 2015]) & P.pfr_id.notna()]
    sh = season_shares(pl.read_parquet(RAW / "snap_counts.parquet")).to_pandas()
    sh = sh.set_index(["pfr_player_id", "season"]).share
    C = pd.read_parquet(RAW / "contracts.parquet")
    C = C[C.apy_cap_pct.notna()]
    rows = []
    for r in P.itertuples():
        y = int(r.rookie_season)
        later = C[(C.gsis_id == r.gsis_id) & (C.year_signed > y)].apy_cap_pct
        price = max(float(later.max()) if len(later) else 0.0, 0.005)
        v = [float(sh.get((r.pfr_id, y + t - 1), 0.0)) * price * CF_OVER_OURS_ALL for t in range(1, 5)]
        if sum(v) > 0:
            rows.append({"pfr_id": r.pfr_id, "draft_year": y, "player_name": r.display_name, "is_udfa": True,
                         **{f"value_y{t}": v[t - 1] for t in range(1, 5)}})
    out = pd.DataFrame(rows).set_index("pfr_id")
    out.to_parquet(UDFA_CACHE)
    return out


def band_of(rnd) -> np.ndarray:
    """Scorecard round bands: R1, R2-4, R5-7."""
    r = np.asarray(rnd)
    return np.select([r == 1, r <= 4], ["R1", "R2-4"], "R5-7")


def once_per_draft(d: pd.DataFrame, how: str = "mean", by_band: bool = True) -> pd.DataFrame:
    """The scorecard's unit: picks where the model's recommendation differs from the team's pick, each recommended
    player once per draft (and round band): the model can recommend one available player to many teams, but only
    one could have drafted him.
      how="first": at his first such pick: what a team following the model would have got (the conservative
                   level vs teams; the Past drafts headline). Fragile for comparing versions or rules: one that
                   moves him to a later pick changes his score without changing the choice (Watson vs Garrett at
                   #1 +13, vs Lattimore at #11 +32).
      how="mean":  averaged over the picks where he was the recommendation (stable for comparisons; flatters the
                   level vs teams, since later picks in a run face weaker team picks at a lower cost).
    Numeric columns per how; the rest from his first pick."""
    dis = d[d.player != d.rec_player].sort_values(["year", "pick"]).copy()
    dis["band"] = band_of(dis["round"])
    keys = ["year", "rec_player"] + (["band"] if by_band else [])
    if how == "first":
        return dis.drop_duplicates(keys).reset_index(drop=True)
    num = [c for c in dis.select_dtypes("number").columns if c not in keys]
    other = [c for c in dis.columns if c not in num and c not in keys]
    return dis.groupby(keys, as_index=False, sort=False).agg({**{c: "mean" for c in num}, **{c: "first" for c in other}})


AV_PER_SNAP_SEASON = 8.25  # drafted players 2014-2021: career AV ~ 8.25 x career snap-share sum (R2 0.82)


class Careers:
    """Seasons as a starter, career AV and best second-contract pay (% of cap) for any player in a draft,
    drafted or not: the old yardsticks came from draft records by (year, pick), so every undrafted player
    counted as zero (La'el Collins: 0 starts). Drafted players: PFR's seasons_started and w_av, and
    pick_outcomes' best_pay (best contract after the rookie deal). Undrafted players (found by name and rookie
    season in nflverse's player list): seasons with >= 50% of his side's snaps (outlook.START_SHARE; for
    drafted players 2014-2021 this tracks PFR's count, corr 0.94), AV_PER_SNAP_SEASON x career snap-share sum,
    best contract signed after his first season. Never on an NFL roster = 0.

    get(metric, year, drafted_at, name): metric "gs", "av" or "pay"."""
    def __init__(self):
        import polars as pl
        from outlook import START_SHARE, season_shares
        dp = pd.read_parquet(RAW / "draft_picks.parquet")
        po = pd.read_parquet(OUT / "pick_outcomes.parquet")
        self.pick = {"gs": dict(zip(zip(dp.season, dp.pick), dp.seasons_started)),
                     "av": dict(zip(zip(dp.season, dp.pick), dp.w_av)),
                     "pay": {k: v * 100 for k, v in zip(zip(po.draft_year, po.pick), po.best_pay)}}
        P = pd.read_parquet(RAW / "players.parquet")
        P = P[P.draft_round.isna() & P.pfr_id.notna() & P.rookie_season.notna()]
        sh = season_shares(pl.read_parquet(RAW / "snap_counts.parquet")).to_pandas()
        g = sh.groupby("pfr_player_id").share
        st, tot = g.apply(lambda x: int((x >= START_SHARE).sum())), g.sum()
        C = pd.read_parquet(RAW / "contracts.parquet")
        C = C[C.apy_cap_pct.notna()]
        self.udfa = {}
        for r in P.itertuples():
            y = int(r.rookie_season)
            later = C[(C.gsis_id == r.gsis_id) & (C.year_signed > y)].apy_cap_pct
            self.udfa[(y, _norm(r.display_name))] = {
                "gs": float(st.get(r.pfr_id, 0)), "av": AV_PER_SNAP_SEASON * float(tot.get(r.pfr_id, 0.0)),
                "pay": float(later.max()) * 100 if len(later) else 0.0}

    def get(self, metric: str, year, drafted_at, name=None) -> float:
        if pd.notna(drafted_at):
            return float(np.nan_to_num(self.pick[metric].get((int(year), int(drafted_at)), 0.0)))
        return self.udfa.get((int(year), _norm(name or "")), {}).get(metric, 0.0)


# Realized QB seasons (Nov 2026). "played" (default): a drafted QB's real seasons count wherever he was drafted; his
# season values already reflect whether he started (a rookie who sat delivers ~0), so the draft-day QB flag isn't
# needed to score what happened. "rule": the model's rule, as before: seasons before the team's draft-day qb_from
# count zero and a team with no flagged QB need gets nothing from a QB (Mahomes scored -0.8 for KC 2017).
REALIZED_QB = os.environ.get("REALIZED_QB", "played")
REALIZED_REC = os.environ.get("REALIZED_REC", "fair")  # "played": a recommended QB gets his real career in full


class RealizedV2:
    """What a player actually delivered, in title odds for one team's situation, at the cost of
    a given pick: cfdarko's season-by-season labels (same units and encoding as outcomes_v2),
    for classes with all four seasons played (through 2022); undrafted players of 2014-2015 from
    udfa_pre2016_labels. Players who never played in the NFL count as zero."""
    def __init__(self, ctx, n=2000, units=None, pos_pay=None):
        # units: see outcomes_v2.unit_factor; pos_pay: outcomes_v2.pay_factor (None = its default)
        self.ctx, self.n, self.units, self.pos_pay = ctx, n, units, pos_pay
        import prospect_model as pm
        L = pd.read_parquet(pm.LABEL_FILE)  # same label as the engine (prospect_model.LABEL)
        self.L = L.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
        extra = udfa_pre2016_labels()  # 2014-2015 undrafted players, missing from cfdarko's labels
        self.L = pd.concat([self.L, extra[~extra.index.isin(self.L.index)]])
        # undrafted players who reached the NFL (labeled from the 2016 class): find by name + class
        self.udfa = {(int(y), _norm(nm)): i for i, y, nm, u in
                     zip(self.L.index, self.L.draft_year, self.L.player_name, self.L.is_udfa) if u}
        dp = pd.read_parquet(RAW / "draft_picks.parquet")
        self.pfr_at = {(y, p): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
        self.base_cache = {}

    def titles(self, sit, grp, year, drafted_at, pick, name=None, scale=1.0, qb=None):
        """scale: multiply his seasons' values (e.g. a depth-chart adjustment); 1 = as he played.
        qb: QB rule for this call ("played" / "rule"; default REALIZED_QB)."""
        from outcomes_v2 import FAMILY, pay_factor, unit_factor
        from pick_value import baseline_paths, pick_effect
        if grp in ("K", "P") and year <= 2022:  # kickers / punters (src/specialists.py)
            return self._spec_many(sit, year, pick, [grp], [drafted_at], [name], 4)[0]
        if grp is None or grp == "SPEC" or year > 2022:
            return np.nan
        L, ctx = self.L, self.ctx
        pid = self.pfr_at.get((year, int(drafted_at))) if pd.notna(drafted_at) else self.udfa.get((year, _norm(name or "")))
        if pid in L.index:
            row = L.loc[pid]
            y = np.array([max(row[f"value_y{t}"], 0) for t in range(1, 5)]) / unit_factor(FAMILY.get(grp, grp), self.units) * scale * pay_factor(grp, self.pos_pay)
        else:
            y = np.zeros(4)
        y5 = np.r_[y, y[-1]]
        price = np.array([max(y5.max(), 1e-6)])
        key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
        if key not in self.base_cache:
            b = baseline_paths(sit, np.random.default_rng(0), self.n)
            self.base_cache[key] = (b, ctx.sb(b).sum(axis=1).mean())
        b, b0 = self.base_cache[key]
        s2 = dict(sit); s2["qb_from"] = 1 if (qb or REALIZED_QB) == "played" else (99 if sit["qb_from"] is None else sit["qb_from"])
        pts = pick_effect((y5 / price[0])[None, :], price, int(pick), grp, s2, ctx.cost, ctx.bar, ctx.rates)
        return (ctx.sb(b + pts).sum(axis=1).mean() - b0) * 100

    def titles_rec(self, sit, p_need, grp, year, drafted_at, pick, name=None):
        """A player the team did NOT draft (the model's recommendation, the best on the board), scored as the
        model predicts him (REALIZED_REC "fair", default since Nov 2026): a QB for a team whose QB need is real
        only with probability p_need counts p_need x his real career + (1 - p_need) x sitting through the window
        (cost charged), the same two branches as the prediction. Before, his real career counted in full: it
        happened at the team that drafted him, usually one that started him (scripts/qb_need_yardstick.py)."""
        v = self.titles(sit, grp, year, drafted_at, pick, name)
        if REALIZED_REC != "fair" or grp != "QB" or not (p_need < 1) or pd.isna(v):
            return v
        sat = self.titles({**sit, "qb_from": None}, grp, year, drafted_at, pick, name, qb="rule")
        return p_need * v + (1 - p_need) * sat

    def _spec_many(self, sit, year, pick, grps, drafted_at, names, n_played):
        import specialists as spx
        from pick_value import baseline_paths
        if not hasattr(self, "_spec"):
            self._spec = spx.Realized()
        key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
        if key not in self.base_cache:
            b = baseline_paths(sit, np.random.default_rng(0), self.n)
            self.base_cache[key] = (b, self.ctx.sb(b).sum(axis=1).mean())
        return np.array([spx.realized_titles(self._spec, g, self.base_cache[key], self.ctx, year, pick, at, nm, n_played)
                         for g, at, nm in zip(grps, drafted_at, names)])

    def last_season(self) -> int:
        """Last NFL season with labels (e.g. 2025: the 2023 class has three seasons)."""
        L = self.L
        return int(max((L.draft_year[L[f"value_y{t}"].notna()] + t - 1).max() for t in range(1, 5)))

    def seasons_played(self, year) -> int:
        return int(np.clip(self.last_season() - year + 1, 0, 4))

    def titles_many(self, sit, year, pick, grps, drafted_at, names, partial=False) -> np.ndarray:
        """titles() for many players at one pick (same numbers, vectorized by position group).
        partial=True also scores classes still on their rookie deals, "so far": only the seasons
        played count, both his value and the pick's cost (later seasons add nothing)."""
        from outcomes_v2 import FAMILY, pay_factor, unit_factor
        from pick_value import baseline_paths, pick_effect
        out = np.full(len(grps), np.nan)
        n_played = self.seasons_played(year) if partial else (4 if year <= 2022 else 0)
        if n_played == 0:
            return out
        L, ctx = self.L, self.ctx
        key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
        if key not in self.base_cache:
            b = baseline_paths(sit, np.random.default_rng(0), self.n)
            self.base_cache[key] = (b, ctx.sb(b).sum(axis=1).mean())
        b, b0 = self.base_cache[key]
        s2 = dict(sit); s2["qb_from"] = 1 if REALIZED_QB == "played" else (99 if sit["qb_from"] is None else sit["qb_from"])
        vals = L[[f"value_y{t}" for t in range(1, 5)]].clip(lower=0).fillna(0.0)
        grps = np.asarray(grps, object)
        sp = np.flatnonzero(np.isin(grps, ["K", "P"]))  # kickers / punters (src/specialists.py)
        if len(sp):
            out[sp] = self._spec_many(sit, year, pick, grps[sp], [drafted_at[i] for i in sp], [names[i] for i in sp], n_played)
        for g in set(grps) - {None, "SPEC", "K", "P"}:
            idx = np.flatnonzero(grps == g)
            y = np.zeros((len(idx), 4))
            for r, i in enumerate(idx):
                at = drafted_at[i]
                pid = self.pfr_at.get((year, int(at))) if pd.notna(at) else self.udfa.get((year, _norm(names[i] or "")))
                if pid in vals.index:
                    y[r] = vals.loc[pid].to_numpy(float)
            y *= pay_factor(g, self.pos_pay) / unit_factor(FAMILY.get(g, g), self.units)
            y5 = np.column_stack([y, y[:, -1]])
            price = np.maximum(y5.max(axis=1), 1e-6)
            pts = pick_effect(y5 / price[:, None], price, int(pick), g, s2, ctx.cost, ctx.bar, ctx.rates)
            if n_played < 4:
                pts[:, n_played:] = 0.0  # seasons not played yet: neither value nor cost
            for c in range(0, len(idx), 64):  # (players, paths, seasons) in chunks
                p_ = pts[c:c + 64]
                out[idx[c:c + 64]] = (ctx.sb(b[None, :, :] + p_[:, None, :]).sum(axis=2).mean(axis=1) - b0) * 100
        return out


def add_realized_v2(df, states, ctx, n=2000, units=None):
    """Realized value (RealizedV2) for the team's pick, the model's pick and the best player
    left on the consensus board. units: outcomes_v2.unit_factor (None = outcomes_v2.UNITS)."""
    sits = {(r.year, r.team): {"start": r.start, "shift": r.shift if isinstance(r.shift, dict) else eval(r.shift),
                               "qb_from": None if pd.isna(r.qb_from) else int(r.qb_from)}
            for r in states.itertuples()}
    pneed = {(r.year, r.team): (r.p_need if "p_need" in states.columns else 1.0) for r in states.itertuples()}
    R = RealizedV2(ctx, n, units=units)
    df["real_team"] = [R.titles(sits[(y, tm)], gg, y, a, k, n_) for y, tm, gg, a, k, n_ in
                       zip(df.year, df.team, df.grp, df.pick, df.pick, df.player)]
    for col, g, at, nm in [("real_model", "rec_grp", "rec_drafted_at", "rec_player"),
                           ("real_consensus", "best_available_grp", "best_available_drafted_at", "best_available")]:
        df[col] = [R.titles_rec(sits[(y, tm)], pneed[(y, tm)], gg, y, a, k, n_) for y, tm, gg, a, k, n_ in
                   zip(df.year, df.team, df[g], df[at], df.pick, df[nm])]
    return df


def rescore():
    """Recompute realized value columns of the saved review (after a yardstick change)."""
    from trade import Context
    f = OUT / "review_picks.parquet"
    df = add_realized_v2(pd.read_parquet(f), pd.read_parquet(OUT / "review_states.parquet"), Context())
    df.to_parquet(f)
    print("rescored", len(df))


# ---------- 5. trades as decisions (for one team's view) ----------

def _sit(r) -> dict:
    sh = r.shift if isinstance(r.shift, dict) else eval(r.shift)
    return {"start": r.start, "shift": sh, "qb_from": None if pd.isna(r.qb_from) else int(r.qb_from)}


def trade_decisions(year, picks_df, states_df, ctx, tables) -> pd.DataFrame:
    """One row per (draft-night trade, team): what the team did vs the model's better option
    of (a) keeping its pick(s) and taking the best player left, or (b) making the same trade
    and then taking its own choices with the picks received.

    Same currency as the pick rows (titles from each player at his pick, for this team's
    situation, with the board as it actually stood at that pick); picks in later drafts are
    valued like the trade calculator (by the giving team's likely finish). The team's side
    uses the players it actually took with the picks it received."""
    pk, night = draft_night(year)
    if night.empty:
        return pd.DataFrame()
    engine_for(year, tables)
    b = board(year, pk)
    P = picks_df[picks_df.year == year]
    used, val_act, player = (dict(zip(P.pick, P[c])) for c in ["team", "value_actual", "player"])
    from positions import blended, weights
    W = weights(year)
    sit = {r.team: _sit(r) for r in states_df[states_df.year == year].itertuples()}
    pneed = {r.team: (r.p_need if "p_need" in states_df.columns else 1.0) for r in states_df[states_df.year == year].itertuples()}

    def table(tm):
        s = sit[tm]
        key = (s["start"], s["qb_from"], tuple(sorted(s["shift"].items())))
        if key not in tables:
            tables[key] = Table({"start": s["start"], "shift": dict(key[2]), "qb_from": key[1]}, ctx, fall_weight=FALL_W)
        return tables[key]

    def best(tm, k, taken=(), extra=None):
        """The model's choice for team tm at pick k: (titles, player, group). `taken` = players
        it already chose in this same option, `extra` = their position counts."""
        tab = table(tm)
        mine = P[(P.team == tm) & (P.pick < k) & (P.grp != "SPEC")].grp.value_counts().to_dict()
        for g_, c_ in (extra or {}).items():
            mine[g_] = mine.get(g_, 0) + c_
        avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC") & ~b["name"].isin(list(taken))]
        if avail.empty:
            return 0.0, None, None
        pn = pneed.get(tm, 1.0)
        vals = np.array([blended(tab.lookup, lab, g, min(int(mine.get(g, 0)), 2), r, k, pn, W)
                         for lab, g, r in zip(avail.pos, avail.grp_label, avail["rank"])])
        i = int(np.argmax(vals))
        return vals[i] * 100, avail["name"].iloc[i], avail.grp_label.iloc[i]

    def plan(tm, ks):
        """The model's choices at several picks in one option, in order, no player twice."""
        out, taken, extra = [], [], {}
        for k in ks:
            v, n, g_ = best(tm, k, taken, extra)
            out.append((v, n))
            if n is not None:
                taken.append(n); extra[g_] = extra.get(g_, 0) + 1
        return out

    def used_value(tm, k):
        """What the team got out of a pick it received: its actual player if it used the pick
        (specialists ~0), otherwise (re-traded the same night) the model's value for it."""
        if used.get(k) == tm:
            v = val_act.get(k)
            return (0.0 if pd.isna(v) else float(v)), f"#{k} {player.get(k)}"
        return best(tm, k)[0], f"#{k} (traded on)"

    rows = []
    for tid, g in night.groupby("trade_id"):
        teams_ = sorted(set(g.gave) | set(g.received))
        cur = g[(g.pick_season == year) & g.pick_number.notna()]
        if len(teams_) != 2 or cur.empty:
            continue
        anchor = int(cur.pick_number.min())
        has_player = g.pick_season.isna().any()
        for tm in teams_:
            gave, got = g[g.gave == tm], g[g.received == tm]
            cur_of = lambda x: sorted(int(v) for v in x[(x.pick_season == year) & x.pick_number.notna()].pick_number)
            gave_cur, got_cur = cur_of(gave), cur_of(got)

            def fut(x):
                f = x[(x.pick_season > year) & x.pick_round.notna()]
                return [(int(r.pick_season - year), int(r.pick_round), r.gave) for r in f.itertuples()]
            fut_gave, fut_got = fut(gave), fut(got)
            # a row without a pick is a player changing teams (on pick rows the log names the draftee)
            plyrs = lambda x: x[x.pick_season.isna()].pfr_name.dropna()
            desc = lambda cur_, fut_, plyr: ", ".join([f"#{k}" for k in cur_] + [f"{year + y} R{rd}" for y, rd, _ in fut_]
                                                      + ([f"player: {n}" for n in plyr]))
            row = {"year": year, "trade_id": tid, "team": tm, "pick": anchor,
                   "gave": desc(gave_cur, fut_gave, plyrs(gave)), "got": desc(got_cur, fut_got, plyrs(got))}
            if has_player or tm not in sit:
                rows.append({**row, "note": "includes a player: not scored"})
                continue
            fv = 0.0
            if fut_gave or fut_got:
                mk = lambda f: [Pick(rnd=rd, year=y, src={"start": sit.get(src, {"start": 0})["start"]}) for y, rd, src in f if y <= 2]
                fv = evaluate(ctx, sit[tm], mk(fut_gave), mk(fut_got), n=4000, seed=int(tid))["change"] * 100
            team_parts = [used_value(tm, k) for k in got_cur]
            trade_best = plan(tm, got_cur)
            keep_best = plan(tm, gave_cur)
            team_v = sum(v for v, _ in team_parts) + fv
            v_trade = sum(v for v, _ in trade_best) + fv
            v_keep = sum(v for v, _ in keep_best)
            if v_keep >= v_trade:
                choice = "don't trade" + "".join(f"; #{k}: {n}" for k, (_, n) in zip(gave_cur, keep_best))
            else:
                choice = "same trade" + "".join(f"; #{k}: {n}" for k, (_, n) in zip(got_cur, trade_best))
            rows.append({**row, "team_did": "; ".join(n for _, n in team_parts), "value_team": team_v,
                         "value_keep": v_keep, "value_same_trade": v_trade, "model_choice": choice,
                         "value_model": max(v_keep, v_trade), "future_value": fv, "note": None})
    return pd.DataFrame(rows)


def rescore_trades():
    """Recompute review_trades.parquet (score_trades) for every reviewed year from the saved states."""
    from trade import Context
    st = pd.read_parquet(OUT / "review_states.parquet")
    st["shift"] = [x if isinstance(x, dict) else eval(x) for x in st["shift"]]
    st["qb_from"] = pd.Series([None if pd.isna(q) else int(q) for q in st.qb_from], index=st.index, dtype=object)
    ctx, out = Context(), []
    for y in sorted(st.year.unique()):
        engine_for(int(y))  # as in the review run: the draft's own leave-one-draft-out engine
        picks, night = draft_night(int(y))
        out.append(score_trades(int(y), night, st[st.year == y], ctx, picks))
        print(f"{y}: {len(out[-1]) // 2} trades", flush=True)
    pd.concat(out, ignore_index=True).sort_values(["year", "trade_id", "team"]).reset_index(drop=True).to_parquet(OUT / "review_trades.parquet")


def build_trade_decisions(years=YEARS):
    """Run trade_decisions for all reviewed years from the saved review files."""
    picks = pd.read_parquet(OUT / "review_picks.parquet")
    states = pd.read_parquet(OUT / "review_states.parquet")
    ctx, tables, out = Context(), {}, []
    for y in years:
        if y in picks.year.values:
            d = trade_decisions(y, picks, states, ctx, tables)
            out.append(d)
            print(f"{y}: {len(d)} team-trade rows, {len(tables)} tables", flush=True)
    df = pd.concat(out, ignore_index=True)
    f = OUT / "review_trade_decisions.parquet"
    if f.exists():  # keep other years already built
        old = pd.read_parquet(f)
        df = pd.concat([old[~old.year.isin(list(years))], df], ignore_index=True).sort_values(["year", "pick", "trade_id"])
    df.reset_index(drop=True).to_parquet(f)
    return df


if __name__ == "__main__":
    if sys.argv[1:2] == ["verdicts"]:  # recompute draft-night trade verdicts only (review_trades.parquet)
        rescore_trades()
    elif sys.argv[1:2] == ["trades"]:
        build_trade_decisions()
    elif sys.argv[1:2] == ["rescore"]:
        rescore()
    else:
        a = [int(x) for x in sys.argv[1:]]
        main(*a)
        build_trade_decisions(range(a[0], (a[1] if len(a) > 1 else YEARS[-1]) + 1) if a else YEARS)
