"""Other-sport signals for NFL prospects.

1. MLB-drafted: every MLB June draft 2005-2023 (MLB stats API, statsapi.mlb.com/api/v1/draft/{year},
   saved to data/raw/mlb_draft/, not in git) matched to the prospect universe.
   Match rule: same normalized name (MLB full name, or legal first + last), MLB draft year between
   NFL class - 7 and the NFL class, and, when an NFL birth date is known (nflreadpy players), the
   same birth date. Without an NFL birth date the match must be unique by name in that window.
2. Other sports in high school / college, from the BACKGROUND text of Brugler's guides (2019-2023).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from public_board import _norm

RAW = Path("data/raw/mlb_draft")
OUT = Path("data/derived")


def mlb_drafts() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.json")):
        d = json.loads(f.read_text())
        for r in d["drafts"]["rounds"]:
            for p in r["picks"]:
                per = p.get("person") or {}
                if not per:
                    continue
                rows.append({
                    "mlb_year": int(p["year"]), "mlb_round": r["round"], "mlb_pick": p.get("pickNumber"),
                    "mlb_name": per.get("fullName"), "first": per.get("firstName"), "last": per.get("lastName"),
                    "use": per.get("useName"), "birth": per.get("birthDate"), "mlb_school": (p.get("school") or {}).get("name"),
                    "mlb_pos": (per.get("primaryPosition") or {}).get("abbreviation"),
                    "mlb_debut": per.get("mlbDebutDate"), "mlb_team": (p.get("team") or {}).get("name"),
                })
    m = pd.DataFrame(rows)
    m["round_num"] = pd.to_numeric(m.mlb_round.str.extract(r"(\d+)")[0], errors="coerce")
    m["hs"] = m.mlb_school.fillna("").str.contains(r"\bHS\b|High School|Academy|Prep", regex=True)
    return m


def nfl_births() -> pd.DataFrame:
    import nflreadpy as nr
    p = nr.load_players().to_pandas()[["display_name", "first_name", "last_name", "birth_date", "pfr_id", "draft_year",
                                       "rookie_season"]]
    return p


def match_mlb(uni: pd.DataFrame) -> pd.DataFrame:
    """One row per (universe player, MLB draft selection) match."""
    m = mlb_drafts()
    keys = []
    for i, r in m.iterrows():
        ks = {_norm(r.mlb_name), _norm(f"{r['first']} {r['last']}"), _norm(f"{r.use} {r['last']}")}
        for k in ks:
            if k:
                keys.append((k, i))
    mk = pd.DataFrame(keys, columns=["key", "mi"]).drop_duplicates()
    u = uni[["player_id", "player_name", "cls", "pfr_id", "age_at_draft", "grp"]].copy()
    u["key"] = u.player_name.map(_norm)
    c = u.merge(mk, on="key").merge(m, left_on="mi", right_index=True)
    c = c[(c.mlb_year >= c.cls - 7) & (c.mlb_year <= c.cls)]
    # birth dates on the NFL side
    nb = nfl_births()
    by_pfr = nb.dropna(subset=["pfr_id", "birth_date"]).drop_duplicates("pfr_id").set_index("pfr_id").birth_date
    nb["key"] = nb.display_name.map(_norm)
    nb["yr"] = nb.draft_year.fillna(nb.rookie_season)
    c["nfl_birth"] = c.pfr_id.map(by_pfr)
    miss = c.nfl_birth.isna()
    if miss.any():
        alt = nb.dropna(subset=["birth_date"]).groupby(["key", "yr"]).birth_date.agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
        c.loc[miss, "nfl_birth"] = [alt.get((k, y)) for k, y in zip(c.key[miss], c.cls[miss])]
    c["birth_ok"] = np.where(c.nfl_birth.notna() & c.birth.notna(),
                             (pd.to_datetime(c.nfl_birth, errors="coerce") - pd.to_datetime(c.birth, errors="coerce")).dt.days.abs() <= 1,
                             np.nan)
    # age check when no birth date: MLB birth date vs age at NFL draft (late April of the class year)
    approx_age = (pd.to_datetime(c.cls.astype(str) + "-04-25") - pd.to_datetime(c.birth, errors="coerce")).dt.days / 365.25
    c["age_ok"] = (approx_age - c.age_at_draft).abs() <= 1.0
    keep = (c.birth_ok == 1) | (c.birth_ok.isna() & c.age_at_draft.notna() & c.age_ok)
    c = c[keep].copy()
    c["confirmed"] = c.birth_ok == 1
    # unconfirmed: require a unique universe player for that name in that class
    dup = u.groupby(["key", "cls"]).size()
    c = c[c.confirmed | (c.set_index(["key", "cls"]).index.map(dup) == 1)]
    return c.drop(columns=["mi"]).sort_values(["cls", "player_name", "mlb_year"])


def mlb_flags(uni: pd.DataFrame) -> pd.DataFrame:
    """Per universe player: drafted by MLB (any), best round, times drafted, any MLB debut."""
    c = match_mlb(uni)
    g = c.groupby("player_id")
    return pd.DataFrame({"mlb_drafted": True, "mlb_best_round": g.round_num.min(), "mlb_times": g.size(),
                         "mlb_confirmed": g.confirmed.any(), "mlb_debut": g.mlb_debut.apply(lambda s: s.notna().any()),
                         "mlb_pos": g.mlb_pos.first()}).reset_index()


# ---- other sports, from the guides' BACKGROUND text
SPORTS = {
    "basketball": r"basketball|\bhoops\b",
    "track": r"track and field|track & field|track team|\bran track\b|on the track team|\btrack\b(?= (?:star|athlete|standout|career|season))|"
             r"\b(?:100|200|400|110)[- ]?(?:m\b|meters?|meter)|hurdle|relay|sprinter|sprints?\b(?= )|long jump|high jump|triple jump|"
             r"shot put|discus|javelin|state (?:champion|title) in the (?:100|200|400)",
    "throws": r"shot put|discus|javelin",
    "baseball": r"baseball|\bmlb\b|pitcher|shortstop|outfielder",
    "wrestling": r"wrestl",
    "soccer_lax_rugby": r"soccer|lacrosse|rugby",
    "multi_unnamed": r"(?:two|three|four|multi)[- ]sport",
}


def text_sports(beast: pd.DataFrame) -> pd.DataFrame:
    txt = beast.background.fillna("")
    f = pd.DataFrame({k: txt.str.contains(v, case=False, regex=True) for k, v in SPORTS.items()}, index=beast.index)
    # "basketball scholarship offers" etc. still mean he played; "his father played basketball" doesn't
    fam = r"(?:father|dad|mother|mom|brother|sister|uncle|cousin|grandfather)[^.]{0,80}"
    for k in ["basketball", "baseball", "track", "wrestling"]:
        only_family = txt.str.contains(fam + SPORTS[k], case=False, regex=True) & \
            ~txt.str.replace(fam + "(?:" + SPORTS[k] + ")", "", case=False, regex=True).str.contains(SPORTS[k], case=False, regex=True)
        f[k] &= ~only_family
    f["any_other"] = f[["basketball", "track", "baseball", "wrestling", "soccer_lax_rugby", "multi_unnamed"]].any(axis=1)
    return f
