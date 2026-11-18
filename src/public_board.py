"""Public rankings for people without a scouting department.

1. parse_board(text): turn a big board copied from any website into a table.
   Handles the usual copy formats, one player per line:
     "1. Jeremiah Smith, WR, Ohio State"
     "1  Jeremiah Smith  WR  Ohio State"        (tabs or spaces, e.g. a copied table)
     "Jeremiah Smith | WR | Ohio State"         (no rank: line order is the rank)
     "1. Jeremiah Smith (WR, Ohio State)"
   Lines without a recognizable position are skipped and reported.

2. consensus(boards): average several boards into one. Each player's score is
   his average log rank; a player missing from a board counts as ranked just past
   that board's end. The consensus order is the talent ranking AND the expected
   draft spot, because our tests (prospects.py) found no public data that predicts
   careers better than where the consensus says a player will go.

3. athletic_nudge(board, year): after the combine, nudge each player's talent rank
   by the small signals that held up in testing (agility drills; for TE, edge
   rushers and safeties, testing in general). Before the combine: no change.
"""
import re

import numpy as np
import pandas as pd

from draftday import POS, SPECIALISTS

POS_TOKENS = set(POS) | SPECIALISTS | {"OL", "IOL", "EDGE", "DL", "DB", "ATH", "QB"}
ALIASES = {"EDGE": "EDGE", "ED": "EDGE", "IOL": "OG", "OL": "OT", "DL": "DT", "DB": "CB", "SAF": "S",
           "HB": "RB", "OLB": "EDGE", "ILB": "LB", "MLB": "LB", "NT": "DT", "DE": "EDGE", "PK": "K"}


def _norm(name: str) -> str:
    """Name key for matching across boards: lowercase, no punctuation or suffixes."""
    s = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s).split())


def parse_board(text: str) -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = [t for t in re.split(r"[\t|,()]+|\s{2,}|\s+[-–]\s+|\s", line) if t]
        rank = None
        if toks and re.fullmatch(r"#?\d+\.?", toks[0]):
            rank = int(re.sub(r"\D", "", toks[0])); toks = toks[1:]
        pos_i = next((i for i, t in enumerate(toks) if t.upper().strip(".:") in POS_TOKENS and i > 0), None)
        if pos_i is None:
            skipped.append(line)
            continue
        pos = toks[pos_i].upper().strip(".:")
        name = " ".join(toks[:pos_i])
        school = " ".join(t for t in toks[pos_i + 1:] if not re.fullmatch(r"[\d.]+", t))
        rows.append({"rank": rank, "name": name, "pos": ALIASES.get(pos, pos), "school": school})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, skipped
    if df["rank"].isna().any():  # no ranks given: use line order
        df["rank"] = np.arange(1, len(df) + 1)
    df = df.drop_duplicates("name").sort_values("rank").reset_index(drop=True)
    return df, skipped


def consensus(boards: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average log rank across boards. Returns name, pos, school, rank (1..n),
    adp (expected pick), n_boards, and the range of ranks across boards."""
    frames = []
    for src, b in boards.items():
        frames.append(b.assign(key=b.name.map(_norm), src=src, board_len=len(b)))
    allb = pd.concat(frames, ignore_index=True)
    players = allb.sort_values("rank").drop_duplicates("key")[["key", "name", "pos", "school"]]
    scores = []
    for src, b in allb.groupby("src"):
        miss_rank = b.board_len.iloc[0] + 25
        r = players.key.map(dict(zip(b.key, b["rank"]))).fillna(miss_rank)
        scores.append(np.log(r + 10))
    S = np.column_stack(scores)
    present = np.column_stack([players.key.isin(b.key) for _, b in allb.groupby("src")])
    out = players.assign(score=S.mean(axis=1), n_boards=present.sum(axis=1),
                         best=np.exp(S.min(axis=1)) - 10, worst=np.exp(S.max(axis=1)) - 10)
    out = out.sort_values(["score", "n_boards"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["adp"] = (np.exp(out.score) - 10).clip(lower=1)
    return out.drop(columns=["key", "score"])


def athletic_nudge(board: pd.DataFrame, year: int) -> tuple[pd.DataFrame, str]:
    """Shift talent rank by the age/athletic signals from prospects.py, once the
    combine for `year` is published by nflverse. Returns (board, note)."""
    try:
        import nflreadpy as nfl
        comb = nfl.load_combine(seasons=[year]).to_pandas()
    except Exception:
        comb = pd.DataFrame()
    board = board.assign(talent_rank=board["rank"].astype(float), nudge="")
    if comb.empty:
        return board, (f"No {year} combine results yet (published late February). "
                       "Talent rank = consensus rank until then.")
    from prospects import TESTS, build, fit_predict
    hist = build()  # historical prospects with z-scored tests
    _, m = fit_predict(hist, hist.iloc[:5], "age+athletic", 0.1)
    coef = m.params
    comb["key"] = comb.player_name.map(_norm)
    ft, inch = comb.ht.str.split("-", expand=True).astype(float).T.values
    comb["ht_in"] = ft * 12 + inch
    comb["speed_score"] = comb.wt * 200 / comb.forty ** 4
    ref = hist.groupby("grp")[TESTS].agg(["mean", "std"])
    lp_coef = coef["lp"]
    by_key = comb.set_index("key")
    moved = 0
    for i, r in board.iterrows():
        k = _norm(r["name"])
        if k not in by_key.index or r.grp not in ref.index:
            continue
        c = by_key.loc[k] if isinstance(by_key.loc[k], pd.Series) else by_key.loc[k].iloc[0]
        delta = 0.0
        for t in TESTS:
            if pd.notna(c.get(t)) and f"z_{t}" in coef:
                z = (c[t] - ref.loc[r.grp, (t, "mean")]) / ref.loc[r.grp, (t, "std")]
                delta += coef[f"z_{t}"] * (-z if t in {"forty", "cone", "shuttle"} else z)
        if pd.isna(c.get("forty")) and "no_forty" in coef:
            delta += coef["no_forty"]
        # change in log(rank+10) worth the same odds of starter-or-better (the model fit_predict returns);
        # capped at about +-20% because the evidence behind it is weak (0.6% better forecasts than
        # consensus alone). lp_coef is negative, so better odds (delta > 0) move the rank up the board.
        shift = float(np.clip(delta / lp_coef, -0.2, 0.2))
        board.loc[i, "talent_rank"] = max(1.0, np.exp(np.log(r["rank"] + 10) + shift) - 10)
        board.loc[i, "nudge"] = f"{board.loc[i, 'talent_rank'] - r['rank']:+.0f}"
        moved += 1
    return board, f"Combine {year}: athletic nudge applied to {moved} players."
