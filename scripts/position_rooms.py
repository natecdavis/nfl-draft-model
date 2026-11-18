"""Which room does a drafted defender play in? Board label + combine size (room check, Nov 16 2026).

The simulator's room check counts a pick in the room of his consensus-board label. Spot checks found 6.6% of
round 1-3 picks counted in the wrong room, nearly all on defense: 20 of 150 board CBs played mostly S (Tyler Nubin,
2024), EDGE vs interior DL both ways (Laiatu Latu, Jared Verse labeled DL; L.J. Collier, Logan Hall played inside),
off-ball LBs who rushed (Vic Beasley, Dallas Turner). Rooms count veterans by PFF position (players.pff_position),
so the target here is the same label for the drafted player.

Families: DB (board CB / S labels -> room CB or S) and front (EDGE / DL / LB labels -> ED, IDL or LB). Drafted board
players 2014-2025 with a PFF position in the family. Predictors known before the draft: the board label (Arif
Hasan's own codes, e.g. NCB, DL5T, OB), combine height / weight / 40 (missing -> label mean + flag).
Models, each draft predicted from the other drafts:
  label    the room of the board label (current)
  freq     P(room | label code), smoothed (positions.weights style)
  size     multinomial logit: label code + weight + height + 40
Scored by accuracy of the top room and log loss; also the share of picks' expected snaps counted in the wrong room.

Usage: python scripts/position_rooms.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
from outlook import RAW  # noqa: E402

PFF = {"ED": "ED", "DI": "IDL", "LB": "LB", "CB": "CB", "S": "S"}
FAMILY = {"CB": "DB", "S": "DB", "ED": "FRONT", "IDL": "FRONT", "LB": "FRONT"}
ROOMS = {"DB": ["CB", "S"], "FRONT": ["ED", "IDL", "LB"]}
YEARS = range(2014, 2027)


def prospects() -> pd.DataFrame:
    """Defensive board players 2014-2026 (label, rank, drafted_at) with combine measures and, if drafted and in
    PFF's player list, the room he plays (target)."""
    from simulator import pool
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pid = {(int(y), int(p)): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    ppos = {i: PFF.get(q) for i, q in zip(pl.pfr_id, pl.pff_position)}
    cb = pd.read_parquet(RAW / "combine.parquet")
    cb["key"] = cb.player_name.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    by_id = cb.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    by_name = cb.drop_duplicates(["season", "key"]).set_index(["season", "key"])
    ht = lambda s: (int(s.split("-")[0]) * 12 + int(s.split("-")[1])) if isinstance(s, str) and "-" in s else np.nan
    rows = []
    for y in YEARS:
        P = pool(y)
        for n, g, lab, rk, at in zip(P["name"], P.grp, P.label, P["rank"], P.drafted_at):
            if g not in FAMILY:
                continue
            p = pid.get((y, int(at))) if pd.notna(at) else None
            if p is not None and p in by_id.index:
                c = by_id.loc[p]
            else:
                k = (y, str(n).lower().replace(".", "").replace("'", "").strip())
                c = by_name.loc[k] if k in by_name.index else None
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[0]
            rows.append({"year": y, "name": n, "grp": g, "label": lab if isinstance(lab, str) else g, "rank": rk,
                         "drafted_at": at, "pfr": p, "fam": FAMILY[g],
                         "wt": float(c.wt) if c is not None and pd.notna(c.wt) else np.nan,
                         "ht": ht(c.ht) if c is not None else np.nan,
                         "forty": float(c.forty) if c is not None and pd.notna(c.forty) else np.nan,
                         "room": ppos.get(p) if p else None})
    D = pd.DataFrame(rows)
    D["target"] = [r if r is not None and FAMILY.get(r) == f else None for r, f in zip(D.room, D.fam)]
    return D


def features(D: pd.DataFrame, fam: str, ref: pd.DataFrame) -> np.ndarray:
    """Label one-hots (codes seen in ref) + standardized size, missing -> the label's mean in ref, + flags."""
    labs = sorted(ref[ref.fam == fam].label.unique())
    X = [np.array([[l == c for c in labs] for l in D.label], float)]
    for v in ["wt", "ht", "forty"]:
        r = ref[ref.fam == fam]
        mu = r.groupby("label")[v].mean()
        x = D[v].fillna(D.label.map(mu)).fillna(r[v].mean()).to_numpy(float)
        X += [((x - r[v].mean()) / r[v].std())[:, None], D[v].isna().to_numpy(float)[:, None]]
    return np.hstack(X)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, fam: str, kind: str) -> np.ndarray:
    """P(room) for test rows of a family, columns ROOMS[fam]."""
    rooms = ROOMS[fam]
    tr = train[(train.fam == fam) & train.target.notna()]
    if kind == "label":
        return np.array([[1.0 if g == r else 0.0 for r in rooms] for g in test.grp])
    if kind == "freq":
        out = []
        for lab, g in zip(test.label, test.grp):
            c = tr[tr.label == lab].target.value_counts()
            w = {r: c.get(r, 0) + (5.0 if r == g else 0.0) for r in rooms}
            s = sum(w.values())
            out.append([w[r] / s for r in rooms])
        return np.array(out)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(features(tr, fam, tr), tr.target)
    p = m.predict_proba(features(test, fam, tr))
    return np.column_stack([p[:, list(m.classes_).index(r)] if r in m.classes_ else np.zeros(len(test)) for r in rooms])


def main():
    import strict_follower as sf
    D = prospects()
    sh = sf.season_shares()
    f = sf.expected_curve(sf.board_players(), sh.groupby(["pfr_player_id", "season"]).share.sum())
    D["e1"] = [float(f(g, r if pd.notna(r) else 330)[0]) for g, r in zip(D.grp, D["rank"])]
    T = D[D.target.notna() & (D.year <= 2025)]
    print(f"drafted defenders with a PFF room: {len(T)} (combine weight known {T.wt.notna().mean() * 100:.0f}%)")
    for fam in ["DB", "FRONT"]:
        t = T[T.fam == fam]
        print(f"\n{fam}: board label room -> PFF room (counts)")
        print(pd.crosstab(t.grp, t.target).to_string())
    res = []
    for kind in ["label", "freq", "size"]:
        for fam in ["DB", "FRONT"]:
            for y in sorted(T.year.unique()):
                te = T[(T.fam == fam) & (T.year == y)]
                if te.empty:
                    continue
                p = fit_predict(T[T.year != y], te, fam, kind)
                idx = np.array([ROOMS[fam].index(r) for r in te.target])
                pt = np.clip(p[np.arange(len(te)), idx], 1e-3, 1)
                res.append(pd.DataFrame({"kind": kind, "fam": fam, "year": y, "rnd13": te.drafted_at <= 105,
                                         "hit": p.argmax(1) == idx, "ll": -np.log(pt), "wrong": (1 - pt) * te.e1,
                                         "e1": te.e1}))
    R = pd.concat(res)
    print("\nleave-one-draft-out: accuracy of the top room / log loss / share of picks' expected snaps in the wrong room")
    g = R.groupby(["fam", "kind"]).apply(lambda d: pd.Series({"n": len(d), "acc": d.hit.mean(), "logloss": d.ll.mean(),
                                                             "wrong_snaps": d.wrong.sum() / d.e1.sum()}), include_groups=False)
    print(g.round(3).to_string())
    g = R[R.rnd13].groupby(["fam", "kind"]).apply(lambda d: pd.Series({"n": len(d), "acc": d.hit.mean(),
                                                                       "wrong_snaps": d.wrong.sum() / d.e1.sum()}), include_groups=False)
    print("\nrounds 1-3 only:")
    print(g.round(3).to_string())
    # examples on the full fit
    for fam in ["DB", "FRONT"]:
        te = T[T.fam == fam].copy()
        p = fit_predict(T, te, fam, "size")
        te[ROOMS[fam]] = p
        te["top"] = [ROOMS[fam][i] for i in p.argmax(1)]
        x = te[(te.top != te.grp) & (te.drafted_at <= 105)]
        print(f"\n{fam}, R1-3 picks the size model moves from the label room (in-sample): {len(x)}, right {np.mean(x.top == x.target):.2f}")
        print(x[["year", "name", "label", "wt", "ht", "forty", "target"] + ROOMS[fam]].round(2).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
