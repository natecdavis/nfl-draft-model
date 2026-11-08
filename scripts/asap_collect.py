"""Collect ASAP Sports football transcripts for board / drafted players (see src/transcripts.py).

Writes data/derived/asap_index.parquet (every football interview link 2011-2024) and
data/derived/asap_answers.parquet (one row per answer by a matched player).
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
import prospect_model as pm
import transcripts as tr
from outlook import OUT
from public_board import _norm

YEARS = range(2011, 2025)  # 2023-24 added for the fresh replication (classes 2023-25)


def main():
    ix = pd.DataFrame(tr.index(YEARS))
    ix.to_parquet(OUT / "asap_index.parquet")
    print(f"interview links {len(ix)}, events {ix.event.nunique()}", flush=True)
    uni = pm.load()
    u = uni[(uni.on_board == 1) | uni.drafted.astype(bool)][["player_id", "player_name", "cls"]].copy()
    u["k"] = u.player_name.map(_norm)
    ix["k"] = ix.who.map(_norm)
    m = ix.merge(u, on="k")
    m = m[(m.cls - m.year).between(0, 5)]
    amb = m.groupby(["id", "k"]).player_id.nunique()
    m = m.join(amb.rename("n_cand"), on=["id", "k"])
    m = m[m.n_cand == 1]
    print(f"matched interviews {m.id.nunique()} for {m.player_id.nunique()} players", flush=True)
    rows = []
    for n, r in enumerate(m.itertuples()):
        t = tr.turns(tr.interview(r.id))
        last = [w for w in r.player_name.upper().replace(".", "").split() if w not in {"JR", "SR", "II", "III", "IV", "V"}][-1]
        q = None
        for spk, txt in t:
            if spk == "Q":
                q = txt
            elif last in spk and "COACH" not in spk:
                rows.append({"player_id": r.player_id, "cls": r.cls, "id": r.id, "event": r.event, "year": r.year,
                             "date": r.date, "q": q, "a": txt})
        if n % 100 == 0:
            print(f"  {n}/{len(m)} interviews, {len(rows)} answers", flush=True)
    a = pd.DataFrame(rows)
    a.to_parquet(OUT / "asap_answers.parquet")
    print(f"answers {len(a)} from {a.player_id.nunique()} players", flush=True)


if __name__ == "__main__":
    main()
