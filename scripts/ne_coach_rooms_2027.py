import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from scipy.stats import norm
import draftday as dd
from coach_pairing_2026 import expected_by_pick, CAP

b = pd.read_csv("data/derived/board_2027.csv")
ev = expected_by_pick()
picks_all = np.arange(1, 263)
# NE round-1 slot: median 18, 10-90% 12-24 (season simulation); later rounds = slot + 32k
slots = np.arange(8, 30); w = norm.pdf(slots, 18, 4.7); w /= w.sum()

def p_at(rank, infl):
    mu = np.log(rank + 10) + dd.drift(rank); sd = dd.spread(rank) * infl
    e = np.log(np.append(picks_all - 0.5, picks_all[-1] + 0.5) + 10)
    return np.diff(norm.cdf(e, mu, sd))

def ne_value(rank, infl, edge12=0.15, edge3=0.05):
    p = p_at(rank, infl); tot = 0; pt = 0
    for s, ws in zip(slots, w):
        for k in range(4):  # rounds 1-4 (own)
            pk = s + 32 * k
            e = edge12 if k < 2 else edge3
            tot += ws * p[pk - 1] * e * ev[pk] * CAP; pt += ws * p[pk - 1]
    return tot, pt

for infl in (1.0, 1.5):
    rows = []
    for _, r in b.iterrows():
        v, pt = ne_value(r["rank"], infl)
        vh, _ = ne_value(r["rank"], infl, 0.27, 0.27)
        rows.append((r.school, r.pos, r["name"], r["rank"], pt, v, vh))
    d = pd.DataFrame(rows, columns=["school", "pos", "name", "rank", "p_ne", "val", "val_hi"])
    room = d.groupby(["school", "pos"]).agg(players=("name", lambda s: ", ".join(s)), ranks=("rank", lambda s: ", ".join(map(str, s))),
                                            p_ne=("p_ne", "sum"), val=("val", "sum"), val_hi=("val_hi", "sum")).reset_index()
    print(f"\n=== spread x{infl}: TOP ROOMS for NE ($M, edge x1.15 R1-2 / x1.05 R3-4; 'hi' = role-matched recent x1.27)")
    print(room.sort_values("val", ascending=False).head(15).round(3).to_string(index=False))
    wr = room[(room.pos == "WR") & room.school.isin(["Indiana", "Texas", "Ohio State", "Arizona State"])]
    print("  WR rooms asked about:"); print(wr.round(3).to_string(index=False))
