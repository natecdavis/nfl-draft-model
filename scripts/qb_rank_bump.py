"""QB value by consensus rank at a fixed pick, all classes vs without 2020 (README "Titles non-increasing in rank").
Run: python scripts/qb_rank_bump.py"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import outcomes_v2 as o2
from draftday import DraftEngine
from trade import Context
import review
pd.set_option("display.width", 250)
R = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 32, 40, 50]
ctx = Context()
sit = {"start": -5.0, "shift": {}, "qb_from": 1}
rows = {}
for name, ho in [("all", None), ("no2020", ("lodo", 2020))]:
    o2.HOLDOUT = ho; o2._MODEL = None
    m = o2.model()
    eng = DraftEngine(sit, ctx=ctx, fall_weight=review.FALL_W)
    for r in R:
        paths, w = m._near("QB", r)
        s = paths.sum(1) * m.scale("QB", r)
        mean = (s * w).sum(); sd = np.sqrt((w * (s - mean) ** 2).sum())
        big = (w * (s > 0.40)).sum()   # 4-season value above 40% of cap
        zero = (w * (s < 0.02)).sum()
        rows[(name, r)] = dict(mean4=mean, sd4=sd, p_big=big, p_zero=zero, exp_reg=m.expected("QB", r).sum(),
                               scale=m.scale("QB", r), n_eff=1 / (w ** 2).sum(),
                               titles_at5=100 * eng._player_value("QB", r, 5),
                               titles_at_rank=100 * eng._player_value("QB", r, r))
    if ho is None:
        df = o2.apply_holdout(o2.pm.load(), None)
        q = df[(df.grp == "QB") & df.hasan_rank.notna() & (df.hasan_rank <= 40) & (df.cls >= 2014)].copy()
        ycols = [c for c in q.columns if c in ("y1", "y2", "y3", "y4")]
        q["sum4"] = q[ycols].sum(axis=1, min_count=1)
        print(list(q.columns)); namecol = next(c for c in q.columns if "name" in c.lower() or "player" in c.lower())
        print(q.sort_values("hasan_rank")[[namecol, "cls", "hasan_rank", "sum4"]].round(3).to_string(index=False))
T = pd.DataFrame(rows).T
for name in ["all", "no2020"]:
    print("\n==", name); print(T.loc[name].round(3).to_string())
