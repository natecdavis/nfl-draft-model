"""Board-less draft classes (outcomes_v2.NO_BOARD_CLASSES: "keep" before Nov 2026, "drop" adopted) and monotone
value in rank (outcomes_v2.MONOTONE: "off" before, "iso" tested, "min" adopted). README section of the same name.

Background (scripts/misses.py, scripts/comp_correct.py engine): drafted players who weren't on the consensus
board were given careers worth ~1.8x what they delivered (QBs ~10x). Cause: Hasan's board starts with the
2014 class, so every drafted player in 2012-2013 reads as off-board, first-rounders included. They were
~2/3 of the off-board comparison pool (off-board QBs drew Luck, Wilson, Cousins). "drop" leaves classes
where fewer than half the drafted players are on the board out of the regression and the comparison pool.

  engine   leave-one-draft-out, 2014-2022: drafted players' actual 4-season value vs the engine's mean
           (comparison careers x local-linear correction x monotone factor), keep / drop / drop+iso / drop+min,
           by consensus band and, for off-board players, by family
  review   leave-one-draft-out past-drafts review, before (holdout/review_picks_lodo_ccll.parquet) vs one
           variant (holdout/review_picks_lodo_nbdrop[_monoiso|_monomin].parquet, from NO_BOARD_CLASSES=drop
           [MONOTONE=iso|min] scripts/holdout_review.py), with scripts/comp_correct.py's scorecard ("off" =
           before, "ll" = the variant)

Run:  python scripts/no_board.py engine
      python scripts/no_board.py review [nbdrop|nbdrop_monoiso|nbdrop_monomin]   (default nbdrop_monomin)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402

pd.set_option("display.width", 250)
HD = OUT / "holdout"


def engine():
    import outcomes_v2 as ov
    import prospect_model as pm
    full, out = pm.load(), []
    for nb in ["keep", "drop", "drop+iso", "drop+min"]:
        ov.NO_BOARD_CLASSES, ov.MONOTONE = (nb.split("+") + ["off"])[:2]
        for y in range(2014, 2023):
            ov.HOLDOUT, ov._MODEL = ("lodo", y), None
            M = ov.model()
            d = full[full.drafted & (full.cls == y)].copy()
            d = d[d.grp.isin(list(M.comps))]
            d["r"] = d.hasan_rank.fillna(pm.OFF_BOARD).clip(1, pm.OFF_BOARD).astype(int)
            d["actual"] = d[[f"y{t}" for t in pm.SEASONS]].sum(axis=1) / ov.unit_factor("")
            d["pred"] = [(lambda pw: float((pw[0].sum(1) * pw[1]).sum()))(M._near(g, r)) * M.scale(g, r)
                         for g, r in zip(d.grp, d.r)]
            out.append(d[["cls", "grp", "hasan_rank", "pick", "actual", "pred"]].assign(nb=nb))
        print(nb, "done", flush=True)
    ov.HOLDOUT, ov._MODEL, ov.NO_BOARD_CLASSES, ov.MONOTONE = None, None, "drop", "min"
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(HD / "no_board_engine.parquet")
    D["band"] = pd.cut(D.hasan_rank.fillna(999), [0, 10, 32, 64, 120, 200, 250, 300, 1e9],
                       labels=["1-10", "11-32", "33-64", "65-120", "121-200", "201-250", "251-300", "off board"])
    D["kind"] = np.where(D.grp == "QB", "QB", "non-QB")
    p = D.pivot_table(index=["kind", "band"], columns="nb", values=["actual", "pred"], aggfunc="mean", observed=True)
    t = pd.DataFrame({"n": D[D.nb == "drop"].groupby(["kind", "band"], observed=True).size(),
                      "actual": p[("actual", "drop")], "keep/actual": p[("pred", "keep")] / p[("actual", "drop")],
                      "drop/actual": p[("pred", "drop")] / p[("actual", "drop")],
                      "drop+iso/actual": p[("pred", "drop+iso")] / p[("actual", "drop")],
                      "drop+min/actual": p[("pred", "drop+min")] / p[("actual", "drop")]})
    print("\n== Leave-one-draft-out, drafted players 2014-2022: engine mean / actual 4-season value ==")
    print(t.round(3).to_string())
    o = D[D.hasan_rank.isna()]
    p = o.pivot_table(index="grp", columns="nb", values=["actual", "pred"], aggfunc="mean")
    print("\n== Off-board drafted players by family: engine mean / actual ==")
    print(pd.DataFrame({"n": o[o.nb == "drop"].groupby("grp").size(), "keep": p[("pred", "keep")] / p[("actual", "drop")],
                        "drop": p[("pred", "drop")] / p[("actual", "drop")],
                        "drop+iso": p[("pred", "drop+iso")] / p[("actual", "drop")],
                        "drop+min": p[("pred", "drop+min")] / p[("actual", "drop")]}).round(2).to_string())
    for nb in ["keep", "drop", "drop+iso", "drop+min"]:
        b = D[(D.nb == nb) & D.hasan_rank.notna()]
        e = b.pred - b.actual
        print(f"board players, {nb}: mean error {e.mean():+.4f}, mean abs error {e.abs().mean():.4f}, "
              f"rmse {np.sqrt((e ** 2).mean()):.4f}")


def review(variant="nbdrop_monomin"):
    import comp_correct
    b = f"review_picks_lodo_{variant}.parquet"
    comp_correct.review("review_picks_lodo_ccll.parquet", b, f"no_board_review_{variant}")
    # off-board players: how often recommended
    for f in ["review_picks_lodo_ccll.parquet", b]:
        d = pd.read_parquet(HD / f)
        d = d[d.year <= 2022]
        print(f"{f}: off-board recommendations {d.rec_rank.isna().mean():.1%} of picks "
              f"(rounds 5-7 {d[d['round'] >= 5].rec_rank.isna().mean():.1%})")


if __name__ == "__main__":
    {"engine": engine, "review": review}[sys.argv[1]](*sys.argv[2:])
