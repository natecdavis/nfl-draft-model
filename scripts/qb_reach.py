"""QB reach penalty: should the model discount a QB who sits well below the current pick on the board?

Background (scripts/misses.py): the model's QB recommendations delivered about what was predicted when the QB
was ranked at or above the pick, but 0.65x when 16-40 spots below and 0.24x when 40+ below (no hits). The
engine's QB curve is roughly calibrated over all drafted QBs (scripts/no_board.py; ranks 33-120 1.18x), so
this looks like selection: the model reaches exactly when a QB's value looks high relative to his board
slot, and those are the overestimates.

Rule: a QB ranked below the pick has his value (titles, when positive) multiplied by
((pick + 10) / (rank + 10)) ** lam (off-board = rank 330), so a reach is discounted by how far, in log
terms, he sits below the pick (20 spots below at #5 counts far more than at #150). lam = 0 is the model.

Past drafts 2014-2022, every pick each team made, every available player valued as in the review
(leave-one-draft-out engine and tables: scripts/defer_test.py's candidates, rebuilt with the current
engine and yardsticks; the same file scripts/defer_test.py uses). Scored as delivered minus the team's actual
pick (titles), plus starts and AV, by round, as every other scorecard: only picks where the choice differs from
the team's player, each chosen player once per draft, averaged over the picks where he was the choice (until Nov
2026 every pick counted, so e.g. Hurts counted at several picks; per-pick numbers still printed). lam is chosen leave-one-draft-out (best on
the other 8 drafts, all rounds together, same score) so the tuned rule isn't scored on the drafts it was tuned
on. Also: delivered / predicted for all available QBs and for the model's QB choices (each once per draft), by
board distance below the pick (miscalibration vs selection).

Run:  python scripts/qb_reach.py          (builds the candidates if missing, ~10 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import defer_test  # noqa: E402
from outlook import OUT  # noqa: E402

pd.set_option("display.width", 250)
YEARS = range(2014, 2023)
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
defer_test.CACHE = OUT / "holdout" / "defer_candidates.parquet"
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


def once(d: pd.DataFrame) -> pd.DataFrame:
    """Picks where the choice differs from the team's player; each chosen player once per draft, averaged over
    the picks where he was the choice. (Not his first such pick: that anchors him to one team's player and
    situation, and a rule that moves him to a later pick changes his score without changing the choice:
    Watson at #1 vs Garrett +13 or at #11 vs Lattimore +32.)"""
    d = d[d["name"].to_numpy() != d.team_player.to_numpy()]
    d = d.reset_index(drop=True) if "year" in d.columns else d.reset_index()
    num = [c for c in ["d_real", "d_gs", "d_av"] if c in d.columns]
    keep = [c for c in ["grp", "gap", "rnd", "band"] if c in d.columns]
    return d.groupby(["year", "name"], as_index=False).agg({**{c: "mean" for c in num}, **{c: "first" for c in keep}})


def penalized(C: pd.DataFrame, lam: float) -> np.ndarray:
    r = C["rank"].fillna(330).to_numpy()
    f = np.where((C.grp == "QB").to_numpy() & (r > C.pick.to_numpy()),
                 ((C.pick.to_numpy() + 10) / (r + 10)) ** lam, 1.0)
    t = C.titles.to_numpy()
    return np.where(t > 0, t * f, t)


def main():
    C = defer_test.candidates()
    C = C[C.real.notna()].reset_index(drop=True)
    team = C[C.is_team].drop_duplicates(["year", "team", "pick"]).set_index(["year", "team", "pick"])
    C = C[C.set_index(["year", "team", "pick"]).index.isin(team.index)].reset_index(drop=True)
    C["gap"] = pd.cut(C["rank"].fillna(330) - C.pick, [-1e9, 0, 15, 40, 1e9],
                      labels=["rank <= pick", "1-15 below", "16-40 below", "40+ below"])
    C["band"] = np.select([C.rnd == 1, C.rnd <= 4], ["R1", "R2-4"], "R5-7")

    # calibration: all available QBs vs the model's QB choices, by board distance below the pick
    top = C.loc[C.groupby(["year", "team", "pick"]).titles.idxmax()].sort_values(["year", "pick"])
    top = top.drop_duplicates(["year", "name"])  # each choice once per draft
    print("== QBs: delivered / predicted (titles, positive predictions), by board distance below the pick "
          "(choices: each once per draft, at his first pick) ==")
    for lab, d in [("all available QBs", C), ("model's QB choices", top)]:
        q = d[(d.grp == "QB") & (d.titles > 0)]
        g = q.groupby("gap", observed=True).agg(n=("real", "size"), pred=("titles", "sum"), real=("real", "sum"))
        g["delivered/pred"] = g.real / g.pred
        print(f"-- {lab}\n{g.round(2).to_string()}")

    keys = ["year", "team", "pick"]
    runs = {}
    for lam in LAMS:
        C["adj"] = penalized(C, lam)
        ch = C.loc[C.groupby(keys).adj.idxmax()].set_index(keys)
        t = team.loc[ch.index]
        runs[lam] = pd.DataFrame({"d_real": ch.real - t.real, "d_gs": ch.gs - t.gs, "d_av": ch.av - t.av,
                                  "band": ch.band, "grp": ch.grp, "name": ch["name"], "gap": ch.gap,
                                  "team_player": ch.team_player})
    base = runs[0.0]
    # leave-one-draft-out choice of lam (all rounds together)
    out, chosen = [], {}
    for y in YEARS:
        score = {lam: once(d[d.index.get_level_values("year") != y]).d_real.mean() for lam, d in runs.items()}
        chosen[y] = max(score, key=score.get)
        d = runs[chosen[y]]
        out.append(d[d.index.get_level_values("year") == y])
    tuned = pd.concat(out)
    print(f"\nlam chosen leave-one-draft-out, by held-out year: {chosen}")
    print("\n== Chosen minus team's pick where they differ, each chosen player once per draft ==")
    for b in ["R1", "R2-4", "R5-7"]:
        print(f"-- {b}")
        for name, d in [("model (lam 0)", base), ("tuned", tuned)] + [(f"lam {lam}", runs[lam]) for lam in LAMS[1:]]:
            x = d[d.band == b]
            o = once(x)
            same = (x["name"] == base.loc[x.index, "name"]).mean()
            print(f"  {name:13s} titles {boot(o.d_real)} n={len(o)} | starts {o.d_gs.mean():+.2f} | AV {o.d_av.mean():+.1f} "
                  f"| per pick {x.d_real.mean():+.2f} | QB {(o.grp == 'QB').mean():.0%} | QB reaches 16+ below "
                  f"{((o.grp == 'QB') & o.gap.isin(['16-40 below', '40+ below'])).mean():.1%} | same as model {same:.0%}")
    # paired: tuned minus model at picks where the choice changed
    ch = tuned.join(base, rsuffix="_m")
    ch = ch[ch["name"] != ch.name_m].reset_index().sort_values(["year", "pick"]).drop_duplicates(["year", "name_m", "name"])
    print(f"\nchanged choices (tuned vs model), each swap once per draft: {len(ch)}; titles tuned minus model "
          f"{boot(ch.d_real - ch.d_real_m) if len(ch) else 'n/a'}; starts {(ch.d_gs - ch.d_gs_m).mean():+.2f}; "
          f"AV {(ch.d_av - ch.d_av_m).mean():+.1f}")
    print(ch.assign(diff=ch.d_real - ch.d_real_m).sort_values("diff")[["year", "pick", "name_m", "name", "diff"]].round(2)
          .pipe(lambda d: d if len(d) <= 16 else pd.concat([d.head(8), d.tail(8)])).to_string())
    pd.concat({lam: d for lam, d in runs.items()}).to_parquet(OUT / "holdout" / "qb_reach.parquet")


if __name__ == "__main__":
    main()
