"""The model's biggest misses, past drafts 2014-2022 (leave-one-draft-out engine, as the review).

Source: data/derived/holdout/defer_candidates.parquet (scripts/defer_test.py): every available player at
every pick each team made, with the model's titles for that team at that moment and what the player
actually delivered at that pick's cost (% of a title over the rookie deal).

  1. Recommendations that flopped / hit: the model's top choice, each player once per draft
     (mean predicted vs mean delivered over the picks where he was the choice).
  2. Players the model underrated / overrated: predicted vs delivered at the pick where he was drafted.
  3. Patterns: QB share of the error, recommendations by how far they later fell, team picks by
     consensus band (bootstrap over draft years), QB recommendations by board distance below the pick.
  4. Engine check (full engine): mean 4-season value of the comparison careers the engine samples vs the
     regression's expectation vs what drafted board players at that rank actually did, and a local-linear
     boundary correction of the comparison mean.

Run:  python scripts/misses.py            (sections 1-3, seconds)
      python scripts/misses.py engine     (section 4, ~1 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
rng = np.random.default_rng(0)


def load():
    C = pd.read_parquet(OUT / "holdout" / "defer_candidates.parquet")
    C = C[C.real.notna()].sort_values(["year", "team", "pick", "titles"], ascending=[True, True, True, False])
    C["mrank"] = C.groupby(["year", "team", "pick"]).cumcount() + 1
    return C


def boot_years(d, col):
    yrs = d.year.unique()
    by = {y: d.loc[d.year == y, col].to_numpy() for y in yrs}
    b = [np.concatenate([by[y] for y in rng.choice(yrs, len(yrs))]).mean() for _ in range(2000)]
    return f"{d[col].mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


def ratio_table(d, by, pred, real):
    g = d.groupby(by, observed=True).agg(n=(pred, "size"), pred=(pred, "sum"), real=(real, "sum"))
    g["delivered/pred"] = g.real / g.pred
    return g.round(2)


def decisions():
    C = load()
    top = C[C.mrank == 1].copy()
    team = C[C.is_team].drop_duplicates(["year", "team", "pick"]).set_index(["year", "team", "pick"])
    top = top.join(team[["real"]].rename(columns={"real": "team_real"}), on=["year", "team", "pick"])
    top["vs_team"] = top.real - top.team_real

    # 1. recommendations, each player once per draft
    A = top.groupby(["year", "name", "grp"]).agg(
        n=("pick", "size"), first_pick=("pick", "min"), rank=("rank", "first"), went=("drafted_at", "first"),
        pred=("titles", "mean"), real=("real", "mean"), vs_team=("vs_team", "sum")).reset_index()
    A["miss"] = A.real - A.pred
    cols = ["year", "name", "grp", "rank", "first_pick", "went", "n", "pred", "real", "miss", "vs_team"]
    print("== 1a. Recommendations that flopped (delivered minus predicted, titles) ==")
    print(A.nsmallest(15, "miss")[cols].round(2).to_string(index=False))
    print("\n== 1b. Recommendations that hit ==")
    print(A.nlargest(10, "miss")[cols].round(2).to_string(index=False))

    # 2. at the pick where each player was drafted
    own = C[C.drafted_at == C.pick].copy()
    own["miss"] = own.real - own.titles
    best = C[C.drafted_at.isna() | (C.pick <= C.drafted_at)].groupby(["year", "pid"]).mrank.min().rename("best_mrank")
    own = own.join(best, on=["year", "pid"])
    cols = ["year", "name", "grp", "rank", "drafted_at", "team", "titles", "real", "miss", "best_mrank"]
    print("\n== 2a. Underrated: delivered minus predicted at his own pick ==")
    print(own.nlargest(20, "miss")[cols].round(2).to_string(index=False))
    print("\n== 2b. Overrated at his own pick ==")
    print(own.nsmallest(12, "miss")[cols].round(2).to_string(index=False))
    print("\n== 2c. Big careers the model never ranked in its top 3 at any pick ==")
    print(own[own.best_mrank > 3].nlargest(15, "real")[cols].round(2).to_string(index=False))

    # 3. patterns
    print("\n== 3a. Share of absolute error ==")
    for nm, d, p in [("recommendations", A, "pred"), ("team picks", own, "titles")]:
        e = (d.real - d[p]).abs()
        q = d.grp == "QB"
        print(f"{nm}: QBs are {q.mean():.0%} of rows, {e[q].sum() / e.sum():.0%} of absolute error; "
              f"20 largest misses = {e.nlargest(20).sum() / e.sum():.0%}")
    A["fell"] = np.select(
        [A.went.isna(), A.went <= A.first_pick, A.went <= A.first_pick + 32],
        ["undrafted", "at/before rec pick", "within 32 after"], "33+ after")
    A["kind"] = np.where(A.grp == "QB", "QB", "non-QB")
    print("\n== 3b. Recommendations by where they actually went ==")
    print(ratio_table(A, ["kind", "fell"], "pred", "real"))

    top["gap"] = pd.cut(top["rank"].fillna(330) - top.pick, [-1e9, 0, 15, 40, 1e9],
                        labels=["rank <= pick", "1-15 below", "16-40 below", "40+ below"])
    q = top[top.grp == "QB"].groupby(["gap", "year", "name"], observed=True).agg(
        pred=("titles", "mean"), real=("real", "mean")).reset_index()
    print("\n== 3c. QB recommendations by how far below the pick they sat on the board (each once per bucket) ==")
    print(ratio_table(q, "gap", "pred", "real"))

    own["band"] = pd.cut(own["rank"].fillna(400), [0, 5, 10, 20, 32, 64, 120, 1e9],
                         labels=["1-5", "6-10", "11-20", "21-32", "33-64", "65-120", "121+/off"])
    print("\n== 3d. Team non-QB picks by consensus rank: delivered minus predicted (95% over draft years) ==")
    nq = own[own.grp != "QB"]
    for b, d in nq.groupby("band", observed=True):
        print(f"{b:>9}: n {len(d):4d}  predicted {d.titles.mean():.2f}  delivered {d.real.mean():.2f}  "
              f"miss {boot_years(d, 'miss')}  below prediction {(d.real < d.titles).mean():.0%}")


def engine():
    import outcomes_v2 as ov
    import prospect_model as pm
    M = ov.model()

    def means(f, r):
        """(comparison mean, local-linear comparison mean) of 4-season value for family f at rank r."""
        lp = np.log(M.expected(f, r).sum() + 1e-3)
        keys, _ = M.comps[f]
        paths, w = M._near(f, r)
        y, x = paths.sum(1), keys - lp
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(2), X.T @ (w * y))
        return (y * w).sum(), max(beta[0], 1e-4)

    df = pm.load()
    df = df[df.drafted & df.cls.between(2014, 2022) & df.hasan_rank.notna()].copy()
    df = df[df.grp.isin(["QB", "WR", "EDGE", "DL", "LB", "OL", "DB", "RB", "TE"])]
    df["actual"] = df[[f"y{t}" for t in pm.SEASONS]].sum(axis=1) / ov.unit_factor("")
    df["regression"] = [M.expected(g, r).sum() for g, r in zip(df.grp, df.hasan_rank)]
    m = [means(g, int(r)) for g, r in zip(df.grp, df.hasan_rank)]
    df["comps"], df["local_linear"] = [a for a, _ in m], [b for _, b in m]
    df["band"] = pd.cut(df.hasan_rank, [0, 10, 32, 64, 120, 330], labels=["1-10", "11-32", "33-64", "65-120", "121+"])
    df["kind"] = np.where(df.grp == "QB", "QB", "non-QB")

    def show(g):
        g = g.agg(n=("actual", "size"), actual=("actual", "mean"), regression=("regression", "mean"),
                  comps=("comps", "mean"), local_linear=("local_linear", "mean"))
        for c in ["regression", "comps", "local_linear"]:
            g[f"{c}/actual"] = g[c] / g.actual
        return g.round(3).to_string()

    print("== 4a. Drafted board players 2014-2022: 4-season value (cap share) vs the engine ==")
    print(show(df.groupby(["kind", "band"], observed=True)))
    print("\n== 4b. Consensus top 10 by family ==")
    print(show(df[df.hasan_rank <= 10].groupby("grp")))


if __name__ == "__main__":
    engine() if sys.argv[1:] == ["engine"] else decisions()
