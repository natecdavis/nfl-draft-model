"""Should the model's non-QB picks lean on the consensus board? Past drafts 2014-2022 (leave-one-draft-out
engine, as the review), every pick each team really made.

At each pick, every available player has the model's value (simulator.options: same tables and state as
the review, the team's own earlier picks setting its needs) and what he actually delivered at that pick's
cost for that team (review.RealizedV2 via the simulator's realized cache, % of a title over the rookie
deal). Rules for which player to take:

  model      the model's top choice (the review's recommendation)
  board      the best-ranked available player on the consensus board
  margin m   model's choice if it's a QB or beats the board's best by >= m (% of a title); else the board's
  window k   model's top among the k best-ranked available on the board (its QB call kept if its overall top is a QB)

Scored as delivered minus the team's actual pick (titles), plus seasons started and AV of the chosen
player, by round, as every other scorecard: only picks where the rule's choice differs from the team's
player, each chosen player once per draft (averaged over the picks where he was the choice). (Until Nov 2026 every pick counted, so
one player could be "taken" at many picks: the board rule took La'el Collins at 100 picks in 2015;
per-pick numbers are still printed.) m and k are chosen leave-one-draft-out (best on the other 8
drafts, same once-per-draft score) so the tuned rules aren't scored on the drafts they were tuned on.

Run:  python scripts/defer_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402
from positions import weights  # noqa: E402
from simulator import get_table, options, pool, situation  # noqa: E402

YEARS = range(2014, 2023)
MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
WINDOWS = [1, 2, 3, 5, 10, 20, 40]
CACHE = OUT / "holdout" / "defer_candidates.parquet"


def candidates() -> pd.DataFrame:
    """One row per (pick, available player): model titles, board rank, realized titles, starts, AV."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    from export_simulator_artifact import realized_arrays
    S = pd.read_parquet(OUT / "review_states.parquet")
    R = pd.read_parquet(OUT / "review_picks.parquet")
    real = realized_arrays(S, R)
    from review import Careers
    car = Careers()  # starts / AV, undrafted players included
    out = []
    for y in YEARS:
        P, W = pool(y), weights(y)
        for _, s in S[S.year == y].iterrows():
            mine = R[(R.year == y) & (R.team == s.team)].sort_values("pick")
            if mine.empty or f"{y}_{s.team}" not in real:
                continue
            arr, table = real[f"{y}_{s.team}"], get_table(situation(s), year=y)
            counts = {}
            for j, (_, p) in enumerate(mine.iterrows()):
                k = int(p.pick)
                o = options(P, table, k, set(), counts, float(s.p_need), W)
                o = o.assign(year=y, team=s.team, pick=k, rnd=int(p["round"]), team_player=p.player,
                             real=arr[j, o.pid.to_numpy()],
                             gs=[car.get("gs", y, a, n) for a, n in zip(o.drafted_at, o["name"])],
                             av=[car.get("av", y, a, n) for a, n in zip(o.drafted_at, o["name"])])
                own = P[P.drafted_at == k]
                o["is_team"] = o.pid.isin(own.pid)
                out.append(o[["year", "team", "pick", "rnd", "pid", "name", "grp", "rank", "drafted_at", "titles",
                              "real", "gs", "av", "is_team", "team_player"]])
                g = p.grp
                counts[g] = counts.get(g, 0) + 1
        print(f"{y} done", flush=True)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(CACHE)
    return df


def choose(g: pd.DataFrame, rule: str, x=None) -> int:
    """Row label of the chosen player at one pick (g sorted by model titles, best first)."""
    top = g.index[0]
    if rule == "model":
        return top
    r = g["rank"].fillna(1e9)
    board = r.idxmin()
    if rule == "board":
        return board
    if g.at[top, "grp"] == "QB":
        return top
    if rule == "margin":
        return top if g.at[top, "titles"] - g.at[board, "titles"] >= x else board
    if rule == "window":
        ok = r.rank(method="first") <= x
        return g[ok].index[0]
    raise ValueError(rule)


def main():
    C = candidates()
    C = C[C.real.notna()].sort_values(["year", "team", "pick", "titles"], ascending=[True, True, True, False])
    groups = {k: g for k, g in C.groupby(["year", "team", "pick"], sort=False)}
    team = C[C.is_team].set_index(["year", "team", "pick"])[["real", "gs", "av"]]
    team = team[~team.index.duplicated()]
    keys = [k for k in groups if k in team.index]
    rnd = pd.Series({k: groups[k].rnd.iat[0] for k in keys})
    R = pd.read_parquet(OUT / "review_picks.parquet").set_index(["year", "pick"]).rec_player
    R = R[~R.index.duplicated()]
    print(f"{len(keys)} picks with the team's player found in the pool "
          f"(model top = review rec: {np.mean([groups[k]['name'].iat[0] == R.get((k[0], k[2])) for k in keys]):.0%})")

    def once(d):
        """Picks where the choice differs from the team's player; each chosen player once per draft, averaged over
        the picks where he was the choice. (Not his first such pick: that anchors him to one team's player and
        situation, and a rule that moves him to a later pick changes his score without changing the choice:
        Watson at #1 vs Garrett +13 or at #11 vs Lattimore +32.)"""
        d = d[d["name"].to_numpy() != d.team_player.to_numpy()]
        d = d.reset_index(drop=True) if "year" in d.columns else d.reset_index()
        num = [c for c in ["d_real", "d_gs", "d_av"] if c in d.columns]
        keep = [c for c in ["grp", "gap", "rnd", "band"] if c in d.columns]
        return d.groupby(["year", "name"], as_index=False).agg({**{c: "mean" for c in num}, **{c: "first" for c in keep}})

    def picks(rule, x=None):
        rows = [groups[k].loc[choose(groups[k], rule, x)] for k in keys]
        d = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["year", "team", "pick"]))
        d["d_real"] = d.real.to_numpy() - team.loc[keys].real.to_numpy()
        d["d_gs"] = d.gs.to_numpy() - team.loc[keys].gs.to_numpy()
        d["d_av"] = d.av.to_numpy() - team.loc[keys].av.to_numpy()
        d["rnd"] = rnd.to_numpy()
        return d

    def tuned(rule, grid, phase):
        """Leave-one-draft-out choice of the rule's setting for one round band."""
        runs = {x: picks(rule, x) for x in grid}
        out, chosen = [], {}
        for y in YEARS:
            score = {x: once(d[(d.index.get_level_values("year") != y) & phase(d.rnd)]).d_real.mean() for x, d in runs.items()}
            b = max(score, key=score.get)
            chosen[y] = b
            d = runs[b]
            out.append(d[d.index.get_level_values("year") == y])
        return pd.concat(out), chosen, runs

    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"

    bands = [("R1", lambda r: r == 1), ("R2-4", lambda r: (r >= 2) & (r <= 4)), ("R5-7", lambda r: r >= 5)]
    base = {"model": picks("model"), "board": picks("board")}
    rows = []
    for bl, ph in bands:
        res = dict(base)
        res["margin (tuned)"], cm, mr = tuned("margin", MARGINS, ph)
        res["window (tuned)"], cw, wr = tuned("window", WINDOWS, ph)
        print(f"\n=== {bl}: chosen minus team's pick where they differ, each chosen player once per draft ===  (tuned margin by year {cm}; window {cw})")
        for name, d in res.items():
            x = d[ph(d.rnd)]
            o = once(x)
            same = (x["name"].to_numpy() == base["model"].loc[x.index, "name"].to_numpy()).mean()
            print(f"  {name:15s} titles {boot(o.d_real)} n={len(o)} | starts {o.d_gs.mean():+.2f} | AV {o.d_av.mean():+.1f} "
                  f"| per pick {x.d_real.mean():+.2f} | same as model {same:.0%} | QB {(o.grp == 'QB').mean():.0%} "
                  f"| WR {(o.grp == 'WR').mean():.0%}")
            rows.append({"band": bl, "rule": name, "titles": o.d_real.mean(), "starts": o.d_gs.mean(), "av": o.d_av.mean(),
                         "n": len(o), "titles_per_pick": x.d_real.mean()})
        print("  every setting, all 9 drafts, each player once (titles | starts):")
        for lab, rr in [("margin", mr), ("window", wr)]:
            print("   " + lab + ": " + ", ".join(f"{x}: {once(d[ph(d.rnd)]).d_real.mean():+.2f} | {once(d[ph(d.rnd)]).d_gs.mean():+.2f}"
                                               for x, d in rr.items()))
    pd.DataFrame(rows).to_parquet(OUT / "holdout" / "defer_test.parquet")


if __name__ == "__main__":
    main()
