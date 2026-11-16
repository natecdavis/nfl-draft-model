"""Does depth matter more in high-leverage games? (user's lead, Nov 15 2026)

Idea (from a post on WR3s in the playoffs): with more preparation, defenses take away a team's best player, so
the 2nd / 3rd option matters more against good, well-prepared opponents. Contracts pay stars for regular-season
production, so if this holds, depth players are under-priced for titles and stars over-priced.

Data: carry-forward absences over regular season + playoffs (scripts/opp_strength_absence.py cache).
Role within a team-game's regulars at a group: rank by pre-season price (bridge.preseason_prices; rookie deals
priced at the expected price for their slot), ties by recent share. top = rank 1; depth = rank 2+; WR3 / CB3 =
rank 3+. Robustness: rank by recent snap share instead.

Model: y - (r_home - r_away) = hfa + sum_role b_role D_role + sum_role c_role L_role, with r = full-strength
team-season ratings from the regular season, D = absent home - away, and leverage L:
  opp   : D interacted with opponent rating (home absences x away rating - away absences x home rating)
  po    : D x playoff game
  rematch: D x second+ meeting of the pair this season (more film, no playoff selection)
Pre-specified statistic: c_depth - c_top (spread and margin). The lead holds if it has the predicted sign
(depth costs more, top less, in high-leverage games) on the spread AND the margin, p < 0.05 on at least one,
for WR or pooled non-QB.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import absence_test as at  # noqa: E402
import opp_strength_absence as osa  # noqa: E402

MULTI = ["WR", "TE", "OL", "DL", "LB", "CB", "S"]
RANK_BY = sys.argv[1] if len(sys.argv) > 1 else "price"


def load():
    A = pd.read_parquet(at.OUT / "absence_starters_allgames.parquet")
    pr = at.prices()
    A["price"] = [pr.get((p, y), np.nan) for p, y in zip(A.pfr, A.season)]
    A["price"] = A.price.fillna(0.005)
    key = ["price", "avg_share"] if RANK_BY == "price" else ["avg_share", "price"]
    A = A.sort_values(["game_id", "team", "grp"] + key, ascending=[True, True, True, False, False])
    A["rank"] = A.groupby(["game_id", "team", "grp"]).cumcount() + 1
    A["n_reg"] = A.groupby(["game_id", "team", "grp"]).pfr.transform("size")
    return A


def games(A):
    reg = osa.schedule({"REG"})
    reg = reg[reg.game_id.isin(set(A.game_id))]
    post = osa.schedule(osa.PLAYOFF)
    post = post[post.game_id.isin(set(A.game_id))]
    g = pd.concat([reg.assign(po=0.0), post.assign(po=1.0)], ignore_index=True)
    g = g.sort_values(["season", "gameday" if "gameday" in g else "week"]).reset_index(drop=True)
    pair = g.apply(lambda r: tuple(sorted([r.home_team, r.away_team])), axis=1)
    g["rematch"] = (g.groupby([g.season, pair]).cumcount() > 0).astype(float)
    return g


def main():
    A = load()
    cols = [f"abs_{x}" for x in at.GROUPS]
    for x in at.GROUPS:
        A[f"abs_{x}"] = (A.absent & (A.grp == x)).astype(float)
    g = games(A)
    ts = sorted(set(g[g.po == 0].home_team + g[g.po == 0].season.astype(str))
                | set(g[g.po == 0].away_team + g[g.po == 0].season.astype(str)))
    r = osa.ratings(g[g.po == 0].reset_index(drop=True), A, cols, ts)
    g = g[(g.home_team + g.season.astype(str)).isin(r.index) & (g.away_team + g.season.astype(str)).isin(r.index)]
    g = g.reset_index(drop=True)
    zh = (g.home_team + g.season.astype(str)).map(r).to_numpy()
    za = (g.away_team + g.season.astype(str)).map(r).to_numpy()
    off = zh - za
    hfa = (g.location != "Neutral").astype(float).to_numpy()[:, None]
    print(f"rank by {RANK_BY}; games: regular {int((g.po == 0).sum())}, playoff {int(g.po.sum())}, "
          f"rematches {int(g.rematch.sum())} ({int(((g.rematch == 1) & (g.po == 1)).sum())} in playoffs)")

    ab = A[A.absent]
    print("absent starter-games by role (all games):")
    print(pd.crosstab(ab.grp, ab["rank"].clip(upper=3)).loc[MULTI].to_string())

    QB = (A.absent & (A.grp == "QB")).astype(float)
    other = (A.absent & A.grp.isin(["RB"])).astype(float)  # RB has one regular: control only
    sets = {
        "WR": {"top": (A.grp == "WR") & (A["rank"] == 1), "depth": (A.grp == "WR") & (A["rank"] >= 2)},
        "WR3": {"top": (A.grp == "WR") & (A["rank"] <= 2), "depth": (A.grp == "WR") & (A["rank"] >= 3)},
        "CB": {"top": (A.grp == "CB") & (A["rank"] == 1), "depth": (A.grp == "CB") & (A["rank"] >= 2)},
        "CB3": {"top": (A.grp == "CB") & (A["rank"] <= 2), "depth": (A.grp == "CB") & (A["rank"] >= 3)},
        "non-QB pooled": {"top": A.grp.isin(MULTI) & (A["rank"] == 1), "depth": A.grp.isin(MULTI) & (A["rank"] >= 2)},
    }
    rows = []
    for name, roles in sets.items():
        # every absence is in the model: roles of interest + the rest of each group as controls
        sel = roles["top"] | roles["depth"]
        A["v_top"] = (A.absent & roles["top"]).astype(float)
        A["v_depth"] = (A.absent & roles["depth"]).astype(float)
        A["v_rest"] = (A.absent & ~sel & (A.grp != "QB")).astype(float)
        A["v_qb"] = QB
        vc = ["v_top", "v_depth", "v_rest", "v_qb"]
        H, W = osa.team_counts(A, g, vc)
        D = H - W
        for lev in ["opp", "po", "rematch"]:
            if lev == "opp":
                L = H * za[:, None] - W * zh[:, None]
                Lh = None
            else:
                L = D * g[lev].to_numpy()[:, None]
                Lh = hfa * g[lev].to_numpy()[:, None]
            parts = [D, L[:, :2], hfa] + ([Lh] if Lh is not None else [])
            if lev == "opp":
                parts.append(L[:, 2:])  # leverage controls for rest / QB
            else:
                parts.append(L[:, 2:])
            M = np.hstack(parts)
            for y in ["spread_line", "result"]:
                yy = g[y].to_numpy(float) - off
                m = sm.OLS(yy, M).fit(cov_type="cluster", cov_kwds={"groups": g.home_team.to_numpy()})
                R = np.zeros(M.shape[1]); R[5] = 1; R[4] = -1  # c_depth - c_top
                t = m.t_test(R)
                lo, hi = np.asarray(t.conf_int()).ravel()
                n_lev = int((np.abs(D[:, :2]) * (1 if lev == "opp" else g[lev].to_numpy()[:, None])).sum(axis=0)[1]) \
                    if lev != "opp" else int(np.abs(D[:, 1]).sum())
                rows.append(dict(set=name, leverage=lev, y="spread" if y == "spread_line" else "margin",
                                 b_top=m.params[0], b_depth=m.params[1], c_top=m.params[4], c_depth=m.params[5],
                                 diff=float(np.ravel(t.effect)[0]), lo=lo, hi=hi, p=float(np.ravel(t.pvalue)[0]),
                                 depth_absences=n_lev))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nb = pts per absent starter in ordinary games (negative = hurts the home side); c = extra per +1 pt of "
          "opponent (opp) or in playoff / rematch games. Lead predicts diff = c_depth - c_top < 0 (depth hurts more).")
    print(T.round(3).to_string(index=False))
    T.to_csv(at.OUT / f"depth_leverage_{RANK_BY}.csv", index=False)


if __name__ == "__main__":
    main()
