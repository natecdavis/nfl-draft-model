"""Need-aware choices vs teams' picks, drafts 2014-2022 (realized value known).

At every pick, for the team on the clock, the model picks the available board player with the
most title value under four need rules (same board, team states, candidates and realized-value
scoring as the Past drafts review):

  current   the model's rule today: each earlier pick at the group this draft (team's actual
            picks) cuts years 1-2 playing time to x0.6, then x0.3; roster ignored
  within    measured within-draft effect only: playing time in all seasons x(1 + b_w[k]) for k
            earlier rookies at the group
  roster    measured roster effect only: x(1 + b_r * open spots at the group), open spots on
            draft day from scripts/roster_need.py (clipped to +-1.5)
  both      within x roster

b_r, b_w are measured on rookies' 4-season value (scripts/roster_need.py regression, controls for
position x pick, position x consensus rank, team strength, year), leaving out the draft being
scored. QB values use the current rule (QB need from the QB rules; filled after one QB).

Value tables are rebuilt with the fall shading applied once (review.Table applies it twice:
see README), on a grid of playing-time multipliers; cached in data/derived/need_tables/.

Scoring (same as the review): realized titles for this team, from each player's actual first
four seasons at the cost of this pick (review.RealizedV2). Model's player minus team's player,
where they disagree, each recommended player counted once per draft. 'Adjusted' rescales a
recommended player's realized value by the roster effect of this team's open spots vs the open
spots of the team that actually drafted him (his real career happened in that team's depth chart).

Run:  python scripts/need_choice.py [workers=6]
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OUTCOME_UNITS", "family")  # its tables (need_tables) were built in family units

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from draftday import DraftEngine  # noqa: E402
from outlook import GROUPS, OUT  # noqa: E402
from review import K_GRID, OFF_BOARD_RANK, T_GRID, board, draft_night  # noqa: E402

YEARS = range(2014, 2023)
import os  # noqa: E402
TDIR = OUT / os.environ.get("NEED_TABLE_DIR", "need_tables")
# (early multiplier on seasons 1-2, multiplier on all seasons)
LEVELS = [(1.0, 1.0), (0.6, 1.0), (0.3, 1.0)] + [(1.0, m) for m in (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)]
if os.environ.get("NEED_LEVELS") == "current":  # only the current rule's levels (faster)
    LEVELS = LEVELS[:3]
ALL_M = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
ALL_IDX = [3, 4, 5, 0, 6, 7, 8]  # LEVELS index for each ALL_M value
FW = 0.85


def _key(sit):
    sh = "_".join(f"{k}:{v:g}" for k, v in sorted(sit["shift"].items())) or "none"
    return f"start{sit['start']:g}_qb{sit['qb_from']}_shift{sh}"


def build_table(sit):
    f = TDIR / f"{_key(sit)}.npz"
    if f.exists():
        return
    from pick_value import pick_effect
    from trade import Context
    eng = DraftEngine(sit, ctx=Context(), fall_weight=FW)
    out = {}
    for g in GROUPS:
        if g == "QB":  # [not filled, filled]
            arr = np.array([[[eng.titles(eng.player_points(g, t, k, qb_filled=f_)) for k in K_GRID] for t in T_GRID]
                            for f_ in (False, True)])
        else:
            arr = np.zeros((len(LEVELS), len(T_GRID), len(K_GRID)))
            for a, t in enumerate(T_GRID):
                sh0, price = eng._outcomes(g, int(t))
                for b, k in enumerate(K_GRID):
                    for li, (me, ma) in enumerate(LEVELS):
                        sh = sh0 * ma
                        sh[:, :2] *= me
                        pts = pick_effect(sh, price, int(k), g, {**sit, "qb_from": eng._qb_from()},
                                          eng.ctx.cost, eng.ctx.bar, eng.ctx.rates)
                        arr[li, a, b] = eng.titles(pts)
        out[g] = arr
    TDIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(f)


class NTable:
    def __init__(self, sit):
        z = np.load(TDIR / f"{_key(sit)}.npz")
        self.v = {g: z[g] for g in z.files}
        self.x, self.y = np.log(T_GRID + 10), np.log(K_GRID + 10)

    def _bilin(self, A, rank, pick):
        t = rank if pick <= rank else np.exp(np.log(rank + 10) + FW * (np.log(pick + 10) - np.log(rank + 10))) - 10
        lt, lk = np.log(t + 10), np.log(pick + 10)
        x, y = self.x, self.y
        i = int(np.clip(np.searchsorted(x, lt) - 1, 0, len(x) - 2)); j = int(np.clip(np.searchsorted(y, lk) - 1, 0, len(y) - 2))
        fx = np.clip((lt - x[i]) / (x[i + 1] - x[i]), 0, 1); fy = np.clip((lk - y[j]) / (y[j + 1] - y[j]), 0, 1)
        return (1 - fx) * (1 - fy) * A[..., i, j] + fx * (1 - fy) * A[..., i + 1, j] + (1 - fx) * fy * A[..., i, j + 1] + fx * fy * A[..., i + 1, j + 1]

    def current(self, g, n_earlier, rank, pick):
        if g == "QB":
            return float(self._bilin(self.v[g][int(n_earlier > 0)], rank, pick))
        return float(self._bilin(self.v[g][min(n_earlier, 2)], rank, pick))

    def scaled(self, g, n_earlier, rank, pick, m):
        """All-season playing time x m (linear between grid multipliers)."""
        if g == "QB":
            return self.current(g, n_earlier, rank, pick)
        vals = self._bilin(self.v[g][ALL_IDX], rank, pick)
        return float(np.interp(m, ALL_M, vals))


def coefficients(R, leave_out):
    d = R[(R.year != leave_out) & R.v14.notna() & R.strength_prior.notna()]
    m = smf.ols("v14 ~ open_c + C(earlier_c) + strength_prior + C(year) + C(grp) + C(grp):lp + C(grp):lr + off", d).fit()
    mu = d.v14.mean()
    return m.params["open_c"] / mu, {0: 0.0, 1: m.params["C(earlier_c)[T.1]"] / mu, 2: m.params["C(earlier_c)[T.2]"] / mu}


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    from roster_need import load as rn_load, roster_need
    from simulator import situation

    S = pd.read_parquet(OUT / "review_states.parquet")
    S = S[S.year.isin(YEARS)]
    sits = {_key(situation(r)): situation(r) for _, r in S.iterrows()}
    todo = [s for k, s in sits.items() if not (TDIR / f"{k}.npz").exists()]
    t0 = time.time()
    print(f"{len(sits)} situations, building {len(todo)} tables", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_table, todo))
    print(f"tables ready, {time.time() - t0:.0f}s", flush=True)

    # roster need for every team / group / year, and rookie data for the coefficients
    _, ps, _, _ = rn_load()
    need = pd.concat([roster_need(ps, y) for y in YEARS], ignore_index=True)
    typ = need.groupby("grp").total_prev.median()  # same years as the test; roster_need.py used 2014-25
    need["open_c"] = (need.grp.map(typ) - need.returning - need.added).clip(-1.5, 1.5)
    open_at = {(y, t, g): o for y, t, g, o in zip(need.year, need.team, need.grp, need.open_c)}
    R = pd.read_parquet(OUT / "roster_need_rookies.parquet").merge(
        pd.read_parquet(OUT / "review_states.parquet")[["year", "team", "strength_prior"]], on=["year", "team"], how="left")
    R = R.sort_values(["year", "team", "pick"])
    R["earlier_c"] = R.groupby(["year", "team", "grp"]).cumcount().clip(upper=2)
    coef = {y: coefficients(R, y) for y in YEARS}
    print("Leave-one-draft-out coefficients (relative value per open spot; 2nd / 3rd rookie at group):")
    for y, (br, bw) in coef.items():
        print(f"  {y}: roster {br:+.3f}   within {bw[1]:+.3f} / {bw[2]:+.3f}")

    from review import RealizedV2
    from trade import Context
    real = RealizedV2(Context())
    rp = pd.read_parquet(OUT / "review_picks.parquet")
    rows = []
    for y in YEARS:
        br, bw = coef[y]
        picks, _ = draft_night(y)
        b = board(y, picks)
        st = {r.team: situation(r) for _, r in S[S.year == y].iterrows()}
        drafter = dict(zip(picks.pick, picks.team))
        tabs = {}
        counts = {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            if tm not in st:
                continue
            key = _key(st[tm])
            tab = tabs.setdefault(key, NTable(st[tm]))
            c = counts.setdefault(tm, {})
            avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp != "SPEC")]
            g_, r_ = avail.grp.to_numpy(), avail["rank"].clip(upper=OFF_BOARD_RANK).to_numpy()
            m_roster = np.array([1 + br * open_at.get((y, tm, g), 0.0) for g in g_])
            m_within = np.array([1 + bw[min(c.get(g, 0), 2)] for g in g_])
            vals = {
                "current": [tab.current(g, c.get(g, 0), r, k) for g, r in zip(g_, r_)],
                "within": [tab.scaled(g, c.get(g, 0), r, k, mw) for g, r, mw in zip(g_, r_, m_within)],
                "roster": [tab.scaled(g, c.get(g, 0), r, k, mr) for g, r, mr in zip(g_, r_, m_roster)],
                "both": [tab.scaled(g, c.get(g, 0), r, k, mr * mw) for g, r, mr, mw in zip(g_, r_, m_roster, m_within)],
            }
            row = {"year": y, "pick": k, "round": p["round"], "team": tm, "player": p.pfr_player_name, "grp": p.grp}
            for s, v in vals.items():
                i = int(np.argmax(v))
                rec = avail.iloc[i]
                row[f"{s}_player"], row[f"{s}_grp"], row[f"{s}_at"] = rec["name"], rec.grp, rec.drafted_at
            rows.append(row)
            if p.grp != "SPEC":
                c[p.grp] = c.get(p.grp, 0) + 1
        print(f"{y} choices done", flush=True)
    D = pd.DataFrame(rows).merge(rp[["year", "pick", "real_team", "rec_player", "real_model"]], on=["year", "pick"], how="left")

    # realized value of each rule's choice, raw and adjusted for depth chart
    st_all = {(r.year, r.team): situation(r) for _, r in S.iterrows()}
    dp = pd.read_parquet("data/raw/draft_picks.parquet")
    from review import team as _team
    DR = {(y, p): _team(t) for y, p, t in zip(dp.season, dp.pick, dp.team)}
    for s in ["current", "within", "roster", "both"]:
        for adj in (False, True):
            col = f"real_{s}" + ("_adj" if adj else "")
            out = []
            for r in D.itertuples():
                k_ = r.pick
                g, at, nm = getattr(r, f"{s}_grp"), getattr(r, f"{s}_at"), getattr(r, f"{s}_player")
                sc = 1.0
                if adj and pd.notna(at) and g != "QB":
                    br = coef[r.year][0]
                    sc = (1 + br * open_at.get((r.year, r.team, g), 0.0)) / (1 + br * open_at.get((r.year, DR[(r.year, int(at))], g), 0.0))
                out.append(real.titles(st_all[(r.year, r.team)], g, r.year, at, k_, nm, scale=sc))
            D[col] = out
        print(f"realized {s} done", flush=True)
    D.to_parquet(OUT / "need_choice.parquet")
    report(D)


def report(D):
    rng = np.random.default_rng(0)

    def boot(x, cl):
        x, cl = np.asarray(x, float), np.asarray(cl)
        ids = np.unique(cl); groups = [x[cl == i] for i in ids]
        means = [np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(2000)]
        return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    D = D[D.real_team.notna()].copy()
    D["rnd"] = pd.cut(D["round"], [0, 1, 4, 7], labels=["Round 1", "Rounds 2-4", "Rounds 5-7"])
    D["cl"] = D.year.astype(str) + D.team
    out = []
    for s in ["current", "within", "roster", "both"]:
        for adj in ("", "_adj"):
            dis = D[D[f"{s}_player"] != D.player].drop_duplicates(["year", f"{s}_player"])
            for rn, g in [("All", dis)] + list(dis.groupby("rnd", observed=True)):
                m, lo, hi = boot(g[f"real_{s}{adj}"] - g.real_team, g.cl)
                out.append({"rule": s, "scoring": "adjusted" if adj else "raw", "rounds": rn, "n": len(g),
                            "model - team": m, "lo95": lo, "hi95": hi})
    T = pd.DataFrame(out)
    print("\nModel's player minus team's player where they disagree (titles, % of one), each rec once per draft:")
    print(T.round(2).to_string(index=False))
    # paired vs the current rule, every pick (no dedup), raw scoring
    print("\nPaired vs the current rule, all picks (raw scoring; repeat recommendations included):")
    for s in ["within", "roster", "both"]:
        for rn, g in [("All", D)] + list(D.groupby("rnd", observed=True)):
            m, lo, hi = boot(g[f"real_{s}"] - g.real_current, g.cl)
            same = (g[f"{s}_player"] == g.current_player).mean()
            print(f"  {s:7s} {rn:10s} {m:+.3f} ({lo:+.3f} to {hi:+.3f})  same choice {same:.0%}")
    print("\nCheck: current rule on fixed tables vs the review's stored recommendations: same player "
          f"{(D.current_player == D.rec_player).mean():.0%} of picks")


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "need_choice.parquet"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
