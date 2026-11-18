"""Availability: can other teams' needs, position runs and fallers' momentum predict "will he still be
there at our next pick?" better than the app's scatter around consensus rank?

Data: Arif Hasan's consensus boards 2014-2026 (review.board), every pick with the team that made it
(review.draft_night), roster need on draft day by team x position (scripts/roster_need.py), QB need
(data/derived/review_states.parquet), optionally top-30 visits (~/cfdarko draft_visits, 2015+).

Pick-choice model: at each pick, the picking team chooses one available board player, or someone off
the board. Conditional logit; utility of board player i for team T at pick t:
  rank:     log(rank+10), its square, x log(t+10); "fallen" = log(t+10) - log(rank+10) when > 0 (and squared)
  position: position x round-bucket offsets (R1 / R2-3 / R4-7), WR the base
  need:     open starting spots at his position on T's roster (clipped), x round 1; QB need now / later;
            players T already took at his position this draft (1, 2+)
  run:      board players at his position taken in the previous 8 picks; how many better-ranked
            players at his position are still available (log1p)
  visit:    T hosted him on a top-30 visit (variant only; when visit reports become public is unclear)
  off-board: one constant per round.

Variants: M0 rank+position, M1 + needs, M2 + runs, M3 + visits. Baselines: the app's scatter (drift and sd
by rank bucket, refit on training years, plus each pick's chance of going to an off-board player), and
the same scatter truncated so a player still on the board at pick t is drawn to go after t.

Test: rolling, fit on drafts before Y, test on Y (2017-2026). Decision points = every pick t <= 160 by a
team that picks again later in the draft (team at each pick = who actually picked, so draft-night trades
are known: slightly optimistic for all variants alike). For every board player still available after t
with consensus rank <= next pick + 32, predict P(still available at the team's next pick) by simulating
the intervening picks; score log loss / Brier against what happened.

Run:  python scripts/availability_test.py [fit|test|score]  (default: fit and test)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from outlook import OUT, RAW  # noqa: E402
from public_board import _norm  # noqa: E402
import review  # noqa: E402

YEARS = list(range(2014, 2027))
TEST_YEARS = list(range(2017, 2027))
GRPS = ["QB", "RB", "WR", "TE", "T", "G", "C", "ED", "IDL", "LB", "CB", "S", "SPEC"]
G = len(GRPS)
GI = {g: i for i, g in enumerate(GRPS)}
NEED_GRPS = ["RB", "WR", "TE", "T", "G", "C", "ED", "IDL", "LB", "CB", "S"]
RUN_WIN = 8
MAX_T = 160
CACHE = OUT / "availability_data.pkl"


def rbucket(t):
    return np.where(t <= 32, 0, np.where(t <= 100, 1, 2))


def round_of(t):
    return np.clip((np.asarray(t) - 1) // 32, 0, 6)


# ---------- data ----------

def roster_needs():
    import roster_need as rn
    s, ps, games, dp = rn.load()
    need = pd.concat([rn.roster_need(ps, y) for y in YEARS], ignore_index=True)
    typical = need.groupby("grp").total_prev.median()
    need["open"] = (need.grp.map(typical) - need.returning - need.added).clip(-1.5, 1.5)
    return need[["year", "team", "grp", "open"]]


def visits():
    v = pd.read_parquet(Path.home() / "cfdarko/data/raw/draft_visits.parquet")
    tm = pd.read_parquet(RAW / "teams.parquet")
    nick = dict(zip(tm.team_nick, tm.team_abbr.map(review.team)))
    nick.update({"Rams": "LA", "Chargers": "LAC", "Raiders": "LV"})
    v["tm"] = v.team.map(nick)
    v["key"] = v.player.map(_norm)
    return set(zip(v.draft_year, v.key, v.tm))


def load_data():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    need = roster_needs().set_index(["year", "team", "grp"]).open
    st = pd.read_parquet(OUT / "review_states.parquet")
    flag = st.qb_from.notna()
    qbnow = dict(zip(zip(st.year, st.team), np.where(flag & (st.qb_from <= 1), st.p_need, 0.0)))
    qblater = dict(zip(zip(st.year, st.team), np.where(flag & (st.qb_from > 1), st.p_need, 0.0)))
    vis = visits()
    years = {}
    for y in YEARS:
        d, _ = review.draft_night(y)
        d = d.sort_values("pick").reset_index(drop=True)
        h = review.board(y, d).sort_values("rank").reset_index(drop=True)
        h["g"] = h.grp_label.map(GI).fillna(GI["SPEC"]).astype(int)
        da = h.drafted_at.to_numpy(float)
        da = np.where(np.isnan(da), 10_000, da)
        teams = sorted(set(d.team))
        ti = {t: i for i, t in enumerate(teams)}
        needm = np.zeros((len(teams), G))
        for t, i in ti.items():
            for g in NEED_GRPS:
                needm[i, GI[g]] = need.get((y, t, g), 0.0)
        chosen = {int(p): i for i, p in enumerate(da) if p < 10_000}
        P = len(d)
        years[y] = dict(
            rank=h["rank"].to_numpy(float), g=h.g.to_numpy(), da=da, name=h.name.to_numpy(),
            key=h.key.to_numpy(), pick_team=d.team.map(ti).to_numpy(), picks=d.pick.to_numpy(),
            chosen=np.array([chosen.get(int(p), -1) for p in d.pick]), teams=teams, need=needm,
            qbnow=np.array([qbnow.get((y, t), 0.0) for t in teams]),
            qblater=np.array([qblater.get((y, t), 0.0) for t in teams]),
            visit=np.array([[(y, k, t) in vis for k in h.key] for t in teams], float),
            has_visits=y >= 2015 and y != 2021)
        print(f"{y}: board {len(h)}, picks {P}, board picks {(np.array([chosen.get(int(p), -1) for p in d.pick]) >= 0).mean():.0%}, "
              f"visits {years[y]['visit'].sum():.0f}")
    pd.to_pickle(years, CACHE)
    return years


# ---------- features ----------

FEATS = (["lr", "lr2", "lr_lt", "fallen", "fallen2"]
         + [f"pos_{g}_{b}" for b in range(3) for g in GRPS if g != "WR"]
         + ["need", "need_r1", "qbnow", "qblater", "own1", "own2", "ownqb",
            "run", "posrank", "visit", "visit_r1"]
         + [f"off_{r}" for r in range(7)])
FI = {f: i for i, f in enumerate(FEATS)}
GROUP_OF = {"M0": ["lr", "lr2", "lr_lt", "fallen", "fallen2", "pos", "off"],
            "M1": ["need", "need_r1", "qbnow", "qblater", "own1", "own2", "ownqb"],
            "M2": ["run", "posrank"], "M3": ["visit", "visit_r1"]}
VARIANTS = {"M0": ["M0"], "M1": ["M0", "M1"], "M2": ["M0", "M1", "M2"], "M3": ["M0", "M1", "M2", "M3"]}


def mask_for(variant):
    keep = set()
    for grp in VARIANTS[variant]:
        for f in GROUP_OF[grp]:
            keep |= {x for x in FEATS if x == f or x.startswith(f + "_")}
    return np.array([f in keep for f in FEATS])


def static_feats(Y, t, team):
    """(N, F) features of every board player at pick t for team index `team`, dynamic ones zero."""
    N = len(Y["rank"])
    X = np.zeros((N, len(FEATS)))
    lr = np.log(Y["rank"] + 10) - 3.0
    lt = np.log(t + 10) - 3.0
    X[:, FI["lr"]] = lr
    X[:, FI["lr2"]] = lr ** 2
    X[:, FI["lr_lt"]] = lr * lt
    fallen = np.clip(lt - lr, 0, None)
    X[:, FI["fallen"]] = fallen
    X[:, FI["fallen2"]] = fallen ** 2
    b = int(rbucket(t))
    for g in GRPS:
        if g != "WR":
            X[Y["g"] == GI[g], FI[f"pos_{g}_{b}"]] = 1
    nd = Y["need"][team][Y["g"]]
    X[:, FI["need"]] = nd
    X[:, FI["need_r1"]] = nd * (t <= 32)
    isqb = Y["g"] == GI["QB"]
    X[:, FI["qbnow"]] = isqb * Y["qbnow"][team]
    X[:, FI["qblater"]] = isqb * Y["qblater"][team]
    if Y["has_visits"]:
        X[:, FI["visit"]] = Y["visit"][team]
        X[:, FI["visit_r1"]] = Y["visit"][team] * (t <= 32)
    return X


def off_feats(t):
    x = np.zeros(len(FEATS))
    x[FI[f"off_{int(round_of(t))}"]] = 1
    return x


def build_choices(years, ys):
    """Stacked alternatives for real picks of drafts `ys`: X (rows, F), choice index per pick."""
    Xs, starts, chosen_rows = [], [], []
    n = 0
    for y in ys:
        Y = years[y]
        N = len(Y["rank"])
        g = Y["g"]
        avail = np.ones(N, bool)
        own = np.zeros((len(Y["teams"]), G))
        recent = []
        for k, (t, tm, ch) in enumerate(zip(Y["picks"], Y["pick_team"], Y["chosen"])):
            idx = np.flatnonzero(avail)
            X = static_feats(Y, t, tm)[idx]
            o = own[tm][g[idx]]
            X[:, FI["own1"]] = (o == 1) & (g[idx] != GI["QB"])
            X[:, FI["own2"]] = (o >= 2) & (g[idx] != GI["QB"])
            X[:, FI["ownqb"]] = (o >= 1) & (g[idx] == GI["QB"])
            runc = np.bincount(np.array(recent[-RUN_WIN:], int), minlength=G) if recent else np.zeros(G)
            X[:, FI["run"]] = runc[g[idx]]
            pr = np.zeros(N)
            for gg in range(G):
                m = (g == gg) & avail
                pr[m] = np.arange(m.sum())
            X[:, FI["posrank"]] = np.log1p(pr[idx])
            X = np.vstack([X, off_feats(t)])
            Xs.append(X)
            starts.append(n)
            if ch >= 0:
                chosen_rows.append(n + int(np.searchsorted(idx, ch)))
                avail[ch] = False
                own[tm, g[ch]] += 1
                recent.append(g[ch])
            else:
                chosen_rows.append(n + len(idx))
            n += len(X)
    return np.vstack(Xs), np.array(starts), np.array(chosen_rows)


def fit(X, starts, chosen, mask, l2=1e-3):
    Xm = X[:, mask]
    ends = np.append(starts[1:], len(X))
    seg = np.repeat(np.arange(len(starts)), ends - starts)

    def nll(b):
        u = Xm @ b
        mx = np.maximum.reduceat(u, starts)
        e = np.exp(u - mx[seg])
        s = np.add.reduceat(e, starts)
        p = e / s[seg]
        ll = (u[chosen] - mx - np.log(s)).sum()
        grad = Xm[chosen].sum(0) - np.add.reduceat(p[:, None] * Xm, starts).sum(0)
        return -ll + l2 * b @ b, -grad + 2 * l2 * b

    r = minimize(nll, np.zeros(mask.sum()), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    beta = np.zeros(len(FEATS))
    beta[mask] = r.x
    return beta, r.fun, Xm, seg


def std_errors(X, starts, chosen, beta, mask):
    Xm = X[:, mask]
    ends = np.append(starts[1:], len(X))
    seg = np.repeat(np.arange(len(starts)), ends - starts)
    u = Xm @ beta[mask]
    mx = np.maximum.reduceat(u, starts)
    e = np.exp(u - mx[seg])
    p = e / np.add.reduceat(e, starts)[seg]
    xbar = np.add.reduceat(p[:, None] * Xm, starts)
    d = Xm - xbar[seg]
    H = (d * p[:, None]).T @ d
    se = np.full(len(FEATS), np.nan)
    se[mask] = np.sqrt(np.diag(np.linalg.pinv(H)))
    return se


# ---------- simulation ----------

def sim_choice(Y, beta, t0, t2, S, rng):
    """P(each board player is still available at pick t2), simulating real picks after t0 (exclusive)
    up to t2 (exclusive) with the choice model; history through t0 is real."""
    N = len(Y["rank"])
    g = Y["g"]
    pos = {int(p): k for k, p in enumerate(Y["picks"])}
    k0 = pos[t0]
    avail = Y["da"] > t0
    own = np.zeros((len(Y["teams"]), G))
    recent = []
    for k in range(k0 + 1):
        ch = Y["chosen"][k]
        if ch >= 0:
            own[Y["pick_team"][k], g[ch]] += 1
            recent.append(g[ch])
    A = np.repeat(avail[None], S, 0)
    OWN = np.repeat(own[None], S, 0)
    REC = np.full((S, RUN_WIN), -1)
    r = recent[-RUN_WIN:]
    if r:
        REC[:, -len(r):] = r
    gidx = [np.flatnonzero(g == gg) for gg in range(G)]
    b_own1, b_own2, b_ownqb = beta[FI["own1"]], beta[FI["own2"]], beta[FI["ownqb"]]
    b_run, b_pr = beta[FI["run"]], beta[FI["posrank"]]
    isqb = g == GI["QB"]
    rows = np.arange(S)
    for k in range(k0 + 1, len(Y["picks"])):
        t = Y["picks"][k]
        if t >= t2:
            break
        tm = Y["pick_team"][k]
        u0 = static_feats(Y, t, tm) @ beta
        uo = off_feats(t) @ beta
        o = OWN[:, tm, :][:, g]
        u = u0[None] + b_own1 * ((o == 1) & ~isqb) + b_own2 * ((o >= 2) & ~isqb) + b_ownqb * ((o >= 1) & isqb)
        if b_run:
            runc = np.zeros((S, G))
            for j in range(RUN_WIN):
                v = REC[:, j] >= 0
                runc[rows[v], REC[v, j]] += 1
            u = u + b_run * runc[:, g]
        if b_pr:
            pr = np.zeros((S, N))
            for gg in range(G):
                ix = gidx[gg]
                if len(ix):
                    a = A[:, ix]
                    pr[:, ix] = np.cumsum(a, 1) - a
            u = u + b_pr * np.log1p(pr)
        u = np.where(A, u, -np.inf) + rng.gumbel(size=(S, N))
        best = u.argmax(1)
        off = (uo + rng.gumbel(size=S)) > u[rows, best]
        take = ~off
        A[rows[take], best[take]] = False
        OWN[rows[take], tm, g[best[take]]] += 1
        REC = np.roll(REC, -1, 1)
        REC[:, -1] = np.where(take, g[best], -1)
    return A.mean(0)


def scatter_params(years, train):
    rank, real = [], []
    on = np.zeros(7); tot = np.zeros(7)
    for y in train:
        Y = years[y]
        rank.append(Y["rank"]); real.append(np.where(Y["da"] < 10_000, Y["da"], 300))
        rd = round_of(Y["picks"])
        np.add.at(tot, rd, 1); np.add.at(on, rd, Y["chosen"] >= 0)
    rank, real = np.concatenate(rank), np.concatenate(real)
    gap = np.log(real + 10) - np.log(rank + 10)
    edges = [0, 10, 32, 64, 100, 150, 200, 330]
    pts, mu, sd = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rank > a) & (rank <= b)
        pts.append(np.median(rank[m])); mu.append(gap[m].mean()); sd.append(gap[m].std())
    return dict(pts=pts, mu=mu, sd=sd, q=on / tot)


def sim_scatter(Y, sp, t0, t2, S, rng, truncate):
    N = len(Y["rank"])
    avail = Y["da"] > t0
    r = Y["rank"]
    mu = np.log(r + 10) + np.interp(r, sp["pts"], sp["mu"])
    sd = np.interp(r, sp["pts"], sp["sd"])
    if truncate:  # still on the board after t0: he goes after t0
        lo = ndtr((np.log(t0 + 0.5 + 10) - mu) / sd)
        u = lo + (1 - lo) * rng.random((S, N))
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    else:
        z = rng.standard_normal((S, N))
    real = mu + sd * z
    real = np.where(avail[None], real, np.inf)
    between = [p for p in Y["picks"] if t0 < p < t2]
    nb = (rng.random((S, len(between))) < sp["q"][round_of(np.array(between, int))]).sum(1) if between else np.zeros(S, int)
    order = np.argsort(real, 1)
    rk = np.empty_like(order)
    rk[np.arange(S)[:, None], order] = np.arange(N)[None]
    gone = rk < nb[:, None]
    return (avail[None] & ~gone).mean(0)


# ---------- test ----------

def decision_points(Y):
    out = []
    picks, teams = Y["picks"], Y["pick_team"]
    for k, (t, tm) in enumerate(zip(picks, teams)):
        if t > MAX_T:
            break
        later = picks[k + 1:][teams[k + 1:] == tm]
        if len(later):
            out.append((int(t), int(later[0])))
    return out


def run_test(years, S=400):
    rows = []
    coefs = {}
    for y in TEST_YEARS:
        train = [x for x in YEARS if x < y]
        X, st, ch = build_choices(years, train)
        betas = {v: fit(X, st, ch, mask_for(v))[0] for v in VARIANTS}
        coefs[y] = betas
        sp = scatter_params(years, train)
        Y = years[y]
        rng = np.random.default_rng(y)
        for t0, t2 in decision_points(Y):
            cand = (Y["da"] > t0) & (Y["rank"] <= t2 + 32)
            if not cand.any():
                continue
            yv = (Y["da"] >= t2)[cand]
            preds = {"scatter": sim_scatter(Y, sp, t0, t2, S, rng, False),
                     "scatter_trunc": sim_scatter(Y, sp, t0, t2, S, rng, True)}
            for v, b in betas.items():
                preds[v] = sim_choice(Y, b, t0, t2, S, rng)
            base = dict(year=y, t0=t0, t2=t2)
            for j, i in enumerate(np.flatnonzero(cand)):
                rows.append({**base, "player": Y["name"][i], "rank": Y["rank"][i], "grp": GRPS[Y["g"][i]],
                             "drafted_at": Y["da"][i], "avail": int(yv[j]), **{m: p[i] for m, p in preds.items()}})
        print(f"{y}: {len(decision_points(Y))} decision points done", flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "availability_test.parquet")
    pd.to_pickle(coefs, OUT / "availability_coefs.pkl")
    return R


MODELS = ["scatter", "scatter_trunc", "M0", "M1", "M2", "M3"]


def score(R, S=400):
    eps = 0.5 / S
    out = []
    subsets = {"all": R.index == R.index,
               "round 1 decisions": R.t0 <= 32, "rounds 2-3": (R.t0 > 32) & (R.t0 <= 100), "rounds 4-5": R.t0 > 100,
               "QBs": R.grp == "QB", "fallers (rank < t0)": R["rank"] < R.t0,
               "contested (scatter 0.1-0.9)": R.scatter.between(0.1, 0.9),
               "2025-26 only": R.year >= 2025}
    for name, m in subsets.items():
        d = R[m]
        rec = {"subset": name, "n": len(d), "avail rate": d.avail.mean()}
        for k in MODELS:
            p = d[k].clip(eps, 1 - eps)
            rec[k] = -(d.avail * np.log(p) + (1 - d.avail) * np.log(1 - p)).mean()
        out.append(rec)
    T = pd.DataFrame(out)
    print("\nLog loss, P(still available at the team's next pick) (lower is better):")
    print(T.round(4).to_string(index=False))
    B = pd.DataFrame([{"subset": n, **{k: ((R[m][k] - R[m].avail) ** 2).mean() for k in MODELS}} for n, m in subsets.items()])
    print("\nBrier:")
    print(B.round(4).to_string(index=False))
    # paired bootstrap by draft-decision (cluster), best model vs scatter
    rng = np.random.default_rng(0)
    R = R.assign(cl=R.year.astype(str) + "_" + R.t0.astype(str))
    for k in MODELS[1:]:
        p0, p1 = R.scatter.clip(eps, 1 - eps), R[k].clip(eps, 1 - eps)
        ll = lambda p: -(R.avail * np.log(p) + (1 - R.avail) * np.log(1 - p))
        R["d"] = ll(p0) - ll(p1)
        g = R.groupby("cl").d.agg(["sum", "size"])
        cl = g.index.to_numpy()
        bs = []
        for _ in range(1000):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            bs.append(s["sum"].sum() / s["size"].sum())
        yr = R.groupby("year").d.mean()
        print(f"{k:14s} gain vs scatter {R.d.mean():+.4f} (95% {np.percentile(bs, 2.5):+.4f} to {np.percentile(bs, 97.5):+.4f}); "
              f"years better: {(yr > 0).sum()}/{len(yr)}")
    # calibration
    print("\nCalibration (predicted bin -> actual availability):")
    bins = [0, .1, .3, .5, .7, .9, 1.0001]
    C = {k: R.groupby(pd.cut(R[k], bins, right=False), observed=True).avail.agg(["mean", "size"]) for k in ["scatter", "scatter_trunc", "M2"]}
    print(pd.concat(C, axis=1).round(3).to_string())


def show_coefs(years):
    X, st, ch = build_choices(years, YEARS)
    for v in ["M2", "M3"]:
        m = mask_for(v)
        b, f, *_ = fit(X, st, ch, m)
        se = std_errors(X, st, ch, b, m)
        T = pd.DataFrame({"coef": b, "se": se}, index=FEATS)[m]
        T["z"] = T.coef / T.se
        print(f"\n{v} on all drafts 2014-2026 (picks: {len(st)}), nll {f:.1f}:")
        print(T[~T.index.str.startswith(("pos_", "off_"))].round(3).to_string())
    # position x round offsets, M2
    b, *_ = fit(X, st, ch, mask_for("M2"))
    P = pd.DataFrame({b_: [0.0 if g == "WR" else b[FI[f"pos_{g}_{b_}"]] for g in GRPS] for b_ in range(3)}, index=GRPS)
    P.columns = ["R1", "R2-3", "R4-7"]
    print("\nPosition offsets vs WR at the same consensus rank (M2):")
    print(P.round(2).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    years = load_data()
    if what in ("fit", "all"):
        show_coefs(years)
    if what in ("test", "all"):
        R = run_test(years)
        score(R)
    if what == "score":
        score(pd.read_parquet(OUT / "availability_test.parquet"))
