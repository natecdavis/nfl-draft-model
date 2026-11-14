"""Complaint #4 (late rounds): value a player still on the board at where he'd really land, not at this pick.

Fall shading (draftday.DraftEngine.effective_talent, weight 0.85) treats a player still available at pick k as if
he'll turn out like a pick-k player. It was fit on where fallers really went (undrafted = pick 300). On draft day a
player still there at k will land at k *or later*, often much later: late recommendations went far later or undrafted
76-79% of the time and delivered ~0.36x their predicted titles (README "Calibration, fall shading and cap cost").

Variants (same tables, states, need counts and board as the production review; leave-one-draft-out engine via the
simulator cache):
  now       current: talent shaded toward k (L = k)
  fallen    players already past their rank (k > rank): average over landing L >= k
  all       every player: average over landing L >= k (a player not yet past his rank can still fall)
Landing L: log(L + 10) ~ normal(log(rank + 10) + drift(rank), spread(rank)) (draftday's availability model, Hasan
2014-24), truncated at L >= k, undrafted mass at 300; talent = shade(rank, L) with the same 0.85 rule (no shading if
L <= rank). Value = the table at that talent and this pick (the cost of the pick is still k).

Scored on the fair yardstick (review.RealizedV2.titles_rec; team picks as played), each rec once per draft and band
(averaged), plus paired per-draft sums vs "now". Run: .venv/bin/python scripts/landing_shade.py [report]
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import review as R  # noqa: E402
from draftday import drift, spread  # noqa: E402
from outlook import OUT  # noqa: E402
from positions import blended, weights  # noqa: E402
from simulator import FALL_WEIGHT, get_table, situation  # noqa: E402
from trade import Context  # noqa: E402

YEARS = range(2014, 2023)
VARIANTS = ["now", "fallen", "all"]
UNDRAFTED = 300
NODES = 24
CACHE = OUT / "landing_shade.parquet"
rng = np.random.default_rng(0)
X, Y = np.log(R.T_GRID + 10), np.log(R.K_GRID + 10)


def shade(r, L):
    r, L = np.asarray(r, float), np.asarray(L, float)
    lt, lp = np.log(r + 10), np.log(L + 10)
    return np.where(L <= r, r, np.exp(lt + FALL_WEIGHT * (lp - lt)) - 10)


def landing(r, k):
    """Landing nodes and weights for a player of consensus rank r still available at pick k."""
    mu, sd = np.log(r + 10) + drift(r), spread(r)
    lo, hi = np.log(k + 10), np.log(UNDRAFTED + 10)
    if lo >= hi:
        return np.array([float(UNDRAFTED)]), np.array([1.0])
    edges = np.linspace(lo, hi, NODES + 1)
    c = norm.cdf(edges, mu, sd)
    w = np.append(np.diff(c), 1 - c[-1])  # last: undrafted
    L = np.append(np.exp((edges[:-1] + edges[1:]) / 2) - 10, UNDRAFTED)
    if w.sum() <= 1e-12:  # far past any plausible landing: he goes undrafted
        return np.array([float(UNDRAFTED)]), np.array([1.0])
    return L, w / w.sum()


def raw(tab, g, level, t, k, p_need):
    """review.Table.lookup at a given (already shaded) talent t (array)."""
    A = tab.v[g]
    lt, lk = np.log(np.asarray(t, float) + 10), np.log(k + 10)
    i = np.clip(np.searchsorted(X, lt) - 1, 0, len(X) - 2); j = int(np.clip(np.searchsorted(Y, lk) - 1, 0, len(Y) - 2))
    fx = np.clip((lt - X[i]) / (X[i + 1] - X[i]), 0, 1); fy = np.clip((lk - Y[j]) / (Y[j + 1] - Y[j]), 0, 1)
    at = lambda M: (1 - fx) * (1 - fy) * M[i, j] + fx * (1 - fy) * M[i + 1, j] + (1 - fx) * fy * M[i, j + 1] + fx * fy * M[i + 1, j + 1]
    if g == "QB" and level == 0 and p_need < 1.0:
        return p_need * at(A[0]) + (1 - p_need) * at(A[1])
    return at(A[level])


def make_lookup(tab, variant):
    def f(g, level, rank, k, p_need):
        if g not in tab.v:
            return -1.0
        r = min(rank, R.OFF_BOARD_RANK)
        if variant == "now" or (variant == "fallen" and k <= r):
            return float(raw(tab, g, level, [tab.eng.effective_talent(r, k)], k, p_need)[0])
        L, w = landing(r, k)
        return float(w @ raw(tab, g, level, shade(r, L), k, p_need))
    return f


def choices():
    S = pd.read_parquet(OUT / "review_states.parquet")
    rows = []
    for y in YEARS:
        picks, _ = R.draft_night(y)
        b = R.board(y, picks)
        W = weights(y)
        st = {r.team: r for r in S[S.year == y].itertuples()}
        tabs, counts = {}, {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            r = st[tm]
            sit = situation(r._asdict())
            key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
            if key not in tabs:
                tabs[key] = get_table(sit, year=y)
            tab = tabs[key]
            c = counts.setdefault(tm, {})
            avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
            row = {"year": y, "pick": k, "round": p["round"], "team": tm, "player": p.pfr_player_name, "grp": p.grp,
                   "p_need": r.p_need}
            for vname in VARIANTS:
                f = make_lookup(tab, vname)
                v = np.array([blended(f, lab, g, min(c.get(g, 0), 2), rk, k, r.p_need, W)
                              for lab, g, rk in zip(avail.pos, avail.grp_label, avail["rank"])])
                best = avail.iloc[int(np.argmax(v))]
                row[f"rec_{vname}"], row[f"grp_{vname}"], row[f"at_{vname}"] = best["name"], best.grp, best.drafted_at
                row[f"rank_{vname}"], row[f"pred_{vname}"] = best["rank"], v.max()
            rows.append(row)
            if p.grp != "SPEC":
                c[p.grp] = c.get(p.grp, 0) + 1
        print(f"{y} done", flush=True)
    return pd.DataFrame(rows)


def score(D):
    S = pd.read_parquet(OUT / "review_states.parquet")
    sits = {(r.year, r.team): situation(r._asdict()) for r in S.itertuples()}
    real = R.RealizedV2(Context())
    D["real_team"] = [real.titles(sits[(y, tm)], g, y, k, k, n) for y, tm, g, k, n in zip(D.year, D.team, D.grp, D.pick, D.player)]
    cache = {}
    for v in VARIANTS:
        out = []
        for y, tm, g, at, k, n, pn in zip(D.year, D.team, D[f"grp_{v}"], D[f"at_{v}"], D.pick, D[f"rec_{v}"], D.p_need):
            key = (y, tm, k, n)
            if key not in cache:
                cache[key] = real.titles_rec(sits[(y, tm)], pn, g, y, at, k, n)
            out.append(cache[key])
        D[f"real_{v}"] = out
    car = R.Careers()
    D["gs_team"] = [car.get("gs", y, k, n) for y, k, n in zip(D.year, D.pick, D.player)]
    for v in VARIANTS:
        D[f"gs_{v}"] = [car.get("gs", y, a, n) for y, a, n in zip(D.year, D[f"at_{v}"], D[f"rec_{v}"])]
    return D


def boot_years(d, col):
    ys = d.year.unique()
    g = {y: d.loc[d.year == y, col].to_numpy() for y in ys}
    b = [np.concatenate([g[y] for y in rng.choice(ys, len(ys))]).mean() for _ in range(2000)]
    return f"{d[col].mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


def report(D):
    D["band"] = R.band_of(D["round"])
    print("Model minus team where they disagree, each rec once, averaged (titles x100, 95% over draft years); fair yardstick")
    for v in VARIANTS:
        d = D.rename(columns={f"rec_{v}": "rec_player", f"grp_{v}": "rec_grp", f"at_{v}": "rec_drafted_at",
                              f"real_{v}": "real_model", f"pred_{v}": "pred", f"gs_{v}": "gs_model"})
        d = d[["year", "pick", "round", "team", "player", "grp", "real_team", "rec_player", "rec_grp", "rec_drafted_at",
               "real_model", "pred", "gs_model", "gs_team"]]
        d = d[d.real_team.notna() & d.real_model.notna()].copy()
        d["d_t"], d["d_gs"] = d.real_model - d.real_team, d.gs_model - d.gs_team
        o = R.once_per_draft(d)
        print(f"  {v}")
        for b in ("R1", "R2-4", "R5-7"):
            x = o[o.band == b]
            und = x.rec_drafted_at.isna().mean()
            print(f"    {b:5s} titles {boot_years(x, 'd_t')} n={len(x)} | starts {x.d_gs.mean():+.2f} | delivered/predicted "
                  f"{x.real_model.sum() / x.pred.sum():.2f} | recs undrafted {und:.0%}")
    print("\nPaired vs now, every pick (repeat recs included), sum per draft, mean over drafts (95% over draft years):")
    for v in VARIANTS[1:]:
        for rn, lab in [((1, 1), "R1"), ((2, 4), "R2-4"), ((5, 7), "R5-7")]:
            d = D[D["round"].between(*rn)].dropna(subset=[f"real_{v}", "real_now"])
            per = d.assign(x=d[f"real_{v}"] - d["real_now"]).groupby("year").x.sum()
            b = [rng.choice(per.to_numpy(), len(per)).mean() for _ in range(2000)]
            ch = (d[f"rec_{v}"] != d["rec_now"]).mean()
            print(f"  {v:6s} {lab:4s} {per.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f}) per draft,"
                  f" rec changed at {ch:.0%} of picks")


def main():
    if CACHE.exists() and sys.argv[1:2] == ["report"]:
        D = pd.read_parquet(CACHE)
    else:
        D = choices()
        rp = pd.read_parquet(OUT / "review_picks.parquet")
        chk = D.merge(rp[["year", "pick", "rec_player"]], on=["year", "pick"])
        print(f"check: 'now' matches the production review's recommendation at {(chk['rec_now'] == chk.rec_player).mean():.1%} of picks")
        D = score(D)
        D.to_parquet(CACHE)
    report(D)


if __name__ == "__main__":
    main()
