"""Season-level "DARKO for PFF": context-adjusted talent from PFF season grades, with a movers test.

Per position group g, a state-space model on z = PFF season grade minus the (group x season)
snap-weighted mean:
    z_it   = theta_it + beta_g . ctx_it + gamma_g * opp_it + e,   Var(e) = a + b * 500 / snaps
    theta  : entrant ~ N(mu0, s0^2); then theta_{t+k} = rho^k theta_t + k * (c1 + c2 * (age - 26)) + eta,
             Var(eta) over k seasons = tau^2 (1 - rho^2k) / (1 - rho^2)
ctx = snap-weighted PRE-SEASON talent (Kalman prior) of the player's teammates in context units
(own contribution excluded, centered by season): QB <- OL, receivers; RB <- OL, QB; WR/TE <- QB, OL,
other receivers; OL <- QB; DL/EDGE/LB/DB <- pass rushers, coverage. opp = schedule-average pre-season
talent of opponents' other side of the ball. Using priors, not same-season grades, keeps the context
causal (no shared-shock co-movement, no feedback). Chains are (player, grade family: QB, RB, receivers, OL,
defense), so DL/EDGE/LB/DB and WR/TE moves keep their history; each season uses its own group's parameters.

Fit: Gaussian one-step-ahead likelihood, parameters by group, coordinate passes (ctx recomputed from
the latest priors between passes). Burn-in 2006, likelihood on 2007-2017, test on 2018-2025.

Movers test (test seasons, snaps >= 300, chain seen before, team != team at last season):
  shrunk  : last season's shrunk grade (as in the label, K=200), linear calibration fit on 2007-17
  nocxt   : same filter with beta = gamma = 0 (refit)
  full    : prior talent + beta . new-team ctx + gamma . new schedule
  talent  : full model's prior talent alone (no context term)
Also stayers, and next-season log snaps from the talent alone (team's judgment, not PFF).

Output (all-season fits, without and with context): filtered talent and a Rauch-Tung-Striebel smoothed talent
(talent_smooth: all seasons; talent_win: rookie-deal seasons look ahead only to the end of year 4).

Usage: python scripts/darko_pff.py            # validate + write data/derived/darko_pff.parquet (~7 min)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PFF = Path.home() / "cfdarko/data/raw/pff_nfl"
OUT = ROOT / "data/derived/darko_pff.parquet"
FIRST, TRAIN_END, LAST = 2006, 2017, 2025
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
OFF = {"QB", "RB", "WR", "TE", "OL"}
POS2G = {"QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL",
         "DI": "DL", "ED": "EDGE", "LB": "LB", "CB": "DB", "S": "DB"}
TAB = {"QB": "passing_summary", "RB": "rushing_summary", "WR": "receiving_summary", "TE": "receiving_summary",
       "OL": "offense_blocking", **{g: "defense_summary" for g in ["DL", "EDGE", "LB", "DB"]}}
UNITS = {"QBu": ["QB"], "OLu": ["OL"], "RECu": ["WR", "TE"], "PRu": ["DL", "EDGE"], "COVu": ["LB", "DB"]}
CTX = {"QB": ["OLu", "RECu"], "RB": ["OLu", "QBu"], "WR": ["QBu", "OLu", "RECu"], "TE": ["QBu", "OLu", "RECu"],
       "OL": ["QBu"], **{g: ["PRu", "COVu"] for g in ["DL", "EDGE", "LB", "DB"]}}
TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "STL": "LA", "LAR": "LA",
        "SD": "LAC", "OAK": "LV"}
K_SHRINK = 200


def load():
    rows = []
    for y in range(FIRST, LAST + 1):
        ob = pd.read_csv(PFF / str(y) / "offense_blocking.csv").drop_duplicates("player_id")
        osn = ob.set_index("player_id")["snap_counts_offense"]
        for name in sorted(set(TAB.values())):
            t = pd.read_csv(PFF / str(y) / f"{name}.csv").drop_duplicates("player_id")
            t["g"] = t["position"].map(POS2G)
            t = t[t["g"].notna() & (t["g"].map(TAB) == name)]
            d = name == "defense_summary"
            snaps = t["snap_counts_defense"] if d else t["player_id"].map(osn)
            if name == "passing_summary":
                snaps = np.fmax(snaps.fillna(0), t["passing_snaps"])
            rows.append(pd.DataFrame({"player_id": t["player_id"].to_numpy(), "player": t["player"].to_numpy(),
                                      "season": y, "g": t["g"].to_numpy(),
                                      "tm": t["team_name"].replace(TEAM).to_numpy(),
                                      "grade": (t["grades_defense"] if d else t["grades_offense"]).to_numpy(),
                                      "snaps": np.asarray(snaps, float)}))
    s = pd.concat(rows, ignore_index=True)
    s = s[(s.snaps > 0) & s.grade.notna()]
    s = s.sort_values("snaps").drop_duplicates(["player_id", "season"], keep="last")
    mu = s.assign(gs=s.grade * s.snaps).groupby(["g", "season"])[["gs", "snaps"]].sum()
    s = s.join((mu.gs / mu.snaps).rename("mu"), on=["g", "season"])
    s["z"] = s.grade - s.mu
    s["shrunk"] = s.z * s.snaps / (s.snaps + K_SHRINK)
    pl = pd.read_parquet(ROOT / "data/raw/players.parquet")[["pff_id", "birth_date"]].dropna()
    pl = pl.assign(pff_id=pl.pff_id.astype(int)).drop_duplicates("pff_id").set_index("pff_id")
    bd = pd.to_datetime(s.player_id.map(pl.birth_date))
    s["age"] = (s.season + 0.7 - bd.dt.year - bd.dt.dayofyear / 365).fillna(26.0).clip(20, 42)
    s = s.sort_values(["season", "g", "player_id"]).reset_index(drop=True)
    # chain bookkeeping: previous obs of the same (player, group)
    s["chain"] = s.groupby(["player_id", s.g.map(FAM)]).ngroup()
    grp = s.groupby("chain")
    s["prev_season"] = grp.season.shift()
    s["prev_tm"] = grp.tm.shift()
    s["prev_shrunk"] = grp.shrunk.shift()
    s["mover"] = s.prev_season.notna() & (s.tm != s.prev_tm)
    return s


def schedule_pairs():
    sch = pd.read_parquet(ROOT / "data/raw/schedules.parquet")
    sch = sch[sch.game_type == "REG"]
    h, a = sch.home_team.replace(TEAM), sch.away_team.replace(TEAM)
    return pd.DataFrame({"season": np.r_[sch.season, sch.season], "tm": np.r_[h, a], "opp": np.r_[a, h]})


# ---------------------------------------------------------------- parameters
NB = {g: len(CTX[g]) for g in GROUPS}


def unpack(th, g):
    rho = 1 / (1 + np.exp(-th[0]))
    return dict(rho=rho, tau=np.exp(th[1]), a=np.exp(th[2]), b=np.exp(th[3]), mu0=th[4], s0=np.exp(th[5]),
                c1=th[6], c2=th[7], beta=np.asarray(th[8:8 + NB[g]]), gamma=th[8 + NB[g]])


def init(g):
    return np.r_[2.0, np.log(3.0), np.log(10.0), np.log(20.0), -3.0, np.log(6.0), 0.0, 0.0, np.zeros(NB[g]), 0.0]


# ---------------------------------------------------------------- filter (all groups)
FAM = {"QB": "QB", "RB": "RB", "WR": "REC", "TE": "REC", "OL": "OL", "DL": "DEF", "EDGE": "DEF", "LB": "DEF", "DB": "DEF"}


def row_params(thetas, s):
    """Per-row parameter arrays from each row's group (transition into a season uses that season's group)."""
    gi = s.g.map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    P = np.array([[unpack(thetas[g], g)[k] for k in ["rho", "tau", "a", "b", "mu0", "s0", "c1", "c2"]]
                  for g in GROUPS])
    return {k: P[gi, j] for j, k in enumerate(["rho", "tau", "a", "b", "mu0", "s0", "c1", "c2"])}


def ctx_term(thetas, s, C, use_ctx):
    ct = np.zeros(len(s))
    if use_ctx:
        for g in GROUPS:
            m = s.g.to_numpy() == g
            p = unpack(thetas[g], g)
            ct[m] = C.loc[m, CTX[g]].to_numpy() @ p["beta"] + p["gamma"] * C.loc[m, "opp"].to_numpy()
    return ct


def filt(thetas, s, C, use_ctx=True):
    """Kalman filter over all rows (season-sorted); chains are (player, grade family), so DL/EDGE/LB/DB moves
    keep their history. Returns per-row prior mean/var, obs noise R, context term, posterior, loglik."""
    p = row_params(thetas, s)
    n = len(s)
    chain = s.chain.to_numpy(); season = s.season.to_numpy(); age = s.age.to_numpy()
    z = s.z.to_numpy(); snaps = np.maximum(s.snaps.to_numpy(), 10)
    nc = chain.max() + 1
    m = np.full(nc, np.nan); P = np.full(nc, np.nan); last = np.full(nc, -1)
    mp, Pp, mo, Po, ll = (np.empty(n) for _ in range(5))
    R = p["a"] + p["b"] * 500 / snaps
    cterm = ctx_term(thetas, s, C, use_ctx)
    bounds = np.r_[0, np.flatnonzero(np.diff(season)) + 1, n]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        c = chain[lo:hi]; y = season[lo]; sl = slice(lo, hi)
        rho, tau = p["rho"][sl], p["tau"][sl]
        seen = last[c] >= 0
        k = np.where(seen, y - last[c], 0)
        rk = rho ** k
        drift = k * (p["c1"][sl] + p["c2"][sl] * (age[sl] - 0.5 * k - 26))
        mpr = np.where(seen, rk * m[c] + drift, p["mu0"][sl])
        Ppr = np.where(seen, rk ** 2 * P[c] + tau ** 2 * (1 - rk ** 2) / (1 - rho ** 2), p["s0"][sl] ** 2)
        S = Ppr + R[sl]
        v = z[sl] - mpr - cterm[sl]
        Kg = Ppr / S
        mp[sl], Pp[sl] = mpr, Ppr
        mo[sl] = mpr + Kg * v; Po[sl] = (1 - Kg) * Ppr
        ll[sl] = -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)
        m[c], P[c], last[c] = mo[sl], Po[sl], y
    return dict(mp=mp, Pp=Pp, R=R, cterm=cterm, mo=mo, Po=Po, ll=ll)


def fit(s, sp, train_end, use_ctx=True, passes=3, log=True):
    """Coordinate passes: each group's parameters by maximum likelihood on its own train rows, holding the
    other groups' parameters (and the context, from the latest priors) fixed."""
    thetas = {g: init(g) for g in GROUPS}
    C = context(s, np.zeros(len(s)), sp)
    yr = s.season.to_numpy(); grp = s.g.to_numpy()
    for it in range(passes):
        tot = 0.0
        for g in GROUPS:
            tr = (grp == g) & (yr > FIRST) & (yr <= train_end)
            nb = NB[g] + 1

            def nll(th):
                t = dict(thetas); t[g] = th if use_ctx else np.r_[th, np.zeros(nb)]
                return -filt(t, s, C, use_ctx)["ll"][tr].sum()
            th0 = thetas[g] if use_ctx else thetas[g][:8]
            r = minimize(nll, th0, method="L-BFGS-B")
            thetas[g] = r.x if use_ctx else np.r_[r.x, np.zeros(nb)]
            tot += r.fun
        if log:
            print(f"  pass {it + 1} ({'ctx' if use_ctx else 'no ctx'}): train -loglik {tot:,.0f}", flush=True)
        if use_ctx:
            C = context(s, filt(thetas, s, C, use_ctx)["mp"], sp)
    f = filt(thetas, s, C, use_ctx)
    return thetas, pd.DataFrame({k: f[k] for k in ["mp", "Pp", "R", "cterm", "mo", "Po", "ll"]}, index=s.index), C


def context(s, prior, sp):
    """prior: array of pre-season talent per row of s. Returns ctx unit columns and opp per row."""
    w = s.snaps.to_numpy(); m = prior
    out = pd.DataFrame(index=s.index)
    key = pd.MultiIndex.from_arrays([s.tm, s.season])
    for u, gs in UNITS.items():
        inu = s.g.isin(gs).to_numpy()
        agg = pd.DataFrame({"tm": s.tm, "season": s.season, "w": w * inu, "wm": w * m * inu}) \
            .groupby(["tm", "season"])[["w", "wm"]].sum()
        W = agg.w.reindex(key).to_numpy(); WM = agg.wm.reindex(key).to_numpy()
        own = inu * w
        val = (WM - own * m) / np.maximum(W - own, 1e-9)
        val = np.where(W - own > 50, val, np.nan)
        col = pd.Series(val, index=s.index)
        col = col - col.groupby(s.season).transform("mean")
        out[u] = col.fillna(0.0).to_numpy()
    # opponent strength: schedule-average of opponents' other-side pre-season talent
    off = s.g.isin(OFF).to_numpy()
    side = pd.DataFrame({"tm": s.tm, "season": s.season, "off": off, "w": w, "wm": w * m})
    agg = side.groupby(["tm", "season", "off"])[["w", "wm"]].sum()
    strength = (agg.wm / agg.w).rename("str").reset_index()
    sp2 = sp.merge(strength, left_on=["opp", "season"], right_on=["tm", "season"], suffixes=("", "_o"))
    sos = sp2.groupby(["tm", "season", "off"])["str"].mean()  # opp's side `off` strength
    # an offensive player faces the opponent's defense (off=False) and vice versa
    look = pd.MultiIndex.from_arrays([s.tm, s.season, ~off])
    o = pd.Series(sos.reindex(look).to_numpy(), index=s.index)
    o = o - o.groupby([s.season, s.g]).transform("mean")
    out["opp"] = o.fillna(0.0).to_numpy()
    return out


# ---------------------------------------------------------------- fitting
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def evaluate(s, full, noc):
    E = s.assign(full=full.mp + full.cterm, talent=full.mp, nocxt=noc.mp, cterm=full.cterm,
                 lsn=np.log1p(s.snaps))
    E["prev_lsn"] = E.groupby("chain").lsn.shift()
    base = E[E.prev_season.notna()].copy()
    tr = base[(base.season <= TRAIN_END) & (base.snaps >= 300)]
    # linear calibration of each predictor on training seasons (fair to the shrunk-grade baseline)
    preds = {}
    for name in ["prev_shrunk", "nocxt", "talent", "full"]:
        b = np.polyfit(tr[name], tr.z, 1)
        preds[name] = np.polyval(b, base[name])
    te = (base.season > TRAIN_END) & (base.snaps >= 300)
    rows = []
    for g in GROUPS + ["ALL"]:
        for lab, msk in [("movers", base.mover), ("stayers", ~base.mover)]:
            sel = te & msk & ((base.g == g) if g != "ALL" else True)
            if sel.sum() < 20:
                continue
            y = base.z[sel].to_numpy()
            row = dict(g=g, sample=lab, n=int(sel.sum()))
            for name, lab2 in [("prev_shrunk", "shrunk"), ("nocxt", "nocxt"), ("talent", "talent"),
                               ("full", "full")]:
                row[lab2] = r2(y, preds[name][sel.to_numpy()])
            rows.append(row)
    R = pd.DataFrame(rows)
    # within-group R2 for ALL: demean by group already (z centered by group-season)
    # snaps check: next-season log snaps from talent alone (calibrated within group on train)
    srows = []
    for g in GROUPS:
        d = base[base.g == g]
        trg = d[d.season <= TRAIN_END]
        teg = d[(d.season > TRAIN_END) & d.mover]
        row = dict(g=g, n=len(teg))
        for name in ["prev_shrunk", "nocxt", "talent"]:
            X = np.c_[np.ones(len(trg)), trg[name], trg.prev_lsn]
            b = np.linalg.lstsq(X, trg.lsn, rcond=None)[0]
            row[name] = r2(teg.lsn.to_numpy(), np.c_[np.ones(len(teg)), teg[name], teg.prev_lsn] @ b)
        srows.append(row)
    return R, pd.DataFrame(srows), base


def main():
    pd.set_option("display.width", 200)
    s = load()
    sp = schedule_pairs()
    print(f"{len(s):,} player-seasons {FIRST}-{LAST}, {s.chain.nunique():,} chains; "
          f"movers with >=300 snaps in test seasons: {int((s.mover & (s.season > TRAIN_END) & (s.snaps >= 300)).sum())}")
    print("fit with context (train <= %d):" % TRAIN_END)
    th_full, full, C = fit(s, sp, TRAIN_END, use_ctx=True)
    print("fit without context:")
    th_noc, noc, _ = fit(s, sp, TRAIN_END, use_ctx=False)
    print("\nparameters (context model, train fit):")
    prm = []
    for g in GROUPS:
        p = unpack(th_full[g], g)
        q = unpack(th_noc[g], g)
        row = dict(g=g, rho=p["rho"], tau=p["tau"], a=p["a"], b=p["b"], mu0=p["mu0"], s0=p["s0"],
                   c1=p["c1"], c2=p["c2"], gamma=p["gamma"], rho_noc=q["rho"], tau_noc=q["tau"])
        row.update({f"b_{u}": v for u, v in zip(CTX[g], p["beta"])})
        prm.append(row)
    print(pd.DataFrame(prm).round(3).to_string(index=False))
    # context effect sizes: points per SD of the context feature
    print("\ncontext: grade points per 1 SD of feature (train fit)")
    eff = []
    for g in GROUPS:
        idx = s.g == g
        p = unpack(th_full[g], g)
        for u, b in zip(CTX[g] + ["opp"], list(p["beta"]) + [p["gamma"]]):
            eff.append(dict(g=g, feature=u, pts_per_sd=b * C.loc[idx, u].std()))
    print(pd.DataFrame(eff).pivot(index="g", columns="feature", values="pts_per_sd").round(2).to_string())
    R, S, base = evaluate(s, full, noc)
    print(f"\n== next-season grade, test seasons {TRAIN_END + 1}-{LAST}, snaps >= 300: out-of-sample R^2 ==")
    print("   shrunk = last season's shrunk grade (label's q); nocxt = filter w/o context; "
          "talent = context-adjusted prior alone; full = talent + new team ctx + schedule")
    print(R.round(3).to_string(index=False))
    print("\n== movers: next-season log snaps (team's judgment), calibrated with prior log snaps ==")
    print(S.round(3).to_string(index=False))
    write_output(s, sp)


def entry_year(s):
    """First NFL season: players.parquet rookie_season/draft_year by PFF id, else first PFF season seen
    (NaN if first seen in the burn-in year FIRST: may have entered earlier)."""
    pl = pd.read_parquet(ROOT / "data/raw/players.parquet")[["pff_id", "rookie_season", "draft_year"]]
    pl = pl.dropna(subset=["pff_id"]).assign(pff_id=lambda d: d.pff_id.astype(int)).drop_duplicates("pff_id")
    e = pl.set_index("pff_id").rookie_season.fillna(pl.set_index("pff_id").draft_year)
    first = s.groupby("player_id").season.transform("min")
    out = s.player_id.map(e).astype(float)
    return out.fillna(first.where(first > FIRST).astype(float))


def smooth(s, f, thetas, win_end):
    """Rauch-Tung-Striebel smoother per chain. win_end (per row): for seasons <= win_end, look ahead only
    to seasons <= win_end (rookie-deal window); later seasons use every later season."""
    rho = row_params(thetas, s)["rho"]
    chain, season = s.chain.to_numpy(), s.season.to_numpy()
    nxt = pd.Series(np.arange(len(s))).groupby(chain).shift(-1).to_numpy()  # rows are season-sorted
    has = ~np.isnan(nxt)
    n_ = np.where(has, nxt, 0).astype(int)
    cut = (season <= win_end) & (season[n_] > win_end)  # NaN window -> False -> no cut
    use = has & ~cut
    mo, Po, mp, Pp = (f[k].to_numpy() for k in ["mo", "Po", "mp", "Pp"])
    m_s, P_s = mo.copy(), Po.copy()
    for y in sorted(np.unique(season))[::-1]:
        j = np.flatnonzero((season == y) & use)
        if len(j) == 0:
            continue
        n = n_[j]
        J = Po[j] * rho[n] ** (season[n] - y) / Pp[n]  # transition into n uses n's group parameters
        m_s[j] = mo[j] + J * (m_s[n] - mp[n])
        P_s[j] = Po[j] + J ** 2 * (P_s[n] - Pp[n])
    return m_s, P_s


def write_output(s, sp):
    """All-season fits (no context = label default; context variant alongside), filtered and smoothed."""
    print("\nall-season fits for output table ...")
    o = s[["player_id", "player", "season", "g", "tm", "age", "snaps", "grade", "z", "mover"]].copy()
    o["entry"] = entry_year(s)
    win_end = (o.entry + 3).to_numpy(float)
    for tag, use_ctx in [("", False), ("_ctx", True)]:
        th, f, _ = fit(s, sp, LAST, use_ctx=use_ctx, log=False)
        o["talent_pre" + tag], o["talent_pre_sd" + tag] = f.mp, np.sqrt(f.Pp)
        o["talent_post" + tag], o["talent_post_sd" + tag] = f.mo, np.sqrt(f.Po)
        ms, Ps = smooth(s, f, th, np.full(len(s), np.nan))
        o["talent_smooth" + tag], o["talent_smooth_sd" + tag] = ms, np.sqrt(Ps)
        mw, Pw = smooth(s, f, th, win_end)
        o["talent_win" + tag], o["talent_win_sd" + tag] = mw, np.sqrt(Pw)
        if use_ctx:
            o["context_term"] = f.cterm
    print(f"  entry year known for {o.entry.notna().mean():.1%} of rows; corr(window, full smooth) "
          f"{o.talent_win.corr(o.talent_smooth):.3f}; corr(window, ctx window) {o.talent_win.corr(o.talent_win_ctx):.3f}")
    o.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(o):,} rows)")


if __name__ == "__main__":
    main()
