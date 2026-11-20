"""Free-agent market: who is out of contract, and what contract each is likely to get.

Contract model (checked in scripts/contract_predict.py): LightGBM on past unrestricted free-agent signings (OTC
contract_history 'UFA', 2015-2026, APY >= 0.5% of cap), one model each for log APY (share of cap), years, and
guaranteed share of total value. Features known when he hits the market:
  position group, age in the signing season, season value (the draft label's pricing: snap share x pay curve x games)
  in the last two seasons of data, games fraction, APY (share of cap) of his previous contract, whether that was a
  rookie deal.
LAG = seasons between the last season of data and the signing: 1 normally (March after the season), 2 for planning a
year ahead (data through 2025, signing in March 2027) - the model is trained with the same gap it is used with.

Out of contract in SIGN_SEASON: active players with no real salary that season in OTC season_history (a row with
base salary > 0; void years have none). Contract `years` is not used: OTC counts extensions inconsistently.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

import veteran as vt

RAW, OUT = vt.RAW, vt.OUT
FEATS = ["grp_c", "age", "v1", "v2", "g1", "prev_apy", "prev_rookie", "ufa"]
KINDS = ("UFA",)  # contract types trained on; ufa = 1 for UFA rows (predictions use ufa = 1)
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=25, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
MIN_APY = 0.005
# exp(predicted log APY) is a typical (median) contract; the average is higher. Scale to the mean, measured out of
# sample (scripts/contract_predict.py, leave-one-year-out: actual / predicted 1.128 lag 1, 1.185 lag 2; median 1.0)
SMEAR = {1: 1.128, 2: 1.185}


@lru_cache(maxsize=1)
def _contracts():
    c = pd.read_parquet(RAW / "contracts.parquet")
    cap = (c.apy / c.apy_cap_pct).replace([np.inf], np.nan).groupby(c.year_signed).median()
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    c = c[c.gsis_id.map(g2p).notna() & c.contract_history.notna()].copy()
    c["pfr_id"] = c.gsis_id.map(g2p)
    c = c.assign(n=c.contract_history.map(len)).sort_values("n").drop_duplicates("gsis_id", keep="last")
    return c, cap


@lru_cache(maxsize=1)
def history():
    """pfr_id -> list of (year_signed, type, apy share of cap, years, guarantees share of cap, team)."""
    c, cap = _contracts()
    out = {}
    for p, hist in zip(c.pfr_id, c.contract_history):
        rows = []
        for x in hist:
            y = x.get("year_signed")
            if y is None or x.get("contract_type") == "Practice" or x.get("apy") is None or int(y) not in cap.index:
                continue
            y = int(y)
            rows.append((y, x.get("contract_type"), x["apy"] / cap[y], int(x.get("yrs") or 1),
                         (x.get("guarantees") or 0) / cap[y], x.get("team")))
        out[p] = sorted(rows, key=lambda r: r[0])
    return out


@lru_cache(maxsize=1)
def _values():
    v = vt.season_table()
    V = v.set_index(["pfr_id", "season"])
    grp = v.sort_values("season").groupby("pfr_id").group.last().to_dict()
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    born = dict(zip(pl.pfr_id, pd.to_datetime(pl.birth_date, errors="coerce")))
    return V.value_ours.to_dict(), V.gfrac.to_dict(), grp, born


def features(pfr_ids, Y, lag):
    """Feature rows for players entering the market for season Y, data through season Y - lag."""
    V, G, grp, born = _values()
    H = history()
    rows = []
    for p in pfr_ids:
        last = Y - lag
        prev = [h for h in H.get(p, []) if h[0] < Y]
        b = born.get(p)
        rows.append({"pfr_id": p, "Y": Y, "grp": grp.get(p),
                     "age": (pd.Timestamp(f"{Y}-09-01") - b).days / 365.25 if b is not None and pd.notna(b) else np.nan,
                     "v1": V.get((p, last), 0.0), "v2": V.get((p, last - 1), 0.0), "g1": G.get((p, last), 0.0),
                     "prev_apy": max(prev[-1][2], MIN_APY) if prev else MIN_APY,
                     "prev_rookie": int(bool(prev) and prev[-1][1] in ("Drafted", "UDFA"))})
    d = pd.DataFrame(rows)
    d["grp_c"] = pd.Categorical(d.grp, categories=GROUPS)
    d["ufa"] = 1
    return d


def training(lag, years=(2015, 2026), kinds=None):
    """Past signings (KINDS) with features and outcomes (apy, years, guaranteed share of total)."""
    kinds = kinds or KINDS
    rows = []
    for p, hist in history().items():
        for (y, kind, apy, yrs, gtd, team) in hist:
            if kind in kinds and years[0] <= y <= years[1] and apy >= MIN_APY:
                rows.append((p, y, apy, yrs, gtd, team, int(kind == "UFA")))
    S = pd.DataFrame(rows, columns=["pfr_id", "Y", "apy", "yrs", "gtd", "team", "ufa"]).drop_duplicates(["pfr_id", "Y"], keep="last")
    F = pd.concat([features(g.pfr_id, Y, lag) for Y, g in S.groupby("Y")], ignore_index=True)
    D = S.merge(F.drop(columns="ufa"), on=["pfr_id", "Y"])
    D = D[D.grp.notna() & D.age.notna() & ((D.v1 > 0) | (D.v2 > 0))]
    D["log_apy"] = np.log(D.apy)
    D["gshare"] = (D.gtd / (D.apy * D.yrs)).clip(0, 1)
    return D.reset_index(drop=True)


def fit(D):
    import lightgbm as lgb
    X = D[FEATS]
    return {t: lgb.LGBMRegressor(**PARAMS).fit(X, D[t]) for t in ["log_apy", "yrs", "gshare"]}


def predict(models, F, smear=1.0):
    X = F[FEATS]
    return pd.DataFrame({"apy_pred": smear * np.exp(models["log_apy"].predict(X)),
                         "apy_typical": np.exp(models["log_apy"].predict(X)),
                         "yrs_pred": np.clip(np.round(models["yrs"].predict(X)), 1, 5).astype(int),
                         "gshare_pred": np.clip(models["gshare"].predict(X), 0, 1)}, index=F.index)


@lru_cache(maxsize=2)
def models(lag):
    return fit(training(lag))


def free_agents(sign_season=vt.LAST + 2):
    """Active players with no real salary in sign_season, and value in the last two seasons of data."""
    c, _ = _contracts()
    raw = pd.read_parquet(RAW / "contracts.parquet", columns=["gsis_id", "is_active", "team", "year_signed"])
    act = raw[raw.is_active == True].sort_values("year_signed").drop_duplicates("gsis_id", keep="last")  # noqa: E712
    active_team = dict(zip(act.gsis_id, act.team))
    c = c[c.gsis_id.isin(active_team)].assign(is_active=True, team=lambda d: d.gsis_id.map(active_team))
    out = []
    for p, active, sh, team in zip(c.pfr_id, c.is_active, c.season_history, c.team):
        if not active or sh is None:
            continue
        signed = any(str(x.get("year")) == str(sign_season) and (x.get("base_salary") or 0) > 0 for x in sh)
        if not signed:
            out.append((p, team))
    fa = pd.DataFrame(out, columns=["pfr_id", "otc_team"])
    cur = vt.current_players()
    fa = fa.merge(cur, on="pfr_id")
    # restricted free agents: 3 or fewer seasons with snaps (accrued seasons not in the data; this is a proxy)
    V = vt.season_table()
    n = V[V.share > 0].groupby("pfr_id").season.nunique()
    first = V.groupby("pfr_id").season.min()
    fa["seasons"] = fa.pfr_id.map(n).fillna(0).astype(int)
    fa["restricted"] = (sign_season - fa.pfr_id.map(first)) <= 3
    lag = sign_season - vt.LAST
    F = features(fa.pfr_id, sign_season, lag)
    fa = fa.merge(F[["pfr_id", "age", "prev_apy", "prev_rookie", "grp_c", "ufa"]], on="pfr_id")
    P = predict(models(lag), fa, SMEAR[lag])
    return pd.concat([fa, P], axis=1)


def rank(fa, sit, cap_price="model", n=4000):
    """For one team: each free agent at his predicted contract - titles added, the most he's worth per year at the
    predicted years and guarantee share, and the gap. Same random draws for every player (fair comparison)."""
    skip = vt.LAST + 2 - vt.LAST - 1
    rows = []
    for r in fa.itertuples():
        pl = {"grp": r.grp, "age": r.age - skip, "v1": r.v1, "v2": r.v2, "g1": r.g1}
        yrs = int(r.yrs_pred)
        P = vt.comps().paths(pl, yrs, exclude=False, skip=skip)
        e = vt.evaluate(sit, pl, yrs, r.apy_pred, r.gshare_pred * r.apy_pred * yrs, cap_price=cap_price, paths=P, n=n)
        be = vt.break_even(sit, pl, yrs, r.gshare_pred, skip=skip, cap_price=cap_price, paths=P, n=n, tol=0.001)
        rows.append({"pfr_id": r.pfr_id, "titles": e["titles_added"], "break_even": be,
                     "older_than_comps": pl["age"] - vt.comps().comp_age(pl, need=yrs + skip) > 1.5,
                     "value_yr": float(e["value"].mean()), "value_y1": float(e["value"][0])})
    out = fa.merge(pd.DataFrame(rows), on="pfr_id")
    out["gap"] = out.break_even - out.apy_pred
    return out


def notes(r):
    """Short caveats for a ranked row."""
    n = []
    if r.restricted:
        n.append("likely restricted FA")
    if r.prev_rookie and r.v1 >= 0.06:
        n.append("likely extended/tagged; price could run well above")
    elif r.v1 >= 0.08:
        n.append("tag candidate")
    if r.older_than_comps:
        n.append("older than comps: forecast high")
    if r.grp == "QB":
        n.append("QB: upper end")
    return "; ".join(n)
