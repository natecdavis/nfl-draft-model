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
FEATS = ["grp_c", "fpos_c", "age", "v1", "v2", "g1", "prev_apy", "prev_rookie", "ufa"]
# fine position inside DB and OL (most common OTC contract position): the market pays CBs ~1.12x and safeties
# ~0.89x what a pooled-DB model predicts, tackles ~1.10x / guards 0.96x / centers 0.90x (scripts/cb_s_split_check.py)
FPOS = {"CB": "CB", "S": "S", "LT": "T", "RT": "T", "LG": "G", "RG": "G", "C": "C"}
FPOS_CATS = ["CB", "S", "T", "G", "C", "other"]
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


def fresh_deal_end(contract_history):
    """Last season of the player's current deal when it is a fresh contract (free-agent signing, tag, trade-and-sign:
    year signed + years - 1), else None. Extensions are skipped (OTC counts only the added years, and repeats an
    extension under each team it's traded to) and so are rookie deals (fifth-year options aren't separate entries):
    for those the season rows decide."""
    hist = [x for x in (contract_history if contract_history is not None else [])
            if x.get("contract_type") != "Practice" and x.get("year_signed") is not None]
    if not hist:
        return None
    cur = next((x for x in reversed(hist) if x.get("status") == "Active"), hist[-1])
    if cur.get("contract_type") in ("Extension", "Drafted"):
        return None
    return int(cur["year_signed"]) + int(cur.get("yrs") or 1) - 1


def real_season(x, end=None):
    """A real contract season in an OTC season_history row. Two kinds of void / placeholder year are excluded:
    (1) a base salary above the cap number (99.999999 or 50.0 with a small cap number: A.J. Brown 2030, Arnold
    Ebiketie 2027); (2) seasons after the end of a fresh deal (`end`, fresh_deal_end), which OTC fills with a
    minimum-level base salary that it also counts in the cap number (Travis Kelce, Kirk Cousins, Dallas Goedert 2027 on
    one-year 2026 deals; Deshaun Watson's 2022 deal ends 2026; found against Spotrac's 2027 free-agent list)."""
    b, c = float(x.get("base_salary") or 0.0), float(x.get("cap_number") or 0.0)
    if end is not None and str(x.get("year", "")).isdigit() and int(x["year"]) > end:
        return False
    return b > 0 and c > 0 and b <= c + 0.01


@lru_cache(maxsize=4)
def under_contract(season):
    """One row per player with an active OTC contract: pfr_id (players table, else name match to that season-1's snap
    counts on the same team), team (abbr), OTC position group, and whether he's under contract for `season`.
    Uses the ACTIVE contract row: OTC repeats a player's season history on every contract row, and older rows carry
    shorter histories (DeForest Buckner's 2027 salary is only on his active row). Under contract = a real salary
    (base > 0, not a void year) in `season`; when the active row has no season history at all (most 2026 rookies),
    year signed + years - 1 >= season."""
    import sys
    sys.path.insert(0, str(vt.ROOT / "scripts"))
    import roster_need as rn
    raw = pd.read_parquet(RAW / "contracts.parquet", columns=["player", "gsis_id", "position", "team", "is_active",
                                                              "year_signed", "years", "season_history",
                                                              "contract_history"])
    act = raw[raw.is_active == True].sort_values("year_signed").drop_duplicates(["player", "gsis_id"], keep="last")  # noqa: E712
    nick = dict(zip(*pd.read_parquet(RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS"})
    to_abbr = lambda t: rn.team(nick.get(t, t)) if t else None  # noqa: E731
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    f = RAW / f"snap_counts_{season - 1}.parquet"
    sn = pd.read_parquet(f if f.exists() else RAW / "snap_counts.parquet", columns=["season", "player", "pfr_player_id", "team"])
    sn = sn[sn.season == season - 1].assign(team=lambda d: d.team.map(rn.team)).drop_duplicates(["player", "team"])
    by_name = {(nm, tm): p for nm, tm, p in zip(sn.player, sn.team, sn.pfr_player_id)}
    rows = []
    for nm, g, pos, team, ys, yrs, sh, ch in zip(act.player, act.gsis_id, act.position, act.team, act.year_signed,
                                                  act.years, act.season_history, act.contract_history):
        end = fresh_deal_end(ch)
        years_rows = [x for x in (sh if sh is not None else []) if str(x.get("year", "")).isdigit()]
        cur_team = to_abbr(str(team).split("/")[-1])
        if years_rows:
            nxt = [x for x in years_rows if int(x["year"]) == season]
            signed = bool(nxt) and real_season(nxt[0], end)
            if nxt and nxt[0].get("team"):
                cur_team = to_abbr(nxt[0]["team"])
        else:
            signed = pd.notna(ys) and pd.notna(yrs) and ys + yrs - 1 >= season
        pfr = g2p.get(g) or by_name.get((nm, cur_team))
        rows.append((pfr, nm, cur_team, OTC_TO_G.get(pos), bool(signed), not years_rows))
    return pd.DataFrame(rows, columns=["pfr_id", "player", "team", "g", "signed", "no_history"])


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
def fine_positions():
    """pfr -> CB / S / T / G / C (most common OTC position over his contracts, ties -> latest), else absent."""
    c = pd.read_parquet(RAW / "contracts.parquet", columns=["gsis_id", "position", "year_signed"])
    c = c[c.position.isin(FPOS)].sort_values("year_signed", ascending=False)
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    c = c.assign(pfr=c.gsis_id.map(g2p), fp=c.position.map(FPOS)).dropna(subset=["pfr"])
    return c.groupby("pfr", sort=False).fp.agg(lambda x: x.value_counts().index[0]).to_dict()


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
    fp = fine_positions()
    d["fpos_c"] = pd.Categorical([fp.get(p, "other") if g in ("DB", "OL") else "other" for p, g in zip(d.pfr_id, d.grp)],
                                 categories=FPOS_CATS)
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


def fit(D, half_life=None, ref_year=None):
    """LightGBM per target. half_life (years): weight signings by 0.5 ** ((ref_year - Y) / half_life), recent first
    (scripts/contract_recency.py); None = equal weights."""
    import lightgbm as lgb
    X = D[FEATS]
    w = None
    if half_life:
        ref = ref_year if ref_year is not None else D.Y.max()
        w = 0.5 ** ((ref - D.Y).clip(lower=0) / half_life)
    return {t: lgb.LGBMRegressor(**PARAMS).fit(X, D[t], sample_weight=w) for t in ["log_apy", "yrs", "gshare"]}


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
    uc = under_contract(sign_season)
    out = uc[~uc.signed & uc.pfr_id.notna()].drop_duplicates("pfr_id")[["pfr_id", "team"]].values.tolist()
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
    fa = fa.merge(F[["pfr_id", "age", "prev_apy", "prev_rookie", "grp_c", "fpos_c", "ufa"]], on="pfr_id")
    P = predict(models(lag), fa, SMEAR[lag])
    return pd.concat([fa, P], axis=1)


def rank(fa, sit, cap_price="model", n=4000, team=None, need=False):
    """For one team: each free agent at his predicted contract - titles added, the most he's worth per year at the
    predicted years and guarantee share, and the gap. Same random draws for every player (fair comparison)."""
    skip = vt.LAST + 2 - vt.LAST - 1
    rows = []
    for r in fa.itertuples():
        pl = {"grp": r.grp, "age": r.age - skip, "v1": r.v1, "v2": r.v2, "g1": r.g1}
        yrs = int(r.yrs_pred)
        P = vt.comps().paths(pl, yrs, exclude=False, skip=skip)
        f = 1.0
        if need and team is not None:
            full = rooms()[0]
            f = need_factor(r.grp, full.at[team, r.grp] if team in full.index and r.grp in full.columns else None)
        P = (P[0] * f, P[1])
        e = vt.evaluate(sit, pl, yrs, r.apy_pred, r.gshare_pred * r.apy_pred * yrs, cap_price=cap_price, paths=P, n=n)
        be = vt.break_even(sit, pl, yrs, r.gshare_pred, skip=skip, cap_price=cap_price, paths=P, n=n, tol=0.001)
        rows.append({"pfr_id": r.pfr_id, "titles": e["titles_added"], "break_even": be, "need_factor": f,
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


# ---------------- team needs: how full each position room is for the signing season ----------------
NEED_FILE = OUT / "fa_need_factor.json"  # scripts/fa_need_check.py: slopes, mean fullness, group mean starter-equivalents
OTC_TO_G = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL",
            "RT": "OL", "IDL": "DL", "ED": "EDGE", "LB": "LB", "CB": "DB", "S": "DB"}
ROOM_TO_G = {"RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "IDL": "DL", "ED": "EDGE",
             "LB": "LB", "CB": "DB", "S": "DB", "QB": "QB"}


@lru_cache(maxsize=2)
def rooms(sign_season=vt.LAST + 2):
    """team x group: returning starter-equivalents for sign_season = players under contract with the team that season
    (active OTC contract: base salary > 0 that season; see under_contract) x their snaps per team game in the latest
    season with snap data (2026 so far; last season's if he has played < 3 games), relative to the other teams now
    (1 = typical today). Position = OTC contract position. Before free agency and the draft. Also the players counted."""
    import json
    import sys
    sys.path.insert(0, str(vt.ROOT / "scripts"))
    import roster_need as rn
    def season_se(season):
        f = RAW / f"snap_counts_{season}.parquet"
        s = pd.read_parquet(f if f.exists() else RAW / "snap_counts.parquet")
        s = s[(s.game_type == "REG") & (s.season == season)].assign(team=lambda d: d.team.map(rn.team))
        dp = pd.read_parquet(RAW / "draft_picks.parquet")
        dpos = dict(zip(dp.pfr_player_id, dp.position.map(rn.DRAFT_POS)))
        s["grp"] = [dpos.get(p) or rn.SNAP_POS.get(q) for p, q in zip(s.pfr_player_id, s.position)]
        s["pct"] = np.where(s.grp.isin(rn.DEF), s.defense_pct, s.offense_pct).astype(float)
        s["pct"] = s.pct / (100 if s.pct.max() > 1.5 else 1)
        games = s.groupby("team").game_id.nunique()
        per = s.groupby(["pfr_player_id", "team"]).agg(pct=("pct", "sum"), gp=("game_id", "nunique"),
                                                        grp=("grp", lambda x: x.mode().iloc[0] if x.notna().any() else None)).reset_index()
        per["se"] = per.pct / per.team.map(games)
        g = per.sort_values("pct").groupby("pfr_player_id").grp.last().map(ROOM_TO_G)
        return per.groupby("pfr_player_id").se.sum(), per.groupby("pfr_player_id").gp.sum(), g

    latest = sign_season - 1
    se_now, gp_now, grp = season_se(latest)
    se_prev, _, grp_prev = season_se(latest - 1)
    # players with < 3 games so far this season (injured, suspended, not yet active) keep last season's role
    se = pd.Series({p: (se_now.get(p, 0.0) if gp_now.get(p, 0) >= 3 else max(se_now.get(p, 0.0), se_prev.get(p, 0.0)))
                    for p in set(se_now.index) | set(se_prev.index)})
    grp = grp.combine_first(grp_prev)
    uc = under_contract(sign_season)
    R = uc[uc.signed & uc.pfr_id.notna()].drop_duplicates("pfr_id")[["pfr_id", "team", "g"]].copy()
    # position: the player's active OTC contract (snap-count labels file many edge rushers as LB and some as DL, which
    # left NE with no EDGE); snap label only as a fallback
    R["g"] = [g if g else grp.get(p) for p, g in zip(R.pfr_id, R.g)]
    R["se"] = R.pfr_id.map(se).fillna(0.0)
    R = R[R.g.notna()]
    meta = json.load(open(NEED_FILE))
    gmean = meta["group_mean_se"]
    T = R.groupby(["team", "g"]).se.sum().unstack(fill_value=0.0)
    T = T[[g for g in T.columns if g in gmean]]
    # relative to the other teams now (1 = typical today): only players already under contract count, so absolute
    # levels run below the historical rooms (where unsigned own free agents still counted as returning)
    full = T / T.mean()
    return full, R


def need_factor(g, fullness):
    """Experimental value factor for a signee at group g given his new room's fullness (scripts/fa_room_test.py room
    regression). It did NOT improve past forecasts (scripts/fa_need_check.py), so it's off by default."""
    import json
    meta = json.load(open(NEED_FILE))
    sl = meta["slopes"]
    slope = sl.get(g, sl["other"]) if g in ("OL", "LB") else sl["other"]
    if g == "QB" or fullness is None or pd.isna(fullness):
        return 1.0
    return float(np.clip(1 + slope * (fullness - meta["mean_ret"]), 0.6, 1.25))
